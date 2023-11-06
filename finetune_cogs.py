import os
import argparse
import logging
import math
import copy

import numpy as np

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, GenerationConfig
from datasets import load_dataset

import torch

from tqdm.auto import tqdm

import lightning as L


bad_keys = [
    "transformer.h.0.attn.bias",
    "transformer.h.0.attn.masked_bias",
    "transformer.h.1.attn.bias",
    "transformer.h.1.attn.masked_bias",
    "transformer.h.2.attn.bias",
    "transformer.h.2.attn.masked_bias",
    "transformer.h.3.attn.bias",
    "transformer.h.3.attn.masked_bias",
    "transformer.h.4.attn.bias",
    "transformer.h.4.attn.masked_bias",
    "transformer.h.5.attn.bias",
    "transformer.h.5.attn.masked_bias",
    "transformer.h.6.attn.bias",
    "transformer.h.6.attn.masked_bias",
    "transformer.h.7.attn.bias",
    "transformer.h.7.attn.masked_bias",
    "transformer.h.8.attn.bias",
    "transformer.h.8.attn.masked_bias",
    "transformer.h.9.attn.bias",
    "transformer.h.9.attn.masked_bias",
    "transformer.h.10.attn.bias",
    "transformer.h.10.attn.masked_bias",
    "transformer.h.11.attn.bias",
    "transformer.h.11.attn.masked_bias",
]


def load_model(fabric, model, load_dir):
    state_dict = fabric.load(load_dir)["model"]
    for key in bad_keys:
        del state_dict[key]
    model.load_state_dict(state_dict, strict=False)

def strip_pad_tokens(output, pad_token_id):
    return [token for token in output if token != pad_token_id]


# exact match: just use == on tokenized
def soft_f1_score(output, gold_answer):
    # Tokenize the output and gold answer into words
    output_tokens = set(output.split())
    gold_tokens = set(gold_answer.split())

    # Calculate precision and recall
    common_tokens = output_tokens.intersection(gold_tokens)
    precision = len(common_tokens) / len(output_tokens) if len(output_tokens) > 0 else 0
    recall = len(common_tokens) / len(gold_tokens) if len(gold_tokens) > 0 else 0

    # Calculate the F1 score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def cos_schedule(
    it, warmup_iters=2000, lr_decay_iters=600000, learning_rate=6e-4, min_lr=6e-5
):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# def inner_training_loop(trial, **config):
def train(**config):
    num_nodes = config["num_nodes"]
    seed = config["seed"]
    num_workers = config["num_workers"]
    learning_rate = config["lr"]
    weight_decay = config["weight_decay"]
    output_dir = config["output_dir"]
    epochs = config["epochs"]
    load_model_dir = config["load_model_dir"]
    warmup_iters = config["warmup_iters"]
    bsz = config["bsz"]
    val_set_size = config["val_set_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]

    MAX_LENGTH = 400

    fabric = L.Fabric(
        accelerator="auto",
        strategy="ddp",
        num_nodes=num_nodes,
        devices=-1,
        precision="16-mixed",
    )
    fabric.launch()
    fabric.seed_everything(seed)
    fabric.print(config)

    if fabric.global_rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # load model without pretraining weights
    model_config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(model_config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    generation_config = GenerationConfig.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if load_model_dir is not None:
        load_model(fabric, model, load_model_dir)


    def tokenize_and_pad_left(data_point, max_length=None):
        input_text = data_point["input"]
        output_text = data_point["output"]

        # Tokenize input and output text separately
        input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
        output_tokens = tokenizer.encode(output_text, add_special_tokens=True)

        # Calculate the combined length
        combined_length = len(input_tokens) + len(output_tokens)

        # Determine the target max_length, using max_length if provided or combined_length if not
        target_max_length = max_length if max_length is not None else combined_length

        # Create the tokenized sequences with left padding
        input_ids = [tokenizer.pad_token_id] * (target_max_length - combined_length) + input_tokens + output_tokens

        # Create attention mask
        attention_mask = np.zeros(target_max_length, dtype=np.int64)
        attention_mask[target_max_length - combined_length:] = 1

        labels = np.full((target_max_length,), -100, dtype=np.int64)
        labels[target_max_length - combined_length:target_max_length - len(input_tokens)] = np.array(output_tokens, dtype=np.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        
    def collate_fn(tokenized_samples):
        # Create a batch dictionary to hold the tensors
        batch = {}

        # Get a list of keys from the first sample
        keys = tokenized_samples[0].keys()
        # Loop through each key (e.g., 'input_ids', 'attention_mask', etc.)
        for key in keys:
            # Stack the tensors along a new batch dimension (dim=0)
            batch[key] = torch.stack([torch.tensor(sample[key], dtype=torch.long) for sample in tokenized_samples], dim=0)

        return batch

    def test_collate_fn(tokenized_samples):
        # Create a batch dictionary to hold the tensors
        batch = {}

        # Get a list of keys from the first sample
        keys = tokenized_samples[0].keys()
        # Loop through each key (e.g., 'input_ids', 'attention_mask', etc.)
        for key in ['input_ids', 'attention_mask', 'labels']:
            # Stack the tensors along a new batch dimension (dim=0)
            batch[key] = torch.stack([torch.tensor(sample[key], dtype=torch.long) for sample in tokenized_samples], dim=0)

        batch['output'] = [sample['output'] for sample in tokenized_samples]
        return batch


    
    model = fabric.setup_module(model)

    data = load_dataset(
        "json", data_files={"train": "data/train.json", "test": "data/test.json"}
    )

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(lambda data: tokenize_and_pad_left(data, max_length=MAX_LENGTH))
    val_data = train_val["test"].shuffle().map(lambda data: tokenize_and_pad_left(data, max_length=MAX_LENGTH))
    test_data = data["test"].shuffle().map(lambda data: tokenize_and_pad_left(data, max_length=MAX_LENGTH // 2))

    train_data = train_data.remove_columns(['instruction', 'input', 'gen_type', 'output'])
    val_data = val_data.remove_columns(['instruction', 'input', 'gen_type', 'output'])

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=bsz,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=bsz,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=bsz,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_collate_fn
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    optimizer = fabric.setup_optimizers(optimizer)

    train_dataloader, val_dataloader, test_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader, test_dataloader
    )

    steps = len(train_dataloader) * epochs
    progress_bar = tqdm(
        range(steps),
        desc="pt",
        disable=(not fabric.global_rank == 0),
    )

    iter_num = 1

    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            is_accumulating = iter_num % gradient_accumulation_steps != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                loss = model(**batch).loss
                fabric.backward(loss)
            if not is_accumulating:
                lr = cos_schedule(
                    iter_num,
                    warmup_iters=warmup_iters,
                    lr_decay_iters=steps,
                    learning_rate=learning_rate,
                    min_lr=learning_rate / 10,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()

            if fabric.global_rank == 0:
                progress_bar.update(1)
                logging.info(
                    "Epoch %s, pt batch %s, loss: %s",
                    epoch,
                    iter_num,
                    loss.item(),
                )
            iter_num += 1

            if iter_num >= steps:
                break

        fabric.save(
            os.path.join(output_dir, f"pretrained_model_{epoch}.pt"),
            state={"model": model},
        )

        # compute final pretraining loss on the validation set
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                output = model(**batch)
                loss = output.loss
                eval_loss += loss.item() 

        eval_loss = eval_loss / len(test_dataloader)

        if fabric.global_rank == 0:
            logging.info("Pretrain loss: %s, Epoch: %s", eval_loss, epoch)

        # compute exact match and f1 accuracy
        exact_accuracy = 0
        soft_f1 = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                outputs = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=100,
                    pad_token_id=50256,
                    generation_config=generation_config,
                )

                generated_tokens = outputs[:, -100:]
                text = tokenizer.batch_decode(generated_tokens)
                
                for i, (text, gold_answer) in enumerate(zip(text, batch["output"])):
                    if i == 0:
                        print(f"Generated: {text}")
                        print(f"Gold: {gold_answer}")
                    exact_accuracy += text == gold_answer
                    soft_f1 += soft_f1_score(text, gold_answer)
                
        exact_accuracy /= len(test_dataloader)
        soft_f1 /= len(test_dataloader)

        logging.info("Epoch %d exact accuracy: %f. f1: %f", epoch, exact_accuracy, soft_f1)


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the desired logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()  # Log messages will be displayed in the console
        ],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--split_pct", type=int, default=1)
    parser.add_argument("--save_init", action="store_true")
    parser.add_argument("--load_model_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--val_set_size", type=int, default=256)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bsz", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()

# this file preprocess the hh dataset
# specifically, it merges the helpfulness split, and subsamples 15k data points. For reproducibility, the seed is set to 42.
import random
import argparse
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--store_dir", type=str, default="dataset")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    return parser.parse_args()

# get the helpful splits
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

if __name__ == '__main__':
    args = parser()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ## Length dataset ##
    
    print(f"Loading Length Bias Dataset, and storing them into {args.store_dir}/length_dataset")
    dataset_length = load_dataset("Taywon/HH_length_biased_15k")

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    dataset_length = dataset_length.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    dataset_length = dataset_length.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.max_length
        and len(x["input_ids_rejected"]) <= args.max_length
    )
    flipped_indices = dataset_length['train']['flipped']
    flipped_indices = np.where(flipped_indices)[0]
    dataset_length.save_to_disk(f"{args.store_dir}/length_dataset")
    np.save(f"{args.store_dir}/length_dataset/train/flipped_indices.npy", flipped_indices)
    ## Sycophancy dataset ##
    print(f"Loading Sycophancy Bias Dataset, and storing them into {args.store_dir}/sycophancy_dataset")
    dataset_sycophancy = load_dataset("Taywon/HH_sycophancy_biased_15k")

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    dataset_sycophancy = dataset_sycophancy.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    dataset_sycophancy = dataset_sycophancy.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.max_length
        and len(x["input_ids_rejected"]) <= args.max_length
    )
    flipped_indices = dataset_sycophancy['train']['flipped']
    flipped_indices = np.where(flipped_indices)[0]
    dataset_sycophancy.save_to_disk(f"{args.store_dir}/sycophancy_dataset")
    np.save(f"{args.store_dir}/sycophancy_dataset/train/flipped_indices.npy", flipped_indices)
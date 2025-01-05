# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \
"""

import torch
import os
import datasets
from datasets import load_dataset, DatasetDict, concatenate_datasets
from accelerate import Accelerator
from tqdm import tqdm
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from src.reward_modeling.utils import compute_accuracy
from trl import ModelConfig, RewardConfig, RewardTrainer
from src.reward_modeling.utils import MyRewardConfig

tqdm.pandas()


if __name__ == "__main__":
    parser = H4ArgumentParser((ModelArguments, DataArguments, MyRewardConfig))
    model_config, data_args, reward_config = parser.parse()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    # setup the torch_dtype and quantization_config for model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # load the toknizer and model from model_config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, attn_implementation = "flash_attention_2",
        **model_kwargs
    )
    
    # define pad_tokens for both tokenizer and model if they do not exist.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ################
    # Load Dataset
    ################
    
    print(f"Loading train dataset from {reward_config.train_datapath}")
    train_dataset = datasets.load_from_disk(f"{reward_config.train_datapath}")
    print(f"Loading eval dataset from {reward_config.eval_datapath}")
    eval_dataset = datasets.load_from_disk(reward_config.eval_datapath)
    
    ################
    # Training
    ################
    
    accelerator = Accelerator()
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
        compute_metrics=compute_accuracy,
    )
    trainer.train()
    trainer.save_model(reward_config.output_dir)
    accelerator.wait_for_everyone()
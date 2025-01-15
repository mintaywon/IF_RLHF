# Understanding Impact of Human Feedback via Influence Functions

<p align="center">
  <img src="assets/intro.png" alt="Introductory Figure" width="1000">
</p>

--- 
We provide a codebase for "[Understanding Impact of Human Feedback via Influence Functions](https://arxiv.org/abs/2501.05790)". Our work utilizes influence functions to measure the impact of human feedback on the performance of reward models. In this repository, we provide source code to replicate our work, specifically length/sycophancy bias detection.
## Install
---
### create conda environment
```bash
conda create -n if_rlhf python=3.10 absl-py pyparsing pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
echo "export IF_RLHF_HOME=/path/to/current/directory" >> ~/.bashrc
source ~/.bashrc
conda activate if_rlhf
```
### check gpu
```python
import torch
torch.cuda.is_available() # should show True
```
### install the remaining package dependencies
```bash
python -m pip install -e .
pip install -r requirements.txt
```
### install flash attention
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```
### for deepspeed
```bash
conda install -c conda-forge mpi4py mpich
```

## Datasets
First prepare length and sycophancy biased datasets (15k subset of Anthropic/HH-rlhf dataset).
```bash
cd $IF_RLHF_HOME
mkdir dataset
python src/reward_modeling/make_dataset.py
```

## Reward Modeling
### Reward Modeling using length biased dataset
example script for training reward model (based on LLama3-8B) on length biased dataset
```bash
CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port=1231 src/reward_modeling/reward_modeling.py recipes/reward_modeling/Llama-3-8B_length.yaml
```

### Reward Modeling using sycophancy biased dataset
```bash
CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port=1231 src/reward_modeling/reward_modeling.py recipes/reward_modeling/Llama-3-8B_sycophancy.yaml
```

## Influence Computation
### Cache Gradients
- Length bias
```bash
CUDA_VISIBLE_DEVICES=0 python src/influence/cache_gradients.py \
--model_path "logs/Llama-3-8B_length" \
--data_path "dataset/length_dataset/train" \
--save_name "rapid_grad_train.pt" \
--seed 42

CUDA_VISIBLE_DEVICES=0 python src/influence/cache_gradients.py \
--model_path "logs/Llama-3-8B_length" \
--data_path "dataset/length_dataset/test" \
--save_name "rapid_grad_val.pt" \
--seed 42
```
- Sycophancy bias
```bash
CUDA_VISIBLE_DEVICES=0 python src/influence/cache_gradients.py \
--model_path "logs/Llama-3-8B_sycophancy" \
--data_path "dataset/sycophancy_dataset/train" \
--save_name "rapid_grad_train.pt" \
--seed 42

CUDA_VISIBLE_DEVICES=0 python src/influence/cache_gradients.py \
--model_path "logs/Llama-3-8B_sycophancy" \
--data_path "dataset/sycophancy_dataset/test" \
--save_name "rapid_grad_val.pt" \
--seed 42
```

### Compute Influence Functions using DataInf
Follow the scripts in measure_length_bias.ipynb and measure_sycophancy_bias.ipynb to compute influence values and plot reciever operating characteristics curves.

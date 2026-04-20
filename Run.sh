#!/bin/bash
set -x; set -e
nvidia-smi
export OMP_NUM_THREADS=1

# Toy example: training BUSI data with device augmentation
# torchrun --nproc_per_node=1 Main.py --config_file Configs/Toy_BUSGen_BUSI_DevAug.json

# Sampling from toy BUSI trained generative model
# torchrun --nproc_per_node=1 Main.py --config_file Configs/Toy_BUSGen_BUSI_DevAug.json --eval

# Sampling from BUSGen model
torchrun --nproc_per_node=1 Main.py --config_file Configs/BUSGen_Sampling.json --eval

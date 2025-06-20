#!/bin/bash

NUM_GPUS=$(nvidia-smi -L | wc -l)
mkdir -p ./results/ddp

python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \
  --master_port=29500 \
  src/main.py \
  --result_dir ./results/ddp \
  --verify_path /absolute/path/to/Innolux_verify.pt \
  --enroll_path /absolute/path/to/Innolux_enroll.pt \
  --epochs 20 \
  --batch_size 64 \
  --learning_rate 1e-4 \
  --use_ddp True

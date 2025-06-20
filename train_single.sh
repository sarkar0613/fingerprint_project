#!/bin/bash

mkdir -p ./results/single_gpu

python src/main.py \
  --result_dir ./results/single_gpu \
  --verify_path /absolute/path/to/Innolux_verify.pt \
  --enroll_path /absolute/path/to/Innolux_enroll.pt \
  --epochs 20 \
  --batch_size 64 \
  --learning_rate 1e-4 \
  --use_ddp False

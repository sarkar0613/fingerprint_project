#!/bin/bash

python src/main.py \
  --result_dir ./results/single_gpu \
  --verify_path /absolute/path/to/Innolux_verify_fe.pt \
  --enroll_path /absolute/path/to/Innolux_enroll_fe.pt \
  --batch_size 128 \
  --epochs 20 \
  --use_stn True \
  --use_ddp False

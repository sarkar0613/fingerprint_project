#!/bin/bash

# 限制只使用 GPU 0 和 1
#export CUDA_VISIBLE_DEVICES=0,1
# 設定使用的 GPU 數量（這裡就是 2）
#NUM_GPUS=2

NUM_GPUS=$(nvidia-smi -L | wc -l)

python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \
  --master_port=58889 \
  src/main.py \
  --result_dir ./results/ddp \
  --verify_path /absolute/path/to/Innolux_verify_fe.pt \
  --enroll_path /absolute/path/to/Innolux_enroll_fe.pt \
  --batch_size 256 \
  --epochs 20 \
  --use_stn True \
  --use_ddp True \
  --dist_url tcp://localhost:58889

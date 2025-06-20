import os
import argparse
import logging
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

from train.loss import ContrastiveLoss
from models.model_init import create_siamese_network
from train.trainer import SiameseTrainer
from data.dataloader import prepare_data, create_data_loaders
from utils.utils import set_seed

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Fingerprint Finetune')
    parser.add_argument('--result_dir', type=str, default='/data/nas05/paul/fingerprint_project/src/result/ablation/stn')
    parser.add_argument('--verify_path', type=str, default='/data/nas05/paul/preprocessing/innolux/Innolux_verify_fe.pt')
    parser.add_argument('--enroll_path', type=str, default='/data/nas05/paul/preprocessing/innolux/Innolux_enroll_fe.pt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--use_stn', default=True, help='Use stn inputs')
    parser.add_argument('--use_ddp', type=bool, default=True, help='Enable DistributedDataParallel')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:58889')
    
    return parser.parse_args()


def setup_directories(result_dir):
    checkpoint_dir = os.path.join(result_dir, 'ckpts_0')
    model_dir = os.path.join(result_dir, 'models_0')
    log_dir = os.path.join(result_dir, 'logs_0')
    log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d")}.txt')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return checkpoint_dir, model_dir, log_file

def cleanup_ddp():
    dist.barrier()
    dist.destroy_process_group()

def main():
    set_seed(42)
    args = parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node
    args.rank = 0  # master

    args.checkpoint_dir, args.model_dir, args.log_file = setup_directories(args.result_dir)
    setup_logging(args.log_file)

    mp.spawn(main_worker, args=(vars(args),), nprocs=args.ngpus_per_node)

def main_worker(gpu, args_dict):
    args = argparse.Namespace(**args_dict)
    args.rank += gpu
    torch.cuda.set_device(gpu)

    dist.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )

    device = torch.device(f"cuda:{gpu}")
    logging.info(f"Process {args.rank} using GPU {gpu}")

    # Load data
    data_verify = torch.load(args.verify_path)
    data_enroll = torch.load(args.enroll_path)
    imgs, labels = prepare_data(data_verify, data_enroll)

    # Normalization
    if args.use_fe:
        mean, std = [0.4998] * 3, [0.1099] * 3
    else:
        mean, std = [0.4062] * 3, [0.3087] * 3

    train_loader, test_loader = create_data_loaders(
        imgs, labels, mean, std,
        batch_size=args.batch_size,
        world_size=args.world_size,
        rank=args.rank
    )

    # Model
    model = create_siamese_network(args.use_fe).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    # Optimizer
    optimizer = optim.Adam([
        {'params': model.module.backbone_model.parameters(), 'lr': 1e-3},
        {'params': model.module.stn_model.parameters(), 'lr': 5e-4}
    ])

    # Criterion & Scheduler
    criterion = ContrastiveLoss(margin=2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=(args.rank == 0)
    )

    # Trainer
    trainer = SiameseTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir,
        log_file=args.log_file,
        rank=args.rank
    )

    trainer.run_training(epochs=args.epochs)
    cleanup_ddp()

if __name__ == "__main__":
    main()

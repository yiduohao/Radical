#!/usr/bin/env python

# Based on MoCo codebase

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import time
import datetime
from datetime import datetime

import moco.builder
import moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from radatron.data.dataset import RadatronDataset
from radatron.config import setup
from detectron2.engine import default_argument_parser
import code
from detectron2.modeling import ShapeSpec
from radatron.data import SSLRadatronMapper
from torch import distributed as dist
import wandb
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
import scipy
import matplotlib.pyplot as plt
import gc


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

# Radical Settings
parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
parser.add_argument("--wandb_name", type=str, help="name of wandb run")
parser.add_argument("--use_radical", action="store_true", help="use radical")
parser.add_argument("--output_dir", default="/mnt/sens_data1/yiduo/output/latest", help="path to output folder")
parser.add_argument("--single_gpu", action="store_true", help="use a single gpu for debugging")
parser.add_argument("--source_folder", default="CLIP_Left", type=str, choices=["CLIP_Left", "CLIP_Left_avg", "CLIP_Left_pad", "CLIP_Left_resize", "CLIP_Left_newpad", "VIT_Left_resize"], help="the name of the source folder for ssl")
parser.add_argument("--in_batch_loss", action="store_true", help="use in batch contrastive loss")
parser.add_argument("--intra_weight", default=2, type=float, help="weight for intra loss")
parser.add_argument("--symmetric_loss", action="store_true", help="use symmetric loss in MoCo")
parser.add_argument("--symmetric_loss_version", default=0, type=int, help="version of symmetric loss in MoCo")
parser.add_argument("--intra_only", action="store_true", help="only use intra-modal loss")

# Radar Augmentation Settings
parser.add_argument("--rw_binary", default=0.9, type=float, help="rw_binary")
parser.add_argument("--rw_phase", default=0.1, type=float, help="rw_phase")
parser.add_argument("--no_rw_lowres", action="store_true", help="only use intra-modal loss")
parser.add_argument("--h_flip", action="store_true", help="use horizontal flip with 0.5 prob")
parser.add_argument("--crop", action="store_true", help="use crop")
parser.add_argument("--rot", action="store_true", help="use rot")


# Radatron config
parser.add_argument("--config", default="", metavar="FILE", help="path to config file")
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
)

APPLY_AUGS_LOW_RES = True
LOW_RES_INDICES = [0, 1, 2, 3, 11, 12, 13, 14, 46, 47, 48, 49, 50, 51, 52, 53]
N = 256
# VALS_PHASE = np.linspace(-np.pi, np.pi, N+1)
THRESHOLDING = 0

def main():
    args = parser.parse_args()

    args2 = None
    if args.use_radical:
        args2 = default_argument_parser().parse_args()

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        if args.output_dir[-1] != "/":
            args.output_dir += "/"
    

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    # if args.use_wandb:
    #     wandb.login(key='f9e79b66fbf56456d8f9542c2cd849f88a62a08d')
    #     wandb.init(
    #         entity="davidhao",
    #         project='Radatron-MoCo',
    #         # id=args.wandb_name,  # set id as wandb_name for resume
    #         name=args.wandb_name,
    #         sync_tensorboard=False,
    #     )

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node: ", ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, args2))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, args2)


def main_worker(gpu, ngpus_per_node, args, args2):
    args.gpu = gpu

    if args.use_radical:
        cfg = setup(args2)
    
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    
    if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
    ):
        if args.use_wandb:
            wandb.login(key='f9e79b66fbf56456d8f9542c2cd849f88a62a08d')
            wandb.init(
                entity="davidhao",
                project='Radatron-MoCo',
                # id=args.wandb_name,  # set id as wandb_name for resume
                name=args.wandb_name,
                # sync_tensorboard=False,
            )
    # create model
    if args.use_radical:
        print("=> creating model Radatron Backbone Encoder")
        input_shape = ShapeSpec(channels=1)
        model = moco.builder.MoCo(
            models.__dict__[args.arch],
            args.moco_dim,
            args.moco_k,
            args.moco_m,
            args.moco_t,
            args.mlp,
            cfg=cfg,
            input_shape=input_shape,
            single_gpu=args.single_gpu,
            in_batch_loss=args.in_batch_loss,
            # intra_weight=args.intra_weight,

        )
    else:
        raise NotImplementedError("Only Radical is supported.")
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        if not args.single_gpu:
            raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        if not args.single_gpu:
            raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    if not args.use_radical:
        raise NotImplementedError("Only Radical is supported.")
    else:
        train_dataset = RadatronDataset(cfg, val=False, ssl=False, args=args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        print("Epoch: ", epoch)
        print("args.gpu", args.gpu)
        print("args.gpu", args.gpu)
        print("args.gpu", args.gpu)
        print("args.rank", args.rank)
        print("args.rank", args.rank)
        print("args.rank", args.rank)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            if ((epoch+1) % 50 == 0) or (epoch == args.epochs - 1):
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=False,
                    filename=args.output_dir + "checkpoint_{:04d}.pth.tar".format(epoch),
                )


def main_RMM(X, args):

    data = X.clone()

    bs = data.shape[0]
    shape = (bs, 1, 1, 86)
    binary_mask = torch.from_numpy(np.random.binomial(1, args.rw_binary, size=shape).astype(np.int32)).cuda(args.gpu, non_blocking=True)
    low_res_mask = torch.from_numpy(np.zeros(shape).astype(np.int32)).cuda(args.gpu, non_blocking=True)
    low_res_mask[:, : , :, LOW_RES_INDICES] = 1
    random_phase = torch.from_numpy(np.exp(1j * 2 * np.pi * args.rw_phase * np.random.rand(*shape)).astype(np.complex64)).cuda(args.gpu, non_blocking=True)


    random_array =  (binary_mask * random_phase)

    X1 = (torch.squeeze(torch.abs(torch.sum(data * random_array, axis=-1)))  / args.rw_binary) / 86 # [32, 512, 192]

    if args.no_rw_lowres:
        # X2 = np.squeeze(np.abs(np.sum(data * low_res_mask, axis=-1))) / len(LOW_RES_INDICES)
        X2 = (data * low_res_mask).sum(-1).abs() / len(LOW_RES_INDICES)
    else:
        # X2 = np.squeeze(np.abs(np.sum(data * low_res_mask * random_array, axis=-1))) / len(LOW_RES_INDICES)
        # X2 = ((data * low_res_mask * random_array).sum(-1).abs() / args.rw_binary) / len(LOW_RES_INDICES)
        X2 = (torch.squeeze(torch.abs(torch.sum(data * low_res_mask * random_array, axis=-1)))  / args.rw_binary) / len(LOW_RES_INDICES) # [32, 512, 192]
    

    # #Apply preprocessing
    X1 = main_preprocess_input(X1, stream=1, args=args).unsqueeze(1) # [32, 1, 448, 192]
    X2 = main_preprocess_input(X2, stream=2, args=args).unsqueeze(1) # [32, 1, 448, 192]

    # Apply augmentation

    X1_list = []
    X2_list = []

    for i in range(bs):
        image1 = X1[i] # [1, 448, 192]
        image2 = X2[i]
        if args.h_flip:
            if random.random() < 0.5:
                image1 = TF.hflip(image1)
                image2 = TF.hflip(image2)
        if args.crop:
            crop_size = 0.7
            i, j, h, w = transforms.RandomCrop.get_params(image1,
                output_size=(int(448 * crop_size), int(192 * crop_size)))
            image1 = TF.crop(image1, i, j, h, w)
            image2 = TF.crop(image2, i, j, h, w)
        if args.rot:
            if random.random() < 0.5:
                angle = random.randint(-45, 45)
                image1 = TF.rotate(image1, angle)
                image2 = TF.rotate(image2, angle)
        
        X1_list.append(image1)
        X2_list.append(image2)
    
    X1 = torch.stack(X1_list)
    X2 = torch.stack(X2_list)

    return X1, X2

def main_preprocess_input(X, stream, args):
    if stream == 1:
        norm_factor = 1.3
    elif stream == 2:
        norm_factor = 1.4

    X = X[:, 40:488, :]

    max_val = X.max(-1)[0].max(-1)[0]
    # max_val_mask = max_val > norm_factor
    # max_val[max_val_mask] = norm_factor

    X = X / max_val.unsqueeze(-1).unsqueeze(-1)

    return X


def load_compressed(C, vals, P, VALS_PHASE):
    # Load the MATLAB file
    # mat_contents = scipy.io.loadmat(filepath)

    abs_res = reconstruct_variable(C, vals)
    ang_res = reconstruct_variable(P, VALS_PHASE)
    res = abs_res * torch.exp(1j*ang_res)

    return res


# def threshold_heatmap(heatmap):
#     a = np.percentile(heatmap, 100 * THRESHOLDING, axis=-1, keepdims=True)
#     heatmap[heatmap < a] = 0
#     return heatmap

def reconstruct_variable(C, vals):
    # return vals[C.astype(int)]
    return vals[C.to(torch.long)]


def train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss_Cross", ":.4e")
    losses_intra = AverageMeter("Loss_Intra", ":.4e")
    losses_total = AverageMeter("Loss_Total", ":.4e")
    losses_pos = AverageMeter("Loss_Intra", ":.4e")
    losses_neg = AverageMeter("Loss_Total", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    masked_num = AverageMeter("Mask", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_intra, losses_total, top1, top5, masked_num],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (dicts, _) in enumerate(train_loader):


        data_time.update(time.time() - end)

        C = dicts["image"]["C"].cuda(args.gpu, non_blocking=True) # [32, 512, 192, 1, 86] uint8
        vals = dicts["image"]["vals"].cuda(args.gpu, non_blocking=True) # [32, 256] float64
        P = dicts["image"]["P"].cuda(args.gpu, non_blocking=True) # [32, 512, 192, 1, 86] uint8
        VALS_PHASE = torch.linspace(-np.pi, np.pi, N+1).cuda(args.gpu, non_blocking=True)
        # path = dicts['file_name']

        res_list = []
        for _, (c, v, p) in enumerate(zip(C, vals, P)):
            res = load_compressed(c, v, p, VALS_PHASE)
            res_list.append(res)

        res = torch.squeeze(torch.stack(res_list))
        
        clip_feat = dicts["clip"]["feat"]


        if args.gpu is not None:
            clip_feat = clip_feat.cuda(args.gpu, non_blocking=True).to(torch.float32)

        Q1, Q2 = main_RMM(res, args)

        
        K1, K2 = main_RMM(res, args)


        image_q = {}
        image_k = {}

        image_q['image1'] = Q1.detach()
        image_q['image2'] = Q2.detach()
        image_k['image1'] = K1.detach()
        image_k['image2'] = K2.detach()

        # gc.collect()
        # torch.cuda.empty_cache()

        # compute output
        if args.use_radical:
            output, target, output_intra, target_intra, masked_num_batch, logits_intra_undetached, logits_intra_reverse_detached = model(im_q=image_q, im_k=image_k, feat=clip_feat)
        else:
            raise NotImplementedError("Only SSL Radatron is supported.")
        loss = criterion(output, target)
        
        if args.symmetric_loss:
            assert args.symmetric_loss_version != 0
            if args.symmetric_loss_version == 1:
                loss_pos = criterion(output_intra, target)
                loss_neg = criterion(logits_intra_reverse_detached.T, target)
                loss_intra = (loss_pos + loss_neg) / 2
            elif args.symmetric_loss_version == 2:
                loss_pos = criterion(output_intra, target)
                loss_neg = criterion(output_intra.T, target)
                loss_intra = (loss_pos + loss_neg) / 2
            elif args.symmetric_loss_version == 3:
                loss_pos = criterion(logits_intra_undetached, target)
                loss_neg = criterion(logits_intra_undetached.T, target)
                loss_intra = (loss_pos + loss_neg) / 2
        else:
            loss_intra = criterion(output_intra, target_intra)
            loss_pos = loss_intra.new_zeros(1).detach()
            loss_neg = loss_intra.new_zeros(1).detach()

        loss_total = loss + loss_intra * args.intra_weight
        mult = loss.detach().item() / loss_total.detach().item()
        loss_total = loss_total * mult

        if args.intra_only:
            loss_total = loss_intra * args.intra_weight

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), image_q['image1'].size(0))
        losses_intra.update(loss_intra.item(), image_q['image1'].size(0))
        losses_total.update(loss_total.item(), image_q['image1'].size(0))
        losses_pos.update(loss_pos.item(), image_q['image1'].size(0))
        losses_neg.update(loss_neg.item(), image_q['image1'].size(0))
        top1.update(acc1[0], image_q['image1'].size(0))
        top5.update(acc5[0], image_q['image1'].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        loss_total.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if masked_num_batch is not None:
            masked_num.update(masked_num_batch)

        if i % args.print_freq == 0:
            progress.display(i)
        
        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            if args.use_wandb:
                # try:
                log_data = dict(
                    loss=loss.item(),
                    loss_intra=loss_intra.item(),
                    loss_total=loss_total.item(),
                    loss_pos=loss_pos.item(),
                    loss_neg=loss_neg.item(),
                    acc1=acc1[0],
                    acc5=acc5[0],
                    # mask=masked_num_batch,
                )
                log_data = {"train/"+k: v for k, v in log_data.items()}
                # wandb.log(data=log_data, step= int(epoch * len(train_loader) + i))
                wandb.log(data=log_data)
                # except:
                #     pass
    
    
    if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
    ):
        if args.use_wandb:
            log_data_epo = dict(
                epo=int(epoch+1),
                loss=losses.avg,
                loss_intra=losses_intra.avg,
                loss_total=losses_total.avg,
                loss_pos=losses_pos.avg,
                loss_neg=losses_neg.avg,
                acc1=top1.avg,
                acc5=top5.avg,
            )
            log_data_epo = {"epo/"+k: v for k, v in log_data_epo.items()}
            print("wandb: log_data_epo", log_data_epo)
            wandb.log(data=log_data_epo)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()

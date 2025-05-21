#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/09 19:35:41
@Desc    :   
    2025/03/14: dataset QA passed
    2025/03/15: model, inference and train QA
@Ref     :   
    https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/train.py
'''
import argparse
import json
import logging
import math
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoTransformer3DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

project_dir = str(Path(__file__).resolve().parents[2])
if project_dir not in sys.path:
    sys.path.append(project_dir)

from train_one_step import train_one_step

from vorta.constants import STA_TILE_SIZE, STA_VIDEO_SIZE
from vorta.dataset import LatentDataset, sp_parallel_dataloader_wrapper
from vorta.patch.modeling_hunyuan import apply_vorta_transformer
from vorta.patch.utils import hunyuan_pixel2token, prepare_hunyuan_self_attn_kwargs
from vorta.train import (
    load_optimizer_checkpoint,
    save_checkpoint,
    wrap_hunyuan_with_fsdp,
)
from vorta.ulysses import SP_STATE, TrainingLog, dist_prefix, set_seed
from vorta.utils import (
    arg_to_json,
    dtype_to_str,
    parent_to_ckpt_dir,
    setup_logging,
    str_to_dtype,
)

check_min_version("0.33.1")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hunyuan model")
    parser.add_argument("--exp_dir", type=Path, default=Path("train"), help="Output folder")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # data
    parser.add_argument(
        '--data_json_file', type=Path, help='Path to the data json file',
        default=Path('/raid/data/datasets/HD-Mixkit-Finetune-Hunyuan/videos2caption.json'))
    parser.add_argument('--data_uncond_rate', type=float, default=0.0, help='Unconditional rate for data augmentation')
    parser.add_argument('--dataloader_num_workers', type=int, default=16, help='Number of workers for data loader')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--sp_size', type=int, default=2, help='Sequence parallel size')
    parser.add_argument('--train_sp_batch_size', type=int, default=1, help='Sequence parallel batch size for training')
    parser.add_argument('--video_size', type=int, nargs=3, default=[117, 720, 1280], help='Video size')

    # model
    parser.add_argument('--pretrained_model_path', type=str, default='/raid/data/models/HunyuanVideo')
    parser.add_argument('--revision', type=str, default=None, help='Revision of the pretrained model')
    parser.add_argument('--resume_dir', type=Path, default=None, help='Path to the checkpoint to resume training')
    parser.add_argument(
        '--resume', type=str, default='latest', help='Path to the checkpoint to resume training, step-*')
    parser.add_argument(
        '--lowres_window_size', type=int, nargs=3, default=[2, 2, 2], help='BCS bucket size')
    parser.add_argument('--lowres_reduction_rate', type=float, default=0.5, help='Low resolution reduction rate')
    parser.add_argument(
        '--sliding_attn_window_size', type=int, nargs=3, default=[3, 3, 3], help='Sliding attention window size')
    parser.add_argument(
        '--sliding_attn_tile_size', type=int, nargs=3, default=STA_TILE_SIZE, help='Sliding attention tile size')
    parser.add_argument('--precondition_outputs', action='store_true', help='Precondition outputs')
    parser.add_argument(
        '--t_weighting_scheme', type=str, help='Timestep weighting scheme',
        default="uniform", choices=["uniform"],  # only support uniform for now
        # choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
    )
    parser.add_argument('--t_logit_mean', type=float, default=0.0, help='Logit mean for timestep sampling')
    parser.add_argument('--t_logit_std', type=float, default=1.0, help='Logit std for timestep sampling')
    parser.add_argument('--t_mode_scale', type=float, default=1.29, help='Mode scale for timestep sampling')
    parser.add_argument('--t_interval_index', type=int, default=None, help='Timestep interval index for sampling')
    parser.add_argument('--t_num_intervals', type=int, default=5, help='Timestep interval for sampling')

    # mlops
    parser.add_argument(
        "--fsdp_sharding_startegy", default="full", choices=["full", "hybrid_full", "hybrid_zero2", "none"], 
        help="FSDP sharding strategy for transformer"
    )
    parser.add_argument('--use_cpu_offload', action='store_true', help='Use CPU offload')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing')

    # optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--lr_scheduler", default="constant", help="Learning rate scheduler",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine scheduler")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power for polynomial scheduler")

    parser.add_argument('--max_train_steps', type=int, default=1_000, help='Maximum training steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')

    parser.add_argument('--diffusion_loss_weight', type=float, default=1.0, help='Diffusion loss weight')
    parser.add_argument('--reg_loss_weight', type=float, default=1.0, help='Regularization loss weight')
    parser.add_argument(
        '--last_layer_distill_loss_weight', type=float, default=1.0, help='Last layer distillation loss weight')
    parser.add_argument(
        '--hidden_layer_distill_loss_weight', type=float, default=0.0, help='Hidden layer distillation loss weight')

    # logging
    parser.add_argument('--report_interval', type=int, default=1, help='Report interval')
    parser.add_argument('--ckpt_interval', type=int, default=100, help='Checkpoint interval')

    args = parser.parse_args()

    args.ckpt_dir = args.exp_dir / "ckpt"  # model checkpoints
    args.eval_dir = args.exp_dir / "eval"  # generation results
    args.report_dir = args.exp_dir / "report"  # tensorboard / wandb logs
    if SP_STATE.rank <= 0:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        args.eval_dir.mkdir(parents=True, exist_ok=True)
        args.report_dir.mkdir(parents=True, exist_ok=True)

    args.resume, args.init_step = parent_to_ckpt_dir(args.resume, args.resume_dir or args.ckpt_dir)

    args.train_batch_size = math.ceil(args.train_sp_batch_size / args.sp_size)

    args.latent_shape = hunyuan_pixel2token(args.video_size)

    return args


def main():
    args = parse_args()
    if SP_STATE.rank <= 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(
            output_file=args.exp_dir / datetime.now().strftime("%Y%m%d-%H%M%S.log"),
            log_level=logging.INFO,
        )
    elif SP_STATE.local_rank <= 0:
        setup_logging(log_level=logging.INFO)
    else:
        setup_logging(log_level=logging.WARNING)
    
    # Log args
    args_json = deepcopy(args)
    logger.info(arg_to_json(args_json))

    # Initialize distributed environment
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group(backend="nccl", init_method="env://")
    logger.info(f"Distributed environment initialized {SP_STATE.rank} / {SP_STATE.world_size}")
    SP_STATE.setup_sp_group(sequence_parallel_size=args.sp_size)
    torch.cuda.set_device(SP_STATE.local_rank)
    device = torch.cuda.current_device()
    print(dist_prefix(f"Distributed environment initialized {device}"))

    # Seeding
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)  # each device (subsequence) has a different seed -> different noise

    # Data
    train_dataset = LatentDataset(
        args.data_json_file, args.latent_shape[0], args.data_uncond_rate, model_name='hunyuan', device=device)
    tran_datasampler = DistributedSampler(
        train_dataset, rank=SP_STATE.rank, num_replicas=SP_STATE.world_size, shuffle=False)
    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        sampler=tran_datasampler,
        collate_fn=train_dataset.collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    logger.info(f"Dataloader ({len(train_dataloader):,} batches) initialized")
    sp_data_iterator = sp_parallel_dataloader_wrapper(
        train_dataloader, device, args.train_batch_size, args.sp_size, args.train_sp_batch_size)

    # Model
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model_path, 
        subfolder="transformer",
        revision=args.revision,
        torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False)
    
    # additional forward args for transformer
    self_attention_kwargs = dict(
        # coreset attention
        lowres_window_size=args.lowres_window_size,
        lowres_reduction_rate=args.lowres_reduction_rate,
        # sliding attention
        window_size=args.sliding_attn_window_size,
        tile_size=args.sliding_attn_tile_size,
        latent_shape=args.latent_shape,
    )
    model_config = dict(
        name=transformer.__class__.__name__,
        pretrained_model_path=args.pretrained_model_path,
        revision=args.revision,
        train_video_size=args.video_size,
        self_attention_kwargs=self_attention_kwargs,
    )
    with open(args.exp_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=4)
    self_attention_kwargs = prepare_hunyuan_self_attn_kwargs(self_attention_kwargs, device=device)

    apply_vorta_transformer(
        transformer, train_router=True,
        checkpoint_file=args.resume / 'router.pt' if args.resume else None,
    )
    num_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters = {num_trainable_params:,}")

    # wrap model with FSDP
    transformer, no_split_modules = wrap_hunyuan_with_fsdp(
        transformer, device_id=device, 
        sharding_strategy=args.fsdp_sharding_startegy, use_cpu_offload=args.use_cpu_offload
    )
    # num_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    # logger.info(f"Total trainable parameters per FSDP shard = {num_trainable_params:,}")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    transformer.train()

    # Optimizer
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    if args.resume is not None:
        load_optimizer_checkpoint(args.resume, transformer, optimizer)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=-1,
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler()

    if SP_STATE.rank <= 0:
        # wandb.init(project='vorta-hunyuan', name=args.exp_name, config=args, dir=args.report_dir)
        writer = SummaryWriter(log_dir=args.report_dir)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) * args.sp_size / args.train_sp_batch_size / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    init_epoch = math.floor(args.init_step / num_update_steps_per_epoch)
    total_batch_size = int(
        SP_STATE.world_size * args.train_sp_batch_size / args.sp_size * args.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset):,}")
    logger.info(f"  Dataloader size = {len(train_dataloader):,}")
    logger.info(f"  Number of batches w. SP = {len(train_dataloader) * args.sp_size // args.train_sp_batch_size:,}")
    logger.info(f"  Epochs = {init_epoch:,} -> {num_train_epochs:,}")
    logger.info(f"  Resume training from step {args.init_step:,}")
    logger.info(f"  Instantaneous batch size per sequence parallel group = {args.train_sp_batch_size}")
    logger.info(f"  Total train batch size (w. data parallel, accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps:,}")

    progress_bar = tqdm(
        range(args.max_train_steps), initial=args.init_step, desc="Steps",
        disable=SP_STATE.local_rank > 0, # Only show the progress bar once on each machine.
    )
    training_log = TrainingLog()
    for step in range(args.init_step + 1, args.max_train_steps + 1):
        train_one_step(
            sp_data_iterator,
            transformer,
            optimizer,
            lr_scheduler,
            noise_scheduler,
            args.gradient_accumulation_steps,
            args.max_grad_norm,
            args.precondition_outputs,
            t_sampling_kwargs=dict(
                weighting_scheme=args.t_weighting_scheme,
                logit_mean=args.t_logit_mean,
                logit_std=args.t_logit_std,
                mode_scale=args.t_mode_scale,
                generator=None,
            ),
            self_attention_kwargs=self_attention_kwargs,
            loss_weights=dict(
                diffusion_loss=args.diffusion_loss_weight,
                reg_loss=args.reg_loss_weight,
                last_layer_distill_loss=args.last_layer_distill_loss_weight,
                hidden_layer_distill_loss=args.hidden_layer_distill_loss_weight,
            ),
            training_log=training_log,
            t_interval_index=args.t_interval_index,
            t_num_intervals=args.t_num_intervals,
        )
        progress_bar.set_postfix_str(str(training_log))
        progress_bar.update(1)

        if step % args.report_interval == 0 and SP_STATE.rank <= 0:
            writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], step)
            for i, t_ in enumerate(training_log.timestep):
                writer.add_scalar("t", t_, step * total_batch_size + i)
            writer.add_scalar("loss", training_log.loss, step)
            writer.add_scalar("l_fm", training_log.l_fm, step)
            writer.add_scalar("l_reg", training_log.l_reg, step)
            writer.add_scalar("l_last", training_log.l_last, step)
            writer.add_scalar("l_hidden", training_log.l_hidden, step)
            writer.add_scalar("grad_norm", training_log.grad_norm, step)
            # wandb.log({
            #     "lr": lr_scheduler.get_last_lr()[0],
            #     "t": t,
            #     "loss": loss,
            #     "l_fm": l_fm,
            #     "l_reg": l_reg,
            #     "l_last": l_last,
            #     "l_hidden": l_hidden,
            #     "grad_norm": grad_norm,
            # }, step=step)

        training_log.reset()

        if step % args.ckpt_interval == 0:
            save_checkpoint(args.ckpt_dir, step, transformer, optimizer)
            dist.barrier()

    # Save final checkpoint
    if step % args.ckpt_interval != 0:
        save_checkpoint(args.ckpt_dir, step, transformer)
        dist.barrier()

    if SP_STATE.enabled:
        SP_STATE.cleanup()

    if SP_STATE.rank <= 0:
        writer.close()
        # wandb.finish()

    logger.info("Training finished")


if __name__ == "__main__":
    main()
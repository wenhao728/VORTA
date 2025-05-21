#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/11 20:28:46
@Desc    :   
@Ref     :   
'''
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from ..ulysses import SP_STATE

logger = logging.getLogger(__name__)


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state_dict = model.state_dict()  # call in all ranks
        if optimizer is not None:
            optim_state = FSDP.optim_state_dict(model, optimizer)  # call in all ranks

    if SP_STATE.rank <= 0:
        router_state_dict = {k: v for k, v in full_state_dict.items() if 'router' in k}

        ckpt_step_dir = ckpt_dir / f"step-{step:06d}"
        ckpt_step_dir.mkdir(parents=True, exist_ok=True)
        torch.save(router_state_dict, ckpt_step_dir / "router.pt")
        logger.info(f"Saved router checkpoint at {ckpt_step_dir}")

        if optimizer is not None:
            torch.save(optim_state, ckpt_step_dir / "optimizer.pt")
            logger.info(f"Saved optimizer checkpoint at {ckpt_step_dir}")


def load_optimizer_checkpoint(
    ckpt_step_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    optim_state_dict = torch.load(ckpt_step_dir / "optimizer.pt", weights_only=False)
    fsdp_optim_state_dict = FSDP.optim_state_dict_to_load(
        model=model, optim=optimizer, optim_state_dict=optim_state_dict)
    optimizer.load_state_dict(fsdp_optim_state_dict)
    logger.info(f"Loaded optimizer checkpoint from {ckpt_step_dir}/optimizer.pt")


def load_router_checkpoint(
    ckpt_file: Path,
    transformer: nn.Module,
):
    router_state_dict = torch.load(ckpt_file, weights_only=True)
    logger.info(f"Loaded router checkpoint from {ckpt_file}")

    full_state_dict = transformer.state_dict()
    router_state_dict = {k: v for k, v in router_state_dict.items() if k in full_state_dict}
    full_state_dict.update(router_state_dict)
    transformer.load_state_dict(full_state_dict)
    logger.info(f"Updated transformer with router checkpoint")
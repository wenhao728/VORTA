#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/11 16:25:03
@Desc    :   
@Ref     :   
    https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/utils/fsdp_util.py
    https://huggingface.co/blog/deepspeed-to-fsdp-and-back
'''
import logging
from functools import partial
from typing import Optional

import torch
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTransformerBlock,
)
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from ..patch.router import Router

logger = logging.getLogger(__name__)


def wrap_hunyuan_with_fsdp(
    model, 
    sharding_strategy = "full",
    use_cpu_offload: bool = False,
    weight_type: torch.dtype = torch.bfloat16,
    device_id: Optional[int] = None,
):
    no_split_modules = (
        HunyuanVideoSingleTransformerBlock,
        HunyuanVideoTransformerBlock,
    )
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=no_split_modules,
    )

    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        reduce_dtype=weight_type, # Gradient communication precision.
        buffer_dtype=weight_type,
        cast_forward_inputs=False,
    )

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD  # zero3
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None

    cpu_offload = CPUOffload(offload_params=True) if use_cpu_offload else None
    logger.info(f"Using FSDP({use_cpu_offload=}):\n\t{mixed_precision=}\n\t{sharding_strategy=}")

    return FSDP(
        model, 
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy,
        device_id=device_id or torch.cuda.current_device(),
        limit_all_gathers=True,
        cpu_offload=cpu_offload,
        use_orig_params=True,  # partial trainable params
    ), no_split_modules


def wrap_wan_with_fsdp(
    model,
    sharding_strategy = "full",
    use_cpu_offload: bool = False,
    weight_type: torch.dtype = torch.bfloat16,
    reduce_dtype: Optional[torch.dtype] = torch.float32,
    buffer_dtype: Optional[torch.dtype] = torch.float32,
    process_group = None,
    device_id: Optional[int] = None,
    sync_module_states = True,
):
    auto_wrap_policy = partial(
        lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks)
    
    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        reduce_dtype=reduce_dtype or weight_type, # Gradient communication precision.
        buffer_dtype=buffer_dtype or weight_type,
    )

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD  # zero3
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    
    cpu_offload = CPUOffload(offload_params=True) if use_cpu_offload else None
    logger.info(f"Using FSDP({use_cpu_offload=}):\n\t{mixed_precision=}\n\t{sharding_strategy=}")

    return FSDP(
        model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=device_id or torch.cuda.current_device(),
        cpu_offload=cpu_offload,
        sync_module_states=sync_module_states,
        use_orig_params=True,  # partial trainable params
    )
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/06 23:22:21
@Desc    :   
@Ref     :   
    https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md
    https://github.com/hao-ai-lab/FastVideo/tree/main
'''
from .parallel_states import SP_STATE
from .utils import (  # all_to_all_5D,
    TrainingLog,
    all_gather,
    all_to_all,
    all_to_all_4D,
    broadcast_sp_group,
    dist_prefix,
    reduce_loss,
    set_seed,
    shrink_dim,
)

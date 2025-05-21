#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/02/26 15:43:01
@Desc    :   
    2025/03/17: initalize router using 1 0 0; will that help? no
@Ref     :   
'''
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class Router(nn.Module):
    def __init__(
        self, 
        embedding_dim: int,
        heads: int,
        num_experts: int = 3,
    ):
        super().__init__()
        self.heads = heads
        self.num_experts = num_experts

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim,  heads * num_experts, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        # logger.debug(f"{self.__class__.__name__}: {embedding_dim} -> {heads} * {num_experts}")

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temb (torch.Tensor): Timestep embedding, shape (B, D)

        Returns:
            torch.Tensor: Routing score, shape (B, H, num_experts)
        """
        head_scores: torch.Tensor = self.linear(self.silu(temb))
        head_scores = head_scores.unflatten(1, (self.heads, self.num_experts))
        return self.softmax(head_scores)
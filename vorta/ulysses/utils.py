# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Any, Tuple
from torch import Tensor

import torch.distributed as dist
from .parallel_states import SP_STATE


def _all_to_all_4D(
    input: torch.Tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None
) -> torch.Tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (input.dim() == 4), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous())

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(
                bs, seq_world_size, shard_seqlen, shard_hc, hs
            ).transpose(0, 3).transpose(0, 1).contiguous().reshape(
                seq_world_size, shard_hc, shard_seqlen, bs, hs
            )
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx
            ),
            None,
            None,
        )


def all_to_all_4D(input_: torch.Tensor, scatter_idx: int, gather_idx: int) -> torch.Tensor:
    return SeqAllToAll4D.apply(SP_STATE.group, input_, scatter_idx, gather_idx)


class AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        input_size = list(input_.size())
        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty_like(input_) for _ in range(SP_STATE.sp_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=SP_STATE.group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [input_size] * SP_STATE.sp_size

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[SP_STATE.group_local_rank]

        return grad_input, None


def all_gather(input_: torch.Tensor, dim: int = 0):
    return AllGather.apply(input_, dim)


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(input_: torch.Tensor, scatter_dim: int, gather_dim: int):
    return AllToAll.apply(input_, SP_STATE.group, scatter_dim, gather_dim)


def shrink_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if SP_STATE.enabled:
        local_size = tensor.size(dim) // SP_STATE.sp_size
        return tensor.narrow(dim, local_size * SP_STATE.group_local_rank, local_size)
    else:
        return tensor

def broadcast_sp_group(input_: torch.Tensor):
    src = SP_STATE.group_id * SP_STATE.sp_size  # broadcast from the first rank in the group
    dist.broadcast(input_, src=src, group=SP_STATE.group)


def dist_prefix(msg: str) -> str:
    return (
        f"[Rank: {SP_STATE.local_rank}/{SP_STATE.rank}/{SP_STATE.world_size} | "
        f"DP: {SP_STATE.group_id}/{SP_STATE.num_sp_groups} | "
        f"SP: {SP_STATE.group_local_rank}/{SP_STATE.sp_size}] {msg}"
    )


def set_seed(seed: int, device_specific: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `SP_STATE.rank`.
    """
    if device_specific:
        seed += SP_STATE.rank

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_loss(loss: torch.Tensor) -> float:
    if loss is None:
        return 0.0
    
    if not isinstance(loss, torch.Tensor):
        assert isinstance(loss, float) or isinstance(loss, int), (
            f"loss must be a tensor, float or int, got {type(loss)}")
        return loss

    avg_loss = loss.detach().clone()
    # dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    dist.reduce(avg_loss, dst=0, op=dist.ReduceOp.AVG)
    return avg_loss.item()


class TrainingLog:

    def __init__(self):
        self.reset()

    def __str__(self):
        return (
            f't: {self.timestep.tolist()[0]}, loss: {self.loss:.2e}, l_fm: {self.l_fm:.2e}, '
            f'l_reg: {self.l_reg:.2e}, l_last: {self.l_last:.2e}, l_hidden: {self.l_hidden:.2e}'
            # f', grad_norm: {self.grad_norm:.2e}'
        )

    def reset(self):
        self.timestep = None
        self.loss = 0
        self.l_fm = 0
        self.l_reg = 0
        self.l_last = 0
        self.l_hidden = 0
        self.grad_norm = 0

    def update(self, t, loss, diffusion_loss, reg_loss, last_layer_distill_loss, hidden_layer_distill_loss, grad_norm):
        if self.timestep is None:
            self.timestep = t.detach().cpu().long()
        else:
            self.timestep = torch.cat([self.timestep, t.detach().cpu().long()], dim=0)
        self.loss += reduce_loss(loss)
        self.l_fm += reduce_loss(diffusion_loss)
        self.l_reg += reduce_loss(reg_loss)
        self.l_last += reduce_loss(last_layer_distill_loss)
        self.l_hidden += reduce_loss(hidden_layer_distill_loss)
        if grad_norm is not None:
            self.grad_norm = grad_norm
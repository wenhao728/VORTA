#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/05 17:36:23
@Desc    :   
    Note that the definition of window_size on SWA and STA are different. SWA_window_size = STA_window_size * tile_size
@Ref     :   
    https://pytorch.org/blog/flexattention/
    https://github.com/pytorch/pytorch/issues/135028
    https://github.com/hao-ai-lab/FastVideo/blob/554ee17de54b95432edd4465a65e75d809b4564f/csrc/sliding_tile_attention/test/flex_sta_ref.py
'''
from typing import Callable, Optional, Tuple, Union

import torch
from torch.nn.attention.flex_attention import create_block_mask

from ..ulysses import SP_STATE
from .tile import tile_layout, untile_layout

torch._inductor.config.realize_opcount_threshold = 100


def create_sliding_window_attn_mask_func(
    latent_shape: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    text_seq_length: int,
    text_seq_length_no_pad: int,
    device: torch.device,
):
    t, h, w = latent_shape
    window_t, window_h, window_w = window_size
    video_seq_length = t * h * w
    seq_length = video_seq_length + text_seq_length
    seq_length_no_pad = video_seq_length + text_seq_length_no_pad

    def inverse_raster_index(index: torch.IntTensor) -> Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor]:
        t_idx = index // (h * w)
        hw_idx = index % (h * w)
        h_idx = hw_idx // w
        w_idx = hw_idx % w
        return t_idx, h_idx, w_idx

    def sliding_window_attn_mask_func(
        b: torch.IntTensor,
        h: torch.IntTensor,
        q_idx: torch.IntTensor,
        kv_idx: torch.IntTensor,
    ) -> torch.BoolTensor:
        # query from text
        text_to_all_mask = (q_idx >= video_seq_length) & (q_idx < seq_length_no_pad) & (kv_idx < seq_length_no_pad)

        # query from video, key-value from text
        video_to_text_mask = (q_idx < video_seq_length) & (kv_idx >= video_seq_length) & (kv_idx < seq_length_no_pad)

        # query from video, key-value from video
        q_t_idx, q_h_idx, q_w_idx = inverse_raster_index(q_idx)
        kv_t_idx, kv_h_idx, kv_w_idx = inverse_raster_index(kv_idx)
        video_to_video_mask = (
            (q_idx < video_seq_length) & (kv_idx < video_seq_length) &
            ((q_t_idx - kv_t_idx).abs() <= window_t // 2) &
            ((q_h_idx - kv_h_idx).abs() <= window_h // 2) &
            ((q_w_idx - kv_w_idx).abs() <= window_w // 2)
        )
        return text_to_all_mask | video_to_text_mask | video_to_video_mask

    return create_block_mask(
        sliding_window_attn_mask_func,
        B=None, H=None, Q_LEN=seq_length, KV_LEN=seq_length, device=device, _compile=True
    )


def create_sliding_tile_attn_mask_func(
    latent_shape: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    text_seq_length: int,
    text_seq_length_no_pad: int,
    device: torch.device,
):
    t, h, w = latent_shape
    window_t, window_h, window_w = window_size
    tile_t, tile_h, tile_w = tile_size

    video_seq_length = t * h * w
    num_tokens_per_tile = tile_t * tile_h * tile_w
    num_tile_t = t // tile_t
    num_tile_h = h // tile_h
    num_tile_w = w // tile_w

    seq_length = video_seq_length + text_seq_length
    seq_length_no_pad = video_seq_length + text_seq_length_no_pad

    def inverse_raster_index(index: torch.IntTensor) -> Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor]:
        tile_idx = index // num_tokens_per_tile
        t_idx = tile_idx // (num_tile_h * num_tile_w)
        hw_idx = tile_idx % (num_tile_h * num_tile_w)
        h_idx = hw_idx // num_tile_w
        w_idx = hw_idx % num_tile_w
        return t_idx, h_idx, w_idx

    def sliding_tile_attn_mask_func(
        b: torch.IntTensor,
        h: torch.IntTensor,
        q_idx: torch.IntTensor,
        kv_idx: torch.IntTensor,
    ) -> torch.BoolTensor:
        # query from text
        text_to_all_mask = (q_idx >= video_seq_length) & (q_idx < seq_length_no_pad) & (kv_idx < seq_length_no_pad)
        # text_to_all_mask = (q_idx >= video_seq_length) & (kv_idx < seq_length_no_pad)

        # query from video, key-value from text
        video_to_text_mask = (q_idx < video_seq_length) & (kv_idx >= video_seq_length) & (kv_idx < seq_length_no_pad)

        # query from video, key-value from video
        q_t_idx, q_h_idx, q_w_idx = inverse_raster_index(q_idx)
        kv_t_idx, kv_h_idx, kv_w_idx = inverse_raster_index(kv_idx)

        window_center_t = q_t_idx.clamp(window_t // 2, (num_tile_t - 1) - window_t // 2)
        window_center_h = q_h_idx.clamp(window_h // 2, (num_tile_h - 1) - window_h // 2)
        window_center_w = q_w_idx.clamp(window_w // 2, (num_tile_w - 1) - window_w // 2)

        video_to_video_mask = (
            (q_idx < video_seq_length) & (kv_idx < video_seq_length) &
            ((window_center_t - kv_t_idx).abs() <= window_t // 2) &
            ((window_center_h - kv_h_idx).abs() <= window_h // 2) &
            ((window_center_w - kv_w_idx).abs() <= window_w // 2)
        )
        return text_to_all_mask | video_to_text_mask | video_to_video_mask

    # torch.distributed.breakpoint()
    return create_block_mask(
        sliding_tile_attn_mask_func,
        B=None, H=None, Q_LEN=seq_length, KV_LEN=seq_length, device=device, _compile=True
    )


def sliding_tile_flex_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    flex_attn_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    encoder_query: Optional[torch.Tensor] = None,
    encoder_key: Optional[torch.Tensor] = None,
    encoder_value: Optional[torch.Tensor] = None,
    tile_size: Tuple[int, int, int] = (6, 8, 8),
    latent_shape: Tuple[int, int, int] = (30, 45, 80),
    head_dim: int = 2,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Sliding tile attention.

    Args:
        query (torch.Tensor): Query tensor with shape (B, H, S, D).
        key (torch.Tensor): Key tensor with shape (B, H, S, D).
        value (torch.Tensor): Value tensor with shape (B, H, S, D).
        window_size (Tuple[int, int, int]): Sliding window size (T, H, W).

    Returns:
        torch.Tensor: Output tensor with shape (B, S, H, D).
    """
    is_mmdit = encoder_query is not None

    # Tile, preprocess for sliding tile attention
    query = tile_layout(
        query, sp_size=SP_STATE.sp_size, tile_size=tile_size, latent_shape=latent_shape, head_dim=head_dim)
    key = tile_layout(
        key, sp_size=SP_STATE.sp_size, tile_size=tile_size, latent_shape=latent_shape, head_dim=head_dim)
    value = tile_layout(
        value, sp_size=SP_STATE.sp_size, tile_size=tile_size, latent_shape=latent_shape, head_dim=head_dim)

    if head_dim == 2:
        # transpose, (B, S, H, D) -> (B, H, S, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

    if is_mmdit:
        if head_dim == 2:
            encoder_query = encoder_query.transpose(1, 2)
            encoder_key = encoder_key.transpose(1, 2)
            encoder_value = encoder_value.transpose(1, 2)

        text_sequence_length = encoder_query.shape[2]
        query = torch.cat([query, encoder_query], dim=2)  # concat at sequence dimension
        key = torch.cat([key, encoder_key], dim=2)
        value = torch.cat([value, encoder_value], dim=2)

    # Sliding tile attention, (B, H, S, D) as input
    hidden_states = flex_attn_func(query.contiguous(), key.contiguous(), value.contiguous())

    if is_mmdit:
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :, :-text_sequence_length], 
            hidden_states[:, :, -text_sequence_length:]
        )
        # (B, H, S, D)
        hidden_states = untile_layout(
            hidden_states, sp_size=SP_STATE.sp_size, tile_size=tile_size, latent_shape=latent_shape, head_dim=1)
        if head_dim == 1:
            return hidden_states, encoder_hidden_states
        elif head_dim == 2:
            # (B, H, S, D) -> (B, S, H, D)
            return hidden_states.transpose(1, 2), encoder_hidden_states.transpose(1, 2)

    else:
        # (B, H, S, D)
        hidden_states = untile_layout(
            hidden_states, sp_size=SP_STATE.sp_size, tile_size=tile_size, latent_shape=latent_shape, head_dim=1)
        if head_dim == 1:
            return hidden_states
        elif head_dim == 2:
            # (B, H, S, D) -> (B, S, H, D)
            return hidden_states.transpose(1, 2)
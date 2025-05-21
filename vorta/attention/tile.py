from typing import Tuple

import torch
from einops import rearrange


def tile_layout(
    sequence: torch.Tensor,
    sp_size: int,
    tile_size: Tuple[int, int, int],
    latent_shape: Tuple[int, int, int],
    head_dim: int = 2,
):
    t, h, w = latent_shape
    tile_t, tile_h, tile_w = tile_size
    n_t = t // tile_t
    n_h = h // tile_h
    n_w = w // tile_w

    if head_dim == 1:
        sequence = rearrange(
            sequence, 
            "b head (sp t h w) d -> b head (t sp h w) d", 
            sp=sp_size, t=t // sp_size, h=h, w=w
        )
        return rearrange(
            sequence,
            "b h (n_t ts_t n_h ts_h n_w ts_w) d -> b h (n_t n_h n_w ts_t ts_h ts_w) d",
            n_t=n_t, n_h=n_h, n_w=n_w, ts_t=tile_t, ts_h=tile_h, ts_w=tile_w,
        )
    elif head_dim == 2:
        sequence = rearrange(
            sequence, 
            "b (sp t h w) head d -> b (t sp h w) head d", 
            sp=sp_size, t=t // sp_size, h=h, w=w
        )
        return rearrange(
            sequence,
            "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
            n_t=n_t, n_h=n_h, n_w=n_w, ts_t=tile_t, ts_h=tile_h, ts_w=tile_w,
        )


def untile_layout(
    sequence: torch.Tensor,
    sp_size: int,
    tile_size: Tuple[int, int, int],
    latent_shape: Tuple[int, int, int],
    head_dim: int = 2,
):
    t, h, w = latent_shape
    tile_t, tile_h, tile_w = tile_size
    n_t = t // tile_t
    n_h = h // tile_h
    n_w = w // tile_w

    if head_dim == 1:
        sequence = rearrange(
            sequence,
            "b h (n_t n_h n_w ts_t ts_h ts_w) d -> b h (n_t ts_t n_h ts_h n_w ts_w) d",
            n_t=n_t, n_h=n_h, n_w=n_w, ts_t=tile_t, ts_h=tile_h, ts_w=tile_w,
        )
        return rearrange(
            sequence,
            "b head (t sp h w) d -> b head (sp t h w) d",
            sp=sp_size, t=t // sp_size, h=h, w=w
        )

    elif head_dim == 2:
        sequence = rearrange(
            sequence,
            "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
            n_t=n_t, n_h=n_h, n_w=n_w, ts_t=tile_t, ts_h=tile_h, ts_w=tile_w,
        )
        return rearrange(
            sequence,
            "b (t sp h w) head d -> b (sp t h w) head d",
            sp=sp_size, t=t // sp_size, h=h, w=w
        )
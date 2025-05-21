from typing import Any, Dict, Optional, Tuple

import torch

from ..attention import create_sliding_tile_attn_mask_func, get_group_info


def prepare_wan_self_attn_kwargs(
    self_attention_kwargs: Dict[str, Any],
    device: torch.device,
    tau_sparse: Optional[float] = None,
) -> Dict[str, Any]:
    # low-resolution attention
    lowres_group_info = get_group_info(
        self_attention_kwargs['latent_shape'],
        self_attention_kwargs.pop('lowres_window_size'),
        reduction_rate=self_attention_kwargs.pop("lowres_reduction_rate"),
        device=device,
    )
    self_attention_kwargs.update(lowres_group_info=lowres_group_info)

    # flex sliding attention
    flex_attn_mask_func = create_sliding_tile_attn_mask_func(
        latent_shape=self_attention_kwargs["latent_shape"],
        window_size=self_attention_kwargs["window_size"],
        tile_size=self_attention_kwargs["tile_size"],
        text_seq_length=0,
        text_seq_length_no_pad=0,
        device=device,
    )
    self_attention_kwargs.update(flex_attn_mask_func=flex_attn_mask_func)

    if tau_sparse is not None:
        self_attention_kwargs.update(tau_sparse=tau_sparse)

    return self_attention_kwargs


def prepare_hunyuan_self_attn_kwargs(
    self_attention_kwargs: Dict[str, Any],
    device: torch.device,
    tau_sparse: Optional[float] = None,
) -> Dict[str, Any]:
    # low-resolution attention
    lowres_group_info = get_group_info(
        self_attention_kwargs['latent_shape'],
        self_attention_kwargs.pop('lowres_window_size'),
        reduction_rate=self_attention_kwargs.pop("lowres_reduction_rate"),
        device=device,
    )
    self_attention_kwargs.update(lowres_group_info=lowres_group_info)

    if tau_sparse is not None:
        self_attention_kwargs.update(tau_sparse=tau_sparse)

    return self_attention_kwargs


class Pixel2TokenFactory:

    def __init__(
        self,
        temporal_vae: int,
        spatial_vae: int,
        temporal_patchfy: int = 1,
        spatial_patchfy: int = 2,
    ):
        self.temporal_vae = temporal_vae
        self.spatial_vae = spatial_vae
        self.temporal_patchfy = temporal_patchfy
        self.spatial_patchfy = spatial_patchfy

        self.temporal_total = self.temporal_vae * self.temporal_patchfy
        self.spatial_total = self.spatial_vae * self.spatial_patchfy

    def __call__(self, video_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (
            self.pixel_to_token(video_shape[0], self.temporal_total),
            self.pixel_to_token(video_shape[1], self.spatial_total),
            self.pixel_to_token(video_shape[2], self.spatial_total),
        )

    @staticmethod
    def pixel_to_token(num_pixel: int, pixel2token: int) -> int:
        num_token, mod = divmod(num_pixel, pixel2token)
        if mod == 0:
            return num_token
        elif mod == 1:
            return num_token + 1
        else:
            raise ValueError(f"Number of pixel {num_pixel} is not a multiple of pixel2token {pixel2token}.")


hunyuan_pixel2token = Pixel2TokenFactory(temporal_vae=4, spatial_vae=8)
wan_pixel2token = Pixel2TokenFactory(temporal_vae=4, spatial_vae=8)

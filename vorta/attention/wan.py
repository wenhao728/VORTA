#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/17 10:58:52
@Desc    :   
    2025/03/22: when num_head for specific route is 0
    2025/04/07: test Cosine Sim difference w. and w.o. Normalization -> necessary w/o slowdown
    2025/04/07: implement the new pooling strategy
@Ref     :   
'''
import logging
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch._dynamo
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from torch.nn.attention.flex_attention import flex_attention

from ..ulysses import SP_STATE, all_to_all_4D, shrink_dim
from .coreset_select import (
    LowresGroupInfo,
    pool_sequence_by_similarity,
    unpool_sequence_by_similarity,
)
from .sliding_attn_flex import sliding_tile_flex_attn

torch._dynamo.config.cache_size_limit = 40
flex_attention = torch.compile(flex_attention, dynamic=False)
logger = logging.getLogger(__name__)


def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
    x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
    return x_out.type_as(hidden_states)


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_cross_attn = encoder_hidden_states is not None

        query, key, value, encoder_hidden_states_img = self._input_proj(
            attn, hidden_states, encoder_hidden_states, rotary_emb)

        hidden_states, hidden_states_img = self._attn(
            attn, query, key, value, attention_mask, encoder_hidden_states_img, is_cross_attn)

        hidden_states = self._output_proj(attn, hidden_states, hidden_states_img)
        return hidden_states

    def _input_proj(
        self, 
        attn: Attention, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: Optional[torch.Tensor] = None, 
        rotary_emb: Optional[torch.Tensor] = None
    ):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # cross attention with image
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]

        if encoder_hidden_states is None:
            # self attention
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # rms_norm_across_heads, sequence length is not reducable
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # (B, S, H * D) -> (B, S, H, D) -> (B, H, S, D)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            rotary_emb = shrink_dim(rotary_emb, dim=2)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        return query, key, value, encoder_hidden_states_img
    
    def _attn(
        self, attn, query, key, value, 
        attention_mask, 
        encoder_hidden_states_img, 
        is_cross_attn: bool,
        skip_communication: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if SP_STATE.enabled and not skip_communication:
            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2)
            if is_cross_attn:
                key = shrink_dim(key, dim=1)  # shrink the head dimension
                value = shrink_dim(value, dim=1)
            else:
                key = all_to_all_4D(key, scatter_idx=1, gather_idx=2)
                value = all_to_all_4D(value, scatter_idx=1, gather_idx=2)

        # optional: I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if SP_STATE.enabled and not skip_communication:
                key_img = shrink_dim(key_img, dim=1)
                value_img = shrink_dim(value_img, dim=1)

            # FlashAttention by default
            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.type_as(query)
            if SP_STATE.enabled and not skip_communication:
                hidden_states_img = all_to_all_4D(hidden_states_img, scatter_idx=2, gather_idx=1)

        # FlashAttention by default, attention_mask is None
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.type_as(query)
        if SP_STATE.enabled and not skip_communication:
            hidden_states = all_to_all_4D(hidden_states, scatter_idx=2, gather_idx=1)

        return hidden_states, hidden_states_img

    def _output_proj(self, attn, hidden_states, hidden_states_img):
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        if hidden_states_img is not None:
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttnProcessorTripleTrain(WanAttnProcessor2_0):
    def __init__(self, check_input: bool = False):
        super().__init__()
        self.check_input = check_input

    def _check_input(
        self, 
        hidden_states, 
        lowres_group_info,
        latent_shape,
        window_size,
        tile_size,
    ):
        if self.check_input:
            seq_length = hidden_states.shape[1] * SP_STATE.sp_size
            num_groups = lowres_group_info.center_indices.shape[0]
            group_size = lowres_group_info.center_indices.shape[1] + lowres_group_info.margin_indices.shape[1]
            
            if seq_length != latent_shape[0] * latent_shape[1] * latent_shape[2]:
                raise ValueError(
                    f"Input sequence length {seq_length} does not match latent shape {latent_shape}."
                )
            for t_size, l_size in zip(tile_size, latent_shape):
                if l_size % t_size != 0:
                    raise ValueError(
                        f"Tile size {tile_size} (dim={t_size}) does not divide latent shape {latent_shape} (dim={l_size}).")
            
            if seq_length != num_groups * group_size:
                raise ValueError(
                    f"Input sequence length {seq_length} does not match low-res info {num_groups}x{group_size}."
                )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        # custom arguments
        use_original_attn: bool = False,
        routing_score: Optional[torch.Tensor] = None,
        # Low-resolution attention arguments
        lowres_group_info: Optional[LowresGroupInfo] = None,
        # Sliding window attention arguments
        # # FlexAttention
        flex_attn_mask_func: Optional[Callable[[torch.IntTensor], torch.BoolTensor]] = None,
        # # STA
        window_size: Tuple[int, int, int] = (3, 3, 3),
        tile_size: Tuple[int, int, int] = (6, 8, 8),
        latent_shape: Tuple[int, int, int] = (20, 30, 52),
    ) -> torch.Tensor:
        is_cross_attn = encoder_hidden_states is not None

        if is_cross_attn or use_original_attn:
            # no additional ops on the cross attention
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        else:
            self._check_input(hidden_states, lowres_group_info, latent_shape, window_size, tile_size)

            # (B, S, H * D) -> (B, S, H, D) -> (B, H, S, D)
            query, key, value, _ = self._input_proj(
                attn, hidden_states, encoder_hidden_states=None, rotary_emb=rotary_emb)

            hidden_states_list = []
            full_hidden_states, _ = self._attn(
                attn, query, key, value, attention_mask=None, encoder_hidden_states_img=None, is_cross_attn=False)
            hidden_states_list.append(full_hidden_states)
            
            lowres_hidden_states = self._lowres_attn(attn, query, key, value, lowres_group_info)
            hidden_states_list.append(lowres_hidden_states)

            sliding_hidden_states = self._sliding_attn(
                query, key, value, flex_attn_mask_func, window_size, tile_size, latent_shape)
            hidden_states_list.append(sliding_hidden_states)

            hidden_states = self._combine_attn_outputs(routing_score, hidden_states_list)
            hidden_states = self._output_proj(attn, hidden_states, hidden_states_img=None)
            return hidden_states

    def _lowres_attn(self, attn, query, key, value, lowres_group_info):
        # always self attention
        if SP_STATE.enabled:
            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2)
            key = all_to_all_4D(key, scatter_idx=1, gather_idx=2)
            value = all_to_all_4D(value, scatter_idx=1, gather_idx=2)

        lowres_query, query_matching_results = pool_sequence_by_similarity(
            query, lowres_group_info, matching_results=None)
        lowres_key, _ = pool_sequence_by_similarity(
            key, lowres_group_info, matching_results=query_matching_results)
        lowres_value, _ = pool_sequence_by_similarity(
            value, lowres_group_info, matching_results=query_matching_results)

        lowres_hidden_states, _ = self._attn(
            attn, lowres_query, lowres_key, lowres_value, 
            attention_mask=None, encoder_hidden_states_img=None, is_cross_attn=False, skip_communication=True
        )

        # lowres_hidden_states = unpool_sequence(lowres_hidden_states, lowres_index, head_dim=1)
        lowres_hidden_states = unpool_sequence_by_similarity(
            lowres_hidden_states, lowres_group_info, matching_results=query_matching_results)
        lowres_hidden_states = lowres_hidden_states.type_as(query)

        if SP_STATE.enabled:
            lowres_hidden_states = all_to_all_4D(lowres_hidden_states, scatter_idx=2, gather_idx=1)

        return lowres_hidden_states
    
    def _sliding_attn(
        self, query, key, value,
        flex_attn_mask_func: Optional[Callable[[torch.IntTensor], torch.BoolTensor]],
        window_size: Tuple[int, int, int],
        tile_size: Tuple[int, int, int],
        latent_shape: Tuple[int, int, int],
    ):
        # always self attention
        if SP_STATE.enabled:
            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2)
            key = all_to_all_4D(key, scatter_idx=1, gather_idx=2)
            value = all_to_all_4D(value, scatter_idx=1, gather_idx=2)

        hidden_states = sliding_tile_flex_attn(
            query, key, value, 
            partial(flex_attention, block_mask=flex_attn_mask_func),
            tile_size=tile_size, latent_shape=latent_shape, head_dim=1
        )

        if SP_STATE.enabled:
            hidden_states = all_to_all_4D(hidden_states, scatter_idx=2, gather_idx=1)
        hidden_states = hidden_states.type_as(query)
        return hidden_states

    def _combine_attn_outputs(self, routing_score, hidden_states_list):
        return (
            routing_score[:, :, :, None, None] * # (B, H, 3) -> (B, H, 3, 1, 1)
            torch.stack(hidden_states_list, dim=2) # 3 * (B, H, S, D) -> (B, H, 3, S, D)
        ).sum(dim=2)


class WanAttnProcessorTripleEval(WanAttnProcessorTripleTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_emb: torch.Tensor,
        # custom arguments
        tau_sparse: float,
        routing_score: torch.Tensor,
        # Low-resolution attention arguments
        lowres_group_info: Optional[LowresGroupInfo] = None,
        # Sliding window attention arguments
        # # FlexAttention
        flex_attn_mask_func: Optional[Callable[[torch.IntTensor], torch.BoolTensor]] = None,
        # # STA
        window_size: Tuple[int, int, int] = (3, 3, 3),
        tile_size: Tuple[int, int, int] = (6, 8, 8),
        latent_shape: Tuple[int, int, int] = (20, 30, 52),
        use_original_attn: bool = False,
    ):
        is_cross_attn = encoder_hidden_states is not None

        if is_cross_attn or use_original_attn:
            # no additional ops on the cross attention
            return WanAttnProcessor2_0.__call__(
                self, attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        else:
            self._check_input(hidden_states, lowres_group_info, latent_shape, window_size, tile_size)
            if is_cross_attn:
                raise ValueError("Cross attention is not supported in evaluation mode.")

            # (B, S, H * D) -> (B, S, H, D) -> (B, H, S, D)
            query, key, value, _ = self._input_proj(
                attn, hidden_states, encoder_hidden_states=None, rotary_emb=rotary_emb)

            # Initialize all QKV tensors and masks as None
            full_query = lowres_query = sw_query = None
            full_key = lowres_key = sw_key = None 
            full_value = lowres_value = sw_value = None
            full_head_mask = lowres_head_mask = sw_head_mask = None

            # Get routed QKV based on enabled attention types
            routed_qkvm = self._get_routed_qkv(query, key, value, routing_score, tau_sparse)
            full_query, full_key, full_value, full_head_mask = routed_qkvm[:4]
            lowres_query, lowres_key, lowres_value, lowres_head_mask = routed_qkvm[4:8]
            sw_query, sw_key, sw_value, sw_head_mask = routed_qkvm[-4:]

            # full attention
            if full_query is None or full_query.shape[1] == 0:
                full_hidden_states = None
            else:
                full_hidden_states, _ = self._attn(
                    attn, full_query, full_key, full_value, 
                    attention_mask=None, encoder_hidden_states_img=None, is_cross_attn=False)
            
            # low-resolution attention
            if lowres_query is None or lowres_query.shape[1] == 0:
                lowres_hidden_states = None
            else:
                lowres_hidden_states = self._lowres_attn(
                    attn, lowres_query, lowres_key, lowres_value, lowres_group_info)
            
            # sliding attention
            if sw_query is None or sw_query.shape[1] == 0:
                sliding_hidden_states = None
            else:
                sliding_hidden_states = self._sliding_attn(
                    sw_query, sw_key, sw_value, flex_attn_mask_func, window_size, tile_size, latent_shape)

            # combine the outputs
            hidden_states = self._combine_attn_outputs(
                attn.heads, 
                [full_hidden_states, lowres_hidden_states, sliding_hidden_states], 
                [full_head_mask, lowres_head_mask, sw_head_mask]
            )

            hidden_states = self._output_proj(attn, hidden_states, hidden_states_img=None)
            return hidden_states
    
    def _get_routed_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        routing_score: torch.Tensor,
        tau_sparse: float,
    ) -> Tuple[torch.Tensor]:
        B, _, S, D = query.shape
        # (B, H, num_experts) -> (H, 1), (H, 1)
        top1_score, top1_indice = routing_score[0].topk(1, dim=-1)  # use the 1st sample in the batch
        top1_indice[top1_score < tau_sparse] = 0  # less confident, use full attention, (H, 1)
        top1_indice = top1_indice.squeeze(-1)  # (H, 1) -> (H,)

        outputs = []
        for i in range(routing_score.shape[-1]):
            head_mask = top1_indice == i
            # if not head_mask.any():
            #     query_i, key_i, value_i = None, None, None
            # else:
            # (H,) -> (H, 1) -> (1, H, 1, 1)
            head_indices = torch.nonzero(head_mask).view(1, -1, 1, 1).expand(B, -1, S, D)
            query_i = torch.gather(query, dim=1, index=head_indices)
            key_i = torch.gather(key, dim=1, index=head_indices)
            value_i = torch.gather(value, dim=1, index=head_indices)

            outputs.extend([query_i, key_i, value_i, head_mask])

        return outputs

    def _combine_attn_outputs(
        self,
        num_heads: int,
        hidden_states_list: Tuple[torch.Tensor],
        head_mask_list: Tuple[torch.Tensor],
    ):
        # (B, h1, S, D) + (B, h2, S, D) + (B, h3, S, D) -> (B, H=h1+h2+h3, S, D)
        for head_hidden_states in hidden_states_list:
            if head_hidden_states is not None:
                output_shape = list(head_hidden_states.shape)
                device = head_hidden_states.device
                dtype = head_hidden_states.dtype
                break

        output_shape[1] = num_heads
        hidden_states = torch.empty(*output_shape, device=device, dtype=dtype)
        for head_hidden_states, head_mask in zip(hidden_states_list, head_mask_list):
            if head_hidden_states is not None:
                hidden_states[:, head_mask] = head_hidden_states

        return hidden_states
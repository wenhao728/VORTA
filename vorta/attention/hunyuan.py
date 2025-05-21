#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/02/25 16:21:24
@Desc    :   
    2025/03/13: validate flex attn and STA implementation
    2025/03/22: when num_head for specific route is 0
@Ref     :   
'''
import logging
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch._dynamo
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention.flex_attention import flex_attention

from ..ulysses import SP_STATE, all_gather, all_to_all_4D, shrink_dim
from .coreset_select import (
    LowresGroupInfo,
    pool_sequence_by_similarity,
    unpool_sequence_by_similarity,
)
from .sliding_attn_flex import sliding_tile_flex_attn

torch._dynamo.config.cache_size_limit = 256  # text seq length
flex_attention = torch.compile(flex_attention, dynamic=False)

logger = logging.getLogger(__name__)


class HunyuanVideoFlashAttnProcessor:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoRnRAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def _step_to_qkv_and_unflatten(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        # 1. QKV projections
        if attn.add_q_proj is None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # (B, S, H * D) -> (B, S, H, D) -> (B, H, S, D)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        return query, key, value

    def _step_qk_norm(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        return query, key

    def _step_rotary_emb(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        encoder_hidden_states_seq_len: int,
        image_rotary_emb: torch.Tensor,
    ):
        # 3. Rotational positional embeddings applied to latent stream

        image_rotary_emb = (
            shrink_dim(image_rotary_emb[0], dim=0),
            shrink_dim(image_rotary_emb[1], dim=0),
        )

        if attn.add_q_proj is None:
            encoder_query = query[:, :, -encoder_hidden_states_seq_len :]
            encoder_key = key[:, :, -encoder_hidden_states_seq_len :]

            query = query[:, :, : -encoder_hidden_states_seq_len]
            key = key[:, :, : -encoder_hidden_states_seq_len]

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

        if attn.add_q_proj is None:
            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)

        return query, key

    def _step_encoder_to_qkv_and_concat(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            # (B, S, H * D) -> (B, S, H, D) -> (B, H, S, D)
            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            # (B, H, S + S_t, D)
            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)
        return query, key, value

    def _step_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states_seq_len: int,
        skip_communication: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 5. Attention
        text_seq_length = encoder_hidden_states_seq_len
        if SP_STATE.enabled and not skip_communication:
            # parallel dim: S -> H for attention, (B, H, S, D)
            query, encoder_query = query[:, :, : -text_seq_length], query[:, :, -text_seq_length :]
            key, encoder_key = key[:, :, : -text_seq_length], key[:, :, -text_seq_length :]
            value, encoder_value = value[:, :, : -text_seq_length], value[:, :, -text_seq_length :]

            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2)
            key = all_to_all_4D(key, scatter_idx=1, gather_idx=2)
            value = all_to_all_4D(value, scatter_idx=1, gather_idx=2)

            # local heads
            encoder_query = shrink_dim(encoder_query, dim=1)
            encoder_key = shrink_dim(encoder_key, dim=1)
            encoder_value = shrink_dim(encoder_value, dim=1)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        
        batch_size, _, seq_length, _ = query.shape
        assert batch_size == 1, f"Batch size {batch_size} is not supported for {self.__class__.__name__}."
        effective_seq_length = attention_mask.squeeze().sum()
        hidden_states = F.scaled_dot_product_attention(
            query[:, :, :effective_seq_length], 
            key[:, :, :effective_seq_length], 
            value[:, :, :effective_seq_length], 
            dropout_p=0.0, is_causal=False
        )
        hidden_states = F.pad(hidden_states, (0, 0, 0, seq_length - effective_seq_length), value=0.0)

        # Split the output back to hidden_states and encoder_hidden_states
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :, : -text_seq_length],
            hidden_states[:, :, -text_seq_length :],
        )

        if SP_STATE.enabled and not skip_communication:
            # parallel dim: H -> S for Dense layer
            hidden_states = all_to_all_4D(hidden_states, scatter_idx=2, gather_idx=1)
            encoder_hidden_states = all_gather(encoder_hidden_states, dim=1).contiguous()

        return hidden_states.to(query.dtype), encoder_hidden_states.to(query.dtype)

    def _step_to_output(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        # 6. Output projection
        # (B, H, S, D) -> (B, S, H, D) -> (B, S, H * D)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2).flatten(2, 3)

        if getattr(attn, "to_out", None) is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        if getattr(attn, "to_add_out", None) is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. QKV projections
        # (B, H, S, D)
        query, key, value = self._step_to_qkv_and_unflatten(attn, hidden_states, encoder_hidden_states)

        # 2. QK normalization
        query, key = self._step_qk_norm(attn, query, key)

        # 3. Rotational positional embeddings applied to latent stream
        query, key = self._step_rotary_emb(attn, query, key, encoder_hidden_states.shape[1], image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        query, key, value = self._step_encoder_to_qkv_and_concat(attn, query, key, value, encoder_hidden_states)

        # 5. Attention
        hidden_states, encoder_hidden_states = self._step_attention(
            query, key, value, attention_mask, encoder_hidden_states.shape[1])

        # 6. Output projection
        hidden_states, encoder_hidden_states = self._step_to_output(attn, hidden_states, encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanVideoFlashAttnProcessorTripleTrain(HunyuanVideoFlashAttnProcessor):
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
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        use_original_attn: bool = False,
        routing_score: Optional[torch.Tensor] = None,
        # Low-resolution attention arguments
        lowres_group_info: Optional[LowresGroupInfo] = None,
        # Sliding window attention arguments
        flex_attn_mask_func: Optional[Callable[[torch.IntTensor], torch.BoolTensor]] = None,
        # # STA
        window_size: Tuple[int, int, int] = (3, 3, 3),
        tile_size: Tuple[int, int, int] = (6, 8, 8),
        latent_shape: Tuple[int, int, int] = (30, 48, 80),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            attn (Attention): Attention module with weights.
            hidden_states (torch.Tensor): Hidden states tensor with shape (2B, S, D).
            encoder_hidden_states (torch.Tensor): Encoder hidden states tensor with shape (2B, S_text, D).
            attention_mask (torch.Tensor): Attention mask tensor with shape (2B, 1, 1, S + S_text).
            image_rotary_emb (torch.Tensor): Rotary positional embeddings tensor with shape (S, D).
            routing_score (torch.Tensor): Routing score tensor with shape (B, H, num_experts).

            Low-resolution attention arguments:
                lowres_index (torch.Tensor): Low-res index tensor with shape (S // scale_factor, scale_factor).
                reduction_mod (Literal['mean', 'random', 'corner']): _description_

            Sliding window attention arguments:
                window_size (Tuple[int, int, int]): _description_
                tile_size (Tuple[int, int, int]): _description_
                latent_shape (Tuple[int, int, int]): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hidden_states, encoder_hidden_states
        """
        if use_original_attn:
            # logger.debug(f"original attn for training target")
            attn_hidden_states, attn_encoder_hidden_states = super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask, 
                image_rotary_emb
            )
            # logger.debug(f"full_attn_hidden_states: {attn_hidden_states.shape}")
        else:
            # logger.debug(f"routed attn for training")
            attn_hidden_states, attn_encoder_hidden_states = self._routed_attn_forward(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
                routing_score,
                lowres_group_info=lowres_group_info,
                flex_attn_mask_func=flex_attn_mask_func,
                window_size=window_size,
                tile_size=tile_size,
                latent_shape=latent_shape,
            )
            # logger.debug(f"routed_attn_hidden_states: {attn_hidden_states.shape}")
        return attn_hidden_states, attn_encoder_hidden_states

    def _routed_attn_forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        image_rotary_emb,
        routing_score,
        lowres_group_info: Optional[LowresGroupInfo],
        flex_attn_mask_func: Optional[Callable[[torch.IntTensor], torch.BoolTensor]],
        window_size: Tuple[int, int, int],
        tile_size: Tuple[int, int, int],
        latent_shape: Tuple[int, int, int],
    ):
        self._check_input(hidden_states, lowres_group_info, latent_shape, window_size, tile_size)

        text_seq_length = encoder_hidden_states.shape[1]

        # 1. QKV projections
        query, key, value = self._step_to_qkv_and_unflatten(attn, hidden_states, encoder_hidden_states)
        # logger.debug(f"1. QKV projection: query: {query.shape}, key: {key.shape}, value: {value.shape}")

        # 2. QK normalization
        query, key = self._step_qk_norm(attn, query, key)
        # logger.debug(f"2. QK norm: query: {query.shape}, key: {key.shape}")

        # 3. Rotational positional embeddings applied to latent stream
        query, key = self._step_rotary_emb(attn, query, key, text_seq_length, image_rotary_emb)
        # logger.debug(f"3. Rotary emb: query: {query.shape}, key: {key.shape}")

        # 4. Encoder condition QKV projection and normalization, (B, H, S, D)
        query, key, value = self._step_encoder_to_qkv_and_concat(attn, query, key, value, encoder_hidden_states)
        # logger.debug(f"4. Encoder QKV: query: {query.shape}, key: {key.shape}, value: {value.shape}")

        # 5. Attention
        # # 5.1 Full attention, (B, H, S, D)
        full_hidden_states, full_encoder_hidden_states = self._step_attention(
            query, key, value, attention_mask, text_seq_length)
        # logger.debug(f"5. Full attn: full_hidden_states: {full_hidden_states.shape}")

        # # 5.2 Low-res attention, (B, H, S, D)
        lowres_hidden_states, lowres_encoder_hidden_states = self._step_lowres_attention(
            query, key, value, attention_mask, text_seq_length, lowres_group_info
        )
        # logger.debug(f"5. Lowres attn: lowres_hidden_states (upsampled): {lowres_hidden_states.shape}")

        # # 5.3 Sliding attention, (B, H, S, D)
        sliding_hidden_states, sliding_encoder_hidden_states = self._step_sliding_attention(
            query, key, value, attention_mask, text_seq_length,
            flex_attn_mask_func,
            window_size, tile_size, latent_shape
        )
        # logger.debug(f"5. Sliding window attn: sliding_hidden_states: {sliding_hidden_states.shape}")

        # # 5.4 Combine attention
        hidden_states = self._combine_attn_outputs(
            routing_score, 
            [full_hidden_states, lowres_hidden_states, sliding_hidden_states]
        )
        encoder_hidden_states = self._combine_attn_outputs(
            routing_score, 
            [full_encoder_hidden_states, lowres_encoder_hidden_states, sliding_encoder_hidden_states]
        )

        # 6. Output projection
        hidden_states, encoder_hidden_states = self._step_to_output(attn, hidden_states, encoder_hidden_states)

        return hidden_states, encoder_hidden_states
    
    def _step_lowres_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.IntTensor,
        text_seq_length: int,
        lowres_group_info: Optional[LowresGroupInfo] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query, encoder_query = query[:, :, : -text_seq_length], query[:, :, -text_seq_length :]
        key, encoder_key = key[:, :, : -text_seq_length], key[:, :, -text_seq_length :]
        value, encoder_value = value[:, :, : -text_seq_length], value[:, :, -text_seq_length :]

        if SP_STATE.enabled:
            # parallel dim: S -> H for attention, (B, H, S, D)
            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2)
            key = all_to_all_4D(key, scatter_idx=1, gather_idx=2)
            value = all_to_all_4D(value, scatter_idx=1, gather_idx=2)

            encoder_query = shrink_dim(encoder_query, dim=1)
            encoder_key = shrink_dim(encoder_key, dim=1)
            encoder_value = shrink_dim(encoder_value, dim=1)

        lowres_query, query_matching_results = pool_sequence_by_similarity(
            query, lowres_group_info, matching_results=None)
        lowres_key, key_matching_results = pool_sequence_by_similarity(
            key, lowres_group_info, matching_results=None)
        lowres_value, _ = pool_sequence_by_similarity(
            value, lowres_group_info, matching_results=key_matching_results)

        lowres_seq_length = lowres_query.shape[2]
        lowres_hidden_states, lowres_encoder_hidden_states = self._step_attention(
            torch.cat([lowres_query, encoder_query], dim=2),
            torch.cat([lowres_key, encoder_key], dim=2),
            torch.cat([lowres_value, encoder_value], dim=2),
            attention_mask[:, :, :, - lowres_seq_length - text_seq_length :],
            text_seq_length,
            skip_communication=True,
        )
        # logger.debug(f"5. Lowres attn: lowres_hidden_states (raw): {lowres_hidden_states.shape}")
        lowres_hidden_states = unpool_sequence_by_similarity(
            lowres_hidden_states, lowres_group_info, matching_results=query_matching_results)

        if SP_STATE.enabled:
            lowres_hidden_states = all_to_all_4D(lowres_hidden_states, scatter_idx=2, gather_idx=1)
            lowres_encoder_hidden_states = all_gather(lowres_encoder_hidden_states, dim=1).contiguous()

        return lowres_hidden_states, lowres_encoder_hidden_states

    def _step_sliding_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        text_seq_length: int,
        flex_attn_mask_func: Optional[Callable[[torch.IntTensor], torch.BoolTensor]],
        tile_size: Tuple[int, int, int],
        latent_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sliding window attention.

        Args:
            query (torch.Tensor): Query tensor with shape (B, H, S, D).
            key (torch.Tensor): Key tensor with shape (B, H, S, D).
            value (torch.Tensor): Value tensor with shape (B, H, S, D).
            attention_mask (torch.Tensor): Attention mask tensor with shape (B, 1, 1, S).
        """
        query, encoder_query = query[:, :, : -text_seq_length], query[:, :, -text_seq_length :]
        key, encoder_key = key[:, :, : -text_seq_length], key[:, :, -text_seq_length :]
        value, encoder_value = value[:, :, : -text_seq_length], value[:, :, -text_seq_length :]

        if SP_STATE.enabled:
            # parallel dim: S -> H for attention
            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2)
            key = all_to_all_4D(key, scatter_idx=1, gather_idx=2)
            value = all_to_all_4D(value, scatter_idx=1, gather_idx=2)

            encoder_query = shrink_dim(encoder_query, dim=1)
            encoder_key = shrink_dim(encoder_key, dim=1)
            encoder_value = shrink_dim(encoder_value, dim=1)

        hidden_states, encoder_hidden_states = sliding_tile_flex_attn(
            query, key, value, 
            partial(flex_attention, block_mask=flex_attn_mask_func),
            encoder_query, encoder_key, encoder_value,
            tile_size=tile_size,
            latent_shape=latent_shape,  # arguments for tile & untile
            head_dim=1,
        )  # (B, H, S, D) -> (B, H, S, D)

        if SP_STATE.enabled:
            # parallel dim: H -> S for Dense layer
            hidden_states = all_to_all_4D(hidden_states, scatter_idx=2, gather_idx=1)
            encoder_hidden_states = all_gather(encoder_hidden_states, dim=1).contiguous()

        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        return hidden_states, encoder_hidden_states

    def _combine_attn_outputs(self, routing_score, hidden_states_list):
        return (
            routing_score[:, :, :, None, None] * # (B, H, 3) -> (B, H, 3, 1, 1)
            torch.stack(hidden_states_list, dim=2) # 3 * (B, H, S, D) -> (B, H, 3, S, D)
        ).sum(dim=2)


class HunyuanVideoFlashAttnProcessorTripleEval(HunyuanVideoFlashAttnProcessorTripleTrain):
    def __init__(self, check_input = False):
        super().__init__(check_input)

    @torch.no_grad()
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        routing_score: torch.Tensor,
        tau_sparse: float,
        # Low-resolution attention arguments
        lowres_group_info: Optional[LowresGroupInfo] = None,
        # Sliding attention arguments
        # # FlexAttention
        flex_attn_mask_func: Optional[Callable[[torch.IntTensor], torch.BoolTensor]] = None,
        # # STA
        window_size: Tuple[int, int, int] = (3, 3, 3),
        tile_size: Tuple[int, int, int] = (6, 8, 8),
        latent_shape: Tuple[int, int, int] = (30, 48, 80),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_input(hidden_states, lowres_group_info, latent_shape, window_size, tile_size)

        text_seq_length = encoder_hidden_states.shape[1]

        # 1. QKV projections
        query, key, value = self._step_to_qkv_and_unflatten(attn, hidden_states, encoder_hidden_states)

        # 2. QK normalization
        query, key = self._step_qk_norm(attn, query, key)

        # 3. Rotational positional embeddings applied to latent stream
        query, key = self._step_rotary_emb(attn, query, key, text_seq_length, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        query, key, value = self._step_encoder_to_qkv_and_concat(attn, query, key, value, encoder_hidden_states)

        # 5. Attention
        # Get routed QKV
        (
            full_query, full_key, full_value, full_head_mask,
            lowres_query, lowres_key, lowres_value, lowres_head_mask,
            sw_query, sw_key, sw_value, sw_head_mask
        ) = self._get_routed_qkv(query, key, value, routing_score, tau_sparse)

        # # 5.1 Full attention
        if full_query is None or full_query.shape[1] == 0:
            full_hidden_states = None
            full_encoder_hidden_states = None
        else:
            full_hidden_states, full_encoder_hidden_states = self._step_attention(
                full_query, full_key, full_value, attention_mask, text_seq_length
            )

        # # 5.2 Low-res attention
        if lowres_query is None or lowres_query.shape[1] == 0:
            lowres_hidden_states = None
            lowres_encoder_hidden_states = None
        else:
            lowres_hidden_states, lowres_encoder_hidden_states = self._step_lowres_attention(
                lowres_query, lowres_key, lowres_value, attention_mask,
                text_seq_length, lowres_group_info
            )

        # # 5.3 Sliding window attention
        if sw_query is None or sw_query.shape[1] == 0:
            sliding_hidden_states = None
            sliding_encoder_hidden_states = None
        else:
            sliding_hidden_states, sliding_encoder_hidden_states = self._step_sliding_attention(
                sw_query, sw_key, sw_value, attention_mask, text_seq_length,
                flex_attn_mask_func, window_size, tile_size, latent_shape,
            )

        # # 5.4 Combine attention
        # (B, h1, S, D) + (B, h2, S, D) + (B, h3, S, D) -> (B, H=h1+h2+h3, S, D)
        hidden_states = self._combine_attn_outputs(
            attn.heads, 
            [full_hidden_states, lowres_hidden_states, sliding_hidden_states], 
            [full_head_mask, lowres_head_mask, sw_head_mask]
        )

        encoder_hidden_states = self._combine_attn_outputs(
            attn.heads,
            [full_encoder_hidden_states, lowres_encoder_hidden_states, sliding_encoder_hidden_states],
            [full_head_mask, lowres_head_mask, sw_head_mask]
        )

        # 6. Output projection
        hidden_states, encoder_hidden_states = self._step_to_output(attn, hidden_states, encoder_hidden_states)

        return hidden_states, encoder_hidden_states
    
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
        top1_indice[top1_score < tau_sparse] = 0  # less confident, use full attention (H, 1)
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
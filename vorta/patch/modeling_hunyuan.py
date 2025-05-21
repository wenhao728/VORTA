#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/02/26 15:32:58
@Desc    :   
    2025/03/17: flexattn support
    2025/03/22: align with future changes in diffusers, QA
    2025/04/15: update to diffusers 0.33.1
@Ref     :   
'''
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoConditionEmbedding,
    HunyuanVideoRotaryPosEmbed,
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTransformer3DModel,
    HunyuanVideoTransformerBlock,
)
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from torch.nn import functional as F

from ..attention import (
    HunyuanVideoFlashAttnProcessor,
    HunyuanVideoFlashAttnProcessorTripleEval,
    HunyuanVideoFlashAttnProcessorTripleTrain,
)
from ..attention.sliding_attn_flex import create_sliding_tile_attn_mask_func
from ..train import load_router_checkpoint
from ..ulysses import SP_STATE
from ..utils import accumulate_loss
from .outputs import RoutedTransformerModelOutput
from .router import Router

logger = logging.getLogger(__name__)


def hunyuan_transformer_3d_forward(
    self: HunyuanVideoTransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    # 3. Attention mask preparation
    latent_sequence_length = hidden_states.shape[1]
    if SP_STATE.enabled:
        latent_sequence_length = latent_sequence_length * SP_STATE.sp_size

    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.zeros(
        batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
    )  # [B, N]

    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

    for i in range(batch_size):
        attention_mask[i, : effective_sequence_length[i]] = True
    # [B, 1, 1, N], for broadcasting across attention heads
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

    # 5. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)


def hunyuan_transformer_3d_routed_forward(
    self: HunyuanVideoTransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    self_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_losses: bool = False,
    reture_hidden_layer_distill_loss: bool = False,
    return_routing_scores: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb, token_replace_emb, clean_timesteps_emb = self.time_text_embed(timestep, pooled_projections, guidance)
    # breakpoint()

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    # 3. Attention mask preparation
    latent_sequence_length = hidden_states.shape[1]
    if SP_STATE.enabled:
        latent_sequence_length = latent_sequence_length * SP_STATE.sp_size

    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.zeros(
        batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
    )  # [B, N]

    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

    for i in range(batch_size):
        attention_mask[i, : effective_sequence_length[i]] = True
    # [B, 1, 1, N], for broadcasting across attention heads
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    # torch.distributed.breakpoint()
    if 'flex_attn_mask_func' not in self_attention_kwargs:
        flex_attn_mask_func = create_sliding_tile_attn_mask_func(
            latent_shape=self_attention_kwargs["latent_shape"],
            window_size=self_attention_kwargs["window_size"],
            tile_size=self_attention_kwargs["tile_size"],
            text_seq_length=condition_sequence_length,
            text_seq_length_no_pad=effective_condition_sequence_length[0].item(),  # adhoc: batch_size>1
            device=hidden_states.device,
        )
        self_attention_kwargs.update(flex_attn_mask_func=flex_attn_mask_func)

    reg_loss = None
    hidden_layer_distill_loss = None
    last_layer_distill_loss = None
    routing_scores = []

    # 4. Transformer blocks
    if return_losses:
        # use reference embeddings for distillation
        ref_hidden_states = hidden_states.detach().clone()
        ref_encoder_hidden_states = encoder_hidden_states.detach().clone()

    transformer_block_index = 0
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states, routing_score = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                False, # use_original_attn
                self_attention_kwargs,
                clean_timesteps_emb,
            )
            if return_routing_scores:
                routing_scores.append(routing_score.detach().cpu())

            if return_losses:
                with torch.no_grad():
                    # use `torch.no_grad()` to avoid computing gradients for the reference embeddings
                    ref_hidden_states, ref_encoder_hidden_states, _ = block(
                        ref_hidden_states,
                        ref_encoder_hidden_states,
                        temb, 
                        attention_mask, 
                        image_rotary_emb,
                        token_replace_emb,
                        first_frame_num_tokens,
                        use_original_attn=True,
                    )

                # layer_reg_loss = routing_score[:, :, 0].mean().float()  # L1 regularization
                layer_reg_loss = torch.square(routing_score[:, :, 0]).mean().float()  # L2 regularization
                reg_loss = accumulate_loss(reg_loss, layer_reg_loss)

                if reture_hidden_layer_distill_loss:
                    layer_distill_loss = F.mse_loss(ref_hidden_states.float(), hidden_states.float())
                    hidden_layer_distill_loss = accumulate_loss(hidden_layer_distill_loss, layer_distill_loss)

            transformer_block_index += 1

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states, routing_score = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                False, # use_original_attn=
                self_attention_kwargs,
                clean_timesteps_emb,
            )
            if return_routing_scores:
                routing_scores.append(routing_score.detach().cpu())
            
            if return_losses:
                with torch.no_grad():
                    # use `torch.no_grad()` to avoid computing gradients for the reference embeddings
                    ref_hidden_states, ref_encoder_hidden_states, _ = block(
                        ref_hidden_states,
                        ref_encoder_hidden_states,
                        temb, 
                        attention_mask, 
                        image_rotary_emb,
                        token_replace_emb,
                        first_frame_num_tokens,
                        use_original_attn=True,
                    )

                # layer_reg_loss = routing_score[:, :, 0].mean().float()  # L1 regularization
                layer_reg_loss = torch.square(routing_score[:, :, 0]).mean().float()  # L2 regularization
                reg_loss = accumulate_loss(reg_loss, layer_reg_loss)

                if reture_hidden_layer_distill_loss:
                    layer_distill_loss = F.mse_loss(ref_hidden_states.float(), hidden_states.float())
                    hidden_layer_distill_loss = accumulate_loss(hidden_layer_distill_loss, layer_distill_loss)

            transformer_block_index += 1

    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states, routing_score = block(
                hidden_states, 
                encoder_hidden_states, 
                temb, 
                attention_mask, 
                image_rotary_emb, 
                token_replace_emb,
                first_frame_num_tokens,
                use_original_attn=False,
                self_attention_kwargs=self_attention_kwargs,
                clean_timesteps_emb=clean_timesteps_emb,
            )
            if return_routing_scores:
                routing_scores.append(routing_score.detach().cpu())

            if return_losses:
                with torch.no_grad():
                    # use `torch.no_grad()` to avoid computing gradients for the reference embeddings
                    ref_hidden_states, ref_encoder_hidden_states, _ = block(
                        ref_hidden_states,
                        ref_encoder_hidden_states,
                        temb, 
                        attention_mask, 
                        image_rotary_emb,
                        token_replace_emb,
                        first_frame_num_tokens,
                        use_original_attn=True,
                    )

                # layer_reg_loss = routing_score[:, :, 0].mean().float()  # L1 regularization
                layer_reg_loss = torch.square(routing_score[:, :, 0]).mean().float()  # L2 regularization
                reg_loss = accumulate_loss(reg_loss, layer_reg_loss)

                if reture_hidden_layer_distill_loss:
                    layer_distill_loss = F.mse_loss(ref_hidden_states.float(), hidden_states.float())
                    hidden_layer_distill_loss = accumulate_loss(hidden_layer_distill_loss, layer_distill_loss)

            transformer_block_index += 1

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states, routing_score = block(
                hidden_states, 
                encoder_hidden_states, 
                temb, 
                attention_mask, 
                image_rotary_emb, 
                token_replace_emb,
                first_frame_num_tokens,
                use_original_attn=False,
                self_attention_kwargs=self_attention_kwargs,
                clean_timesteps_emb=clean_timesteps_emb,
            )
            if return_routing_scores:
                routing_scores.append(routing_score.detach().cpu())

            if return_losses:
                with torch.no_grad():
                    # use `torch.no_grad()` to avoid computing gradients for the reference embeddings
                    ref_hidden_states, ref_encoder_hidden_states, _ = block(
                        ref_hidden_states,
                        ref_encoder_hidden_states,
                        temb, 
                        attention_mask, 
                        image_rotary_emb,
                        token_replace_emb,
                        first_frame_num_tokens,
                        use_original_attn=True,
                    )

                # layer_reg_loss = routing_score[:, :, 0].mean().float()  # L1 regularization
                layer_reg_loss = torch.square(routing_score[:, :, 0]).mean().float()  # L2 regularization
                reg_loss = accumulate_loss(reg_loss, layer_reg_loss)

                if reture_hidden_layer_distill_loss:
                    layer_distill_loss = F.mse_loss(ref_hidden_states.float(), hidden_states.float())
                    hidden_layer_distill_loss = accumulate_loss(hidden_layer_distill_loss, layer_distill_loss)

            transformer_block_index += 1

    # 5. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    if return_losses:
        with torch.no_grad():
            # use `torch.no_grad()` to avoid computing gradients for the reference embeddings
            ref_hidden_states = self.norm_out(ref_hidden_states, temb)
            ref_hidden_states = self.proj_out(ref_hidden_states)
        last_layer_distill_loss = F.mse_loss(ref_hidden_states.float(), hidden_states.float())

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (hidden_states, reg_loss, last_layer_distill_loss, hidden_layer_distill_loss, routing_scores)

    return RoutedTransformerModelOutput(
        sample=hidden_states, 
        reg_loss=reg_loss, 
        last_layer_distill_loss=last_layer_distill_loss,
        hidden_layer_distill_loss=hidden_layer_distill_loss, 
        routing_scores=routing_scores,
    )


def hunyuan_single_block_routed_forward(
    self: HunyuanVideoSingleTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: torch.Tensor,
    image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    token_replace_emb: torch.Tensor,
    first_frame_num_tokens: int,
    use_original_attn: bool = False,
    self_attention_kwargs: Optional[Dict[str, Any]] = None,
    clean_timesteps_emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    text_seq_length = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    residual = hidden_states

    # 1. Input normalization
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    norm_hidden_states, norm_encoder_hidden_states = (
        norm_hidden_states[:, :-text_seq_length, :],
        norm_hidden_states[:, -text_seq_length:, :],
    )

    # 2. Attention
    routing_score = None
    if use_original_attn:
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            use_original_attn=True,
        )
    elif hasattr(self, "router") and self.router is not None:
        # Routing, temb.size(-1): self.norm.linear.in_features
        routing_score = self.router(clean_timesteps_emb)
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            routing_score=routing_score,
            **self_attention_kwargs,
        )
    else:
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **self_attention_kwargs,
        )

    attn_output = torch.cat([attn_output, context_attn_output], dim=1)

    # 3. Modulation and residual connection
    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
    hidden_states = hidden_states + residual

    hidden_states, encoder_hidden_states = (
        hidden_states[:, :-text_seq_length, :],
        hidden_states[:, -text_seq_length:, :],
    )

    return hidden_states, encoder_hidden_states, routing_score


def hunyuan_dual_block_routed_forward(
    self: HunyuanVideoTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    token_replace_emb: torch.Tensor,
    first_frame_num_tokens: int,
    use_original_attn: bool = False,
    self_attention_kwargs: Optional[Dict[str, Any]] = None,
    clean_timesteps_emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # 1. Input normalization
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    # 2. Joint attention
    routing_score = None
    if use_original_attn:
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
            use_original_attn=True,
        )
    elif hasattr(self, "router") and self.router is not None:
        # Routing, temb.size(-1): self.norm1.linear.in_features
        routing_score = self.router(clean_timesteps_emb)
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
            routing_score=routing_score,
            **self_attention_kwargs,
        )
    else:
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
            **self_attention_kwargs,
        )

    # 3. Modulation and residual connection
    hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
    encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

    norm_hidden_states = self.norm2(hidden_states)
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    # 4. Feed-forward
    ff_output = self.ff(norm_hidden_states)
    context_ff_output = self.ff_context(norm_encoder_hidden_states)

    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

    return hidden_states, encoder_hidden_states, routing_score


def hunyuan_rope_forward(self: HunyuanVideoRotaryPosEmbed, hidden_states: torch.Tensor) -> torch.Tensor:
    """Add Sequence parallelism to the RoPE layer."""
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    rope_sizes = [
        (num_frames * SP_STATE.sp_size) // self.patch_size_t,
        height // self.patch_size, width // self.patch_size,
    ]

    axes_grids = []
    for i in range(3):
        # Note: The following line diverges from original behaviour. We create the grid on the device, whereas
        # original implementation creates it on CPU and then moves it to device. This results in numerical
        # differences in layerwise debugging outputs, but visually it is the same.
        grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
        axes_grids.append(grid)
    grid = torch.meshgrid(*axes_grids, indexing="ij")  # [W, H, T]
    grid = torch.stack(grid, dim=0)  # [3, W, H, T]

    freqs = []
    for i in range(3):
        freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
        freqs.append(freq)

    freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
    freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)
    return freqs_cos, freqs_sin


def hunyuan_combined_embedding_forward(
    self: HunyuanVideoConditionEmbedding,
    timestep: torch.Tensor,
    pooled_projection: torch.Tensor,
    guidance: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

    pooled_projections = self.text_embedder(pooled_projection)
    conditioning = timesteps_emb + pooled_projections

    token_replace_emb = None
    if self.image_condition_type == "token_replace":
        token_replace_timestep = torch.zeros_like(timestep)
        token_replace_proj = self.time_proj(token_replace_timestep)
        token_replace_emb = self.timestep_embedder(token_replace_proj.to(dtype=pooled_projection.dtype))
        token_replace_emb = token_replace_emb + pooled_projections

    if self.guidance_embedder is not None:
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        conditioning = conditioning + guidance_emb

    return conditioning, token_replace_emb, timesteps_emb


def apply_vorta_transformer(
    model: HunyuanVideoTransformer3DModel,
    train_router: bool = False,
    checkpoint_file: Optional[os.PathLike] = None,
    attn_processor_kwargs: Optional[Dict[str, Any]] = None,
    router_dtype: Optional[torch.dtype] = None,
):
    if train_router:
        AttnProcessorCls = HunyuanVideoFlashAttnProcessorTripleTrain
    else:
        AttnProcessorCls = HunyuanVideoFlashAttnProcessorTripleEval

    model.__class__.forward = hunyuan_transformer_3d_routed_forward
    model.rope.__class__.forward = hunyuan_rope_forward
    model.time_text_embed.__class__.forward = hunyuan_combined_embedding_forward

    model_dtype = next(model.parameters()).dtype
    router_dtype = router_dtype or model_dtype
    logger.info(f"Model {model.__class__.__name__} ({model_dtype=}) is mounted with Router ({router_dtype=})")

    if attn_processor_kwargs is None:
        attn_processor_kwargs = {}
    attn_processor_kwargs.update(check_input=True)
    for block in model.transformer_blocks:
        block: HunyuanVideoTransformerBlock
        if not hasattr(block, "router"):
            embedding_dim = block.norm1.linear.in_features  # AdaLNZero, 128
            block.router = Router(
                embedding_dim=embedding_dim,
                heads=block.attn.heads,
            ).to(dtype=router_dtype)
        if train_router:
            # block.router.train()
            block.router.requires_grad_(True)

        block.attn.set_processor(AttnProcessorCls(**attn_processor_kwargs))
        attn_processor_kwargs.update(check_input=False)  # only check input for the first block

        block.__class__.forward = hunyuan_dual_block_routed_forward

    for block in model.single_transformer_blocks:
        block: HunyuanVideoSingleTransformerBlock
        if not hasattr(block, "router"):
            block.router = Router(
                embedding_dim=block.norm.linear.in_features, 
                heads=block.attn.heads,
            ).to(dtype=router_dtype)
        if train_router:
            # block.router.train()
            block.router.requires_grad_(True)

        block.attn.set_processor(AttnProcessorCls(**attn_processor_kwargs))

        block.__class__.forward = hunyuan_single_block_routed_forward

    if checkpoint_file is not None:
        load_router_checkpoint(checkpoint_file, model)

    return model


def apply_sp_flashattn_transformer(
    model: HunyuanVideoTransformer3DModel,
):
    model.__class__.forward = hunyuan_transformer_3d_forward
    model.rope.__class__.forward = hunyuan_rope_forward

    for block in model.transformer_blocks:
        block: HunyuanVideoTransformerBlock
        block.attn.set_processor(HunyuanVideoFlashAttnProcessor())

    for block in model.single_transformer_blocks:
        block: HunyuanVideoSingleTransformerBlock
        block.attn.set_processor(HunyuanVideoFlashAttnProcessor())

    return model

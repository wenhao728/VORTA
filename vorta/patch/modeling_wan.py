#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/17 19:19:35
@Desc    :   
    2025/03/22: sequence parallel with ROPE
    TODO: check if the same parameters are also used in FSDP and DeepSpeed
@Ref     :   
    All the parameters are forced to be same when the DDP object is instantiated. https://github.com/pytorch/pytorch/blob/1dba81f56dc33b44d7b0ecc92a039fe32ee80f8d/torch/nn/parallel/distributed.py#LL798C63-L798C63
'''
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers.models.transformers.transformer_wan import (
    WanRotaryPosEmbed,
    WanTransformer3DModel,
    WanTransformerBlock,
)
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from torch.nn import functional as F

from ..attention import (
    WanAttnProcessor2_0,
    WanAttnProcessorTripleEval,
    WanAttnProcessorTripleTrain,
)
from ..train import load_router_checkpoint
from ..ulysses import SP_STATE
from ..utils import accumulate_loss
from .outputs import RoutedTransformerModelOutput
from .router import Router

logger = logging.getLogger(__name__)


def wan_transformer_3d_routed_forward(
    self: WanTransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
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
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    reg_loss = None
    hidden_layer_distill_loss = None
    last_layer_distill_loss = None
    routing_scores = []

    # 4. Transformer blocks
    if return_losses:
        # use reference embeddings for distillation
        ref_hidden_states = hidden_states.detach().clone()

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states, routing_score = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb,
                temb, # temb_before_proj
                False, # use_original_attn
                self_attention_kwargs,
            )

            if return_routing_scores:
                routing_scores.append(routing_score.detach().cpu())

            if return_losses:
                with torch.no_grad():
                    # use `torch.no_grad()` to avoid computing gradients for the reference embeddings
                    ref_hidden_states, _ = block(
                        ref_hidden_states, encoder_hidden_states, timestep_proj, rotary_emb,
                        temb_before_proj=temb,
                        use_original_attn=True,
                    )

                # layer_reg_loss = routing_score[:, :, 0].mean().float()  # L1 regularization
                layer_reg_loss = torch.square(routing_score[:, :, 0]).mean().float()  # L2 regularization
                reg_loss = accumulate_loss(reg_loss, layer_reg_loss)

                if reture_hidden_layer_distill_loss:
                    layer_distill_loss = F.mse_loss(ref_hidden_states.float(), hidden_states.float())
                    hidden_layer_distill_loss = accumulate_loss(hidden_layer_distill_loss, layer_distill_loss)
    else:
        for block in self.blocks:
            hidden_states, routing_score = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb,
                temb_before_proj=temb,
                use_original_attn=False,
                self_attention_kwargs=self_attention_kwargs,
            )

            if return_routing_scores:
                routing_scores.append(routing_score.detach().cpu())

            if return_losses:
                with torch.no_grad():
                    # use `torch.no_grad()` to avoid computing gradients for the reference embeddings
                    ref_hidden_states, _ = block(
                        ref_hidden_states, encoder_hidden_states, timestep_proj, rotary_emb,
                        temb_before_proj=temb,
                        use_original_attn=True,
                    )

                # layer_reg_loss = routing_score[:, :, 0].mean().float()  # L1 regularization
                layer_reg_loss = torch.square(routing_score[:, :, 0]).mean().float()  # L2 regularization
                reg_loss = accumulate_loss(reg_loss, layer_reg_loss)

                if reture_hidden_layer_distill_loss:
                    layer_distill_loss = F.mse_loss(ref_hidden_states.float(), hidden_states.float())
                    hidden_layer_distill_loss = accumulate_loss(hidden_layer_distill_loss, layer_distill_loss)

    # 5. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    if return_losses:
        with torch.no_grad():
            # use `torch.no_grad()` to avoid computing gradients for the reference embeddings
            ref_hidden_states = (
                self.norm_out(ref_hidden_states.float()) * (1 + scale) + shift).type_as(ref_hidden_states)
            ref_hidden_states = self.proj_out(ref_hidden_states)
        last_layer_distill_loss = F.mse_loss(ref_hidden_states.float(), hidden_states.float())

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output, reg_loss, last_layer_distill_loss, hidden_layer_distill_loss, routing_scores)

    return RoutedTransformerModelOutput(
        sample=output, 
        reg_loss=reg_loss, 
        last_layer_distill_loss=last_layer_distill_loss,
        hidden_layer_distill_loss=hidden_layer_distill_loss, 
        routing_scores=routing_scores,
    )


def wan_block_routed_forward(
    self: WanTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    temb_before_proj: Optional[torch.Tensor],
    use_original_attn: bool = False,
    self_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table + temb.float()
    ).chunk(6, dim=1)

    # 1. Self-attention
    norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
    
    routing_score = None
    if not use_original_attn:
        # routing_score = self.router(temb)
        routing_score = self.router(temb_before_proj)
    if self_attention_kwargs is None:
        self_attention_kwargs = {}
    attn_output = self.attn1(
        hidden_states=norm_hidden_states,
        rotary_emb=rotary_emb, 
        routing_score=routing_score,
        use_original_attn=use_original_attn,
        **self_attention_kwargs,
    )
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
        hidden_states
    )
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

    return hidden_states, routing_score


def wan_rope_forward(self: WanRotaryPosEmbed, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
    ppf = ppf * SP_STATE.sp_size

    self.freqs = self.freqs.to(hidden_states.device)
    freqs = self.freqs.split_with_sizes(
        [
            self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
            self.attention_head_dim // 6,
            self.attention_head_dim // 6,
        ],
        dim=1,
    )

    freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
    return freqs


def apply_vorta_transformer(
    model: WanTransformer3DModel,
    train_router: bool = False,
    checkpoint_file: Optional[os.PathLike] = None,
    attn_processor_kwargs: Optional[Dict[str, Any]] = None,
    router_dtype: Optional[torch.dtype] = None,
):
    if train_router:
        AttnProcessorCls = WanAttnProcessorTripleTrain
    else:
        AttnProcessorCls = WanAttnProcessorTripleEval

    model.__class__.forward = wan_transformer_3d_routed_forward
    model.rope.__class__.forward = wan_rope_forward

    dtype = router_dtype or next(model.parameters()).dtype
    logger.info(f"Model {model.__class__.__name__} is mounted with Router({dtype=})")

    num_experts = 3
    embedding_dim = model.condition_embedder.time_proj.in_features

    if attn_processor_kwargs is None:
        attn_processor_kwargs = {}
    attn_processor_kwargs.update(check_input=True)
    for block in model.blocks:
        block: WanTransformerBlock
        if not hasattr(block, "router"):
            block.router = Router(
                embedding_dim=embedding_dim, 
                heads=block.attn1.heads,
                num_experts=num_experts,
            ).to(dtype=dtype)
        if train_router:
            # block.router.train()
            block.router.requires_grad_(True)
        
        block.attn1.set_processor(AttnProcessorCls(**attn_processor_kwargs))
        block.attn2.set_processor(WanAttnProcessor2_0())  # for sequence parallel
        attn_processor_kwargs.update(check_input=False)  # only check input for the first block

        block.__class__.forward = wan_block_routed_forward

    if checkpoint_file is not None:
        load_router_checkpoint(checkpoint_file, model)

    return model


def apply_sp_flashattn_transformer(
    model: WanTransformer3DModel,
):
    # model.__class__.forward = wan_transformer_3d_forward  # no attn mask, use original forward
    model.rope.__class__.forward = wan_rope_forward

    for block in model.blocks:
        block: WanTransformerBlock
        block.attn1.set_processor(WanAttnProcessor2_0())
        block.attn2.set_processor(WanAttnProcessor2_0())  # for sequence parallel

    return model
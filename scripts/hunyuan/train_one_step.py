#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/08 22:44:04
@Desc    :   
    2025/03/13: FSDP training with gradient accumulation
@Ref     :   
'''
import logging
from typing import Any, Dict, Iterable

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import nn
from torch.nn import functional as F

from vorta.train import (
    compute_density_for_timestep_sampling,
    get_sigmas,
    nomalize_input_latent,
    rebalance_diffusion_loss_weight,
    renormalize_uniform_sample,
)
from vorta.ulysses import SP_STATE, TrainingLog, broadcast_sp_group

logger = logging.getLogger(__name__)


def train_one_step(
    data_iterator: Iterable,
    transformer: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    precondition_outputs: bool,
    t_sampling_kwargs: Dict[str, Any],
    self_attention_kwargs: Dict[str, Any],
    loss_weights: Dict[str, float],
    training_log: TrainingLog,
    t_interval_index: int = None,
    t_num_intervals: int = 5,
    pooled_projection_dim: int = 768,
) -> None:
    optimizer.zero_grad()
    timesteps_ = None
    for i in range(gradient_accumulation_steps):
        latents, encoder_hidden_states, encoder_attn_mask = next(data_iterator)
        batch_size = latents.size(0)

        latents = nomalize_input_latent(model_name='hunyuan', latents=latents)
        noise = torch.randn_like(latents)

        if timesteps_ is None:
            u = compute_density_for_timestep_sampling(
                batch_size=batch_size * gradient_accumulation_steps, **t_sampling_kwargs
            )
            u = renormalize_uniform_sample(
                u, t_sampling_kwargs.get('generator', None), 
                n_intervals=t_num_intervals, t_interval_index=t_interval_index
            )
            indices = (u * noise_scheduler.config.num_train_timesteps).long()
            timesteps_ = noise_scheduler.timesteps[indices].to(device=latents.device)
            if SP_STATE.enabled:
                broadcast_sp_group(timesteps_)  # make sure all processes have the same timesteps
        timesteps = timesteps_[i * batch_size:(i + 1) * batch_size]

        (
            diffusion_loss_weight, 
            last_layer_distill_loss_weight, 
            hidden_layer_distill_loss_weight,
        ) = rebalance_diffusion_loss_weight(
            diffusion_loss_weight=loss_weights.get('diffusion_loss', 0.0),
            other_loss_weights=[loss_weights["last_layer_distill_loss"], loss_weights["hidden_layer_distill_loss"]],
            timesteps=timesteps,
            n_intervals=t_num_intervals,
            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
        )

        sigmas = get_sigmas(
            noise_scheduler,
            latents.device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

        pooled_projections, encoder_hidden_states = (
            encoder_hidden_states[:, 0, :pooled_projection_dim], encoder_hidden_states[:, 1:]
        )

        model_pred, reg_loss, last_layer_distill_loss, hidden_layer_distill_loss, _ = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attn_mask,
            pooled_projections=pooled_projections,
            guidance=torch.tensor([1000.0], device=noisy_model_input.device, dtype=torch.bfloat16), # hunyuan
            self_attention_kwargs=self_attention_kwargs,
            return_dict=False,
            return_routing_scores=False,
            return_losses=True,
            reture_hidden_layer_distill_loss=hidden_layer_distill_loss_weight > 0.0,
        )

        reg_loss_weight = loss_weights.get('reg_loss', 0.0)
        if  diffusion_loss_weight > 0.0:
            # Loss computation
            if precondition_outputs:
                model_pred = noisy_model_input - model_pred * sigmas
                target = latents
            else:
                target = noise - latents
            diffusion_loss = (
                F.mse_loss(model_pred.float(), target.float()) * diffusion_loss_weight / gradient_accumulation_steps)
        else:
            diffusion_loss = 0.0
        reg_loss = reg_loss * reg_loss_weight / gradient_accumulation_steps
        last_layer_distill_loss = last_layer_distill_loss * last_layer_distill_loss_weight / gradient_accumulation_steps

        loss = diffusion_loss + reg_loss + last_layer_distill_loss
        if hidden_layer_distill_loss_weight > 0.0:
            hidden_layer_distill_loss = (
                hidden_layer_distill_loss * hidden_layer_distill_loss / gradient_accumulation_steps)
            loss += hidden_layer_distill_loss

        loss.backward()

        # Log the loss
        if i < gradient_accumulation_steps - 1:
            training_log.update(
                timesteps, loss, diffusion_loss, reg_loss, last_layer_distill_loss, hidden_layer_distill_loss, 
                grad_norm=None
            )

    # Optimization
    grad_norm = transformer.clip_grad_norm_(max_grad_norm)
    optimizer.step()
    lr_scheduler.step()

    training_log.update(
        timesteps, loss, diffusion_loss, reg_loss, last_layer_distill_loss, hidden_layer_distill_loss, 
        grad_norm=grad_norm
    )
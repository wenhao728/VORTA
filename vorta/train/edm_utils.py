import math
from typing import List, Literal

import torch

from ..constants import SupportedModels


def nomalize_input_latent(
    model_name: SupportedModels,
    latents: torch.Tensor,
):
    if model_name == "hunyuan":
        # return latents * 0.476986
        return latents  # normalization has been done during preprocess
    elif model_name == "wan":
        return latents  # normalization has been done during preprocess
    else:
        raise NotImplementedError(f"Unsupported model: {model_name}")


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size,),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2)**2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
    return u


def get_sigmas(noise_scheduler, device, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def renormalize_uniform_sample(
    u: torch.Tensor,
    generator: torch.Generator,
    n_intervals: int = 5,
    t_interval_index = None,
) -> torch.Tensor:
    if t_interval_index is None:
        t_interval_index = torch.randint(0, n_intervals, (1,), device="cpu", generator=generator)
    else:
        t_interval_index = torch.tensor(t_interval_index, device="cpu")
    t_interval = t_interval_index / n_intervals, (t_interval_index + 1) / n_intervals

    u = torch.clamp(
        u * (t_interval[1] - t_interval[0]) + t_interval[0], 
        min=t_interval[0], max=t_interval[1],
    )

    return u


def rebalance_diffusion_loss_weight(
    diffusion_loss_weight: float,
    other_loss_weights: List[float],
    timesteps: torch.Tensor, 
    n_intervals: int = 5,
    num_train_timesteps: int = 1000,
) -> List[float]:
    t = timesteps[0]

    if diffusion_loss_weight == 0:
        return diffusion_loss_weight, *other_loss_weights

    if t < num_train_timesteps / n_intervals:
        diffusion_loss_weight = 0
        
        num_other_losses = sum([w > 0 for w in other_loss_weights])
        other_loss_weights = [w * (1.0 + 1 / num_other_losses) for w in other_loss_weights]
    
    return diffusion_loss_weight, *other_loss_weights
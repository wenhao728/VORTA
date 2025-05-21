from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from diffusers.utils import BaseOutput


@dataclass
class RoutedTransformerModelOutput(BaseOutput):
    sample: torch.Tensor  # noqa: F821
    reg_loss: Optional[torch.Tensor] = None
    last_layer_distill_loss: Optional[torch.Tensor] = None
    hidden_layer_distill_loss: Optional[torch.Tensor] = None
    routing_scores: Optional[List[torch.Tensor]] = None


@dataclass
class VideoPipelineOutput(BaseOutput):
    r"""
    Output class for HunyuanVideo pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor
    routing_scores: Optional[List[List[torch.Tensor]]] = None
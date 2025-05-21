from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.nn import functional as F


@dataclass
class LowresGroupInfo:
    center_indices: torch.Tensor
    margin_indices: torch.Tensor
    num_unpooled_tokens_per_group: int


def get_group_info(
    latent_video_shape: Tuple[int, int, int],
    compress_window_size: Tuple[int, int, int],
    reduction_rate: float = 0.5,
    device: torch.device = torch.device('cpu'),
) -> LowresGroupInfo:
    """Generate group indices for a video tensor of shape video_shape=(frame, height, width) by partitioning it into 
    non-overlapping groups (windows) of size (f_window, h_window, w_window). The indices within each group preserve the 
    raster order as in the original 3D tensor.

    Returns:
        torch.Tensor: A tensor of shape (num_groups, group_size), where:
            num_groups = (frame // f_window) * (height // h_window) * (width // w_window)
            group_size = f_window * h_window * w_window
    """
    # Create a 3D tensor of indices in raster order
    frame, height, width = latent_video_shape
    indices = torch.arange(frame * height * width).reshape(frame, height, width)

    # Compute the number of groups along each dimension
    f_window, h_window, w_window = compress_window_size
    f_groups = frame // f_window
    h_groups = height // h_window
    w_groups = width // w_window
    num_groups = f_groups * h_groups * w_groups
    indices = indices[:f_groups * f_window, :h_groups * h_window, :w_groups * w_window]

    # Reshape the indices tensor to separate group windows
    indices_blocks = indices.reshape(f_groups, f_window, h_groups, h_window, w_groups, w_window)

    # Permute dimensions so that the group indices are together
    indices_blocks = indices_blocks.permute(0, 2, 4, 1, 3, 5)

    # Flatten each group so that each row corresponds to one group.
    group_indices = indices_blocks.reshape(num_groups, -1).to(device)

    center_idx = f_window // 2 * h_window * w_window + h_window // 2 * w_window + w_window // 2
    center_indices = group_indices[:, center_idx:center_idx + 1]
    margin_indices = torch.cat((group_indices[:, :center_idx], group_indices[:, center_idx + 1:]), dim=1)
    num_unpooled_tokens_per_group = int(f_window * h_window * w_window * (1 - reduction_rate)) - 1

    return LowresGroupInfo(
        center_indices=center_indices,
        margin_indices=margin_indices,
        num_unpooled_tokens_per_group=num_unpooled_tokens_per_group,
    )

@dataclass
class MatchingResults:
    unpooled_argsort_sim: torch.Tensor
    pooled_argsort_sim: torch.Tensor


def pool_sequence_by_similarity(
    hidden_states: torch.Tensor,
    lowres_group_info: LowresGroupInfo,
    matching_results: Optional[MatchingResults] = None,
) -> Tuple[torch.Tensor, MatchingResults]:
    """Pool hidden states of a sequence of entities into a single representative value per group based on similarity.

    Args:
        hidden_states (torch.Tensor): Hidden states of shape (batch_size, num_heads, sequence_length, channels).
        lowres_group_info (LowresGroupInfo): 
            center_indices (torch.Tensor): Group center indices of shape (num_groups, 1).
            margin_indices (torch.Tensor): Group margin indices of shape (num_groups, group_size - 1).
            num_unpooled_tokens_per_group (int): Number of unpooled tokens per group.
        matching_results (MatchingResults, optional): Precomputed similarity indices for margin tokens.
            If None, it will be computed. Defaults to None.

    Returns:
        torch.Tensor: Compressed hidden states of shape (batch_size, num_groups, num_heads, channels).
    """
    channels = hidden_states.shape[-1]

    # 1. Slice hidden states into center and margin tokens
    # (batch_size, num_heads, num_groups, 1, channels)
    center_tokens = hidden_states[:, :, lowres_group_info.center_indices, :]
    # (batch_size, num_heads, num_groups, group_size - 1, channels)
    margin_tokens = hidden_states[:, :, lowres_group_info.margin_indices, :]
    
    if matching_results is None:
        # 2. Compute similarity
        # (batch_size, num_heads, num_groups, group_size - 1)
        similarity = torch.einsum(
            "bhgd,bhgmd->bhgm", 
            F.normalize(center_tokens.squeeze(-2), p=2, dim=-1),
            F.normalize(margin_tokens, p=2, dim=-1), 
            # center_tokens.squeeze(-2), margin_tokens, # cosine similarity without normalization
        )
        # 3. Sort indices and split: least similar tokens at the start -> unpooled tokens
        argsort_sim = similarity.argsort(dim=-1, descending=False)  # cosine
        # argsort_sim = similarity.argsort(dim=-1, descending=True)  # lp distance

        num_unpooled_tokens_per_group = lowres_group_info.num_unpooled_tokens_per_group
        # (batch_size, num_heads, num_groups, num_unpooled_tokens_per_group)
        # (batch_size, num_heads, num_groups, group_size - 1 - num_unpooled_tokens_per_group)
        unpooled_argsort_sim, pooled_argsort_sim = argsort_sim.split_with_sizes(
            [num_unpooled_tokens_per_group, argsort_sim.shape[-1] - num_unpooled_tokens_per_group], dim=-1,
        )
        matching_results = MatchingResults(unpooled_argsort_sim, pooled_argsort_sim)

    # 4. Gather unpooled and pooled margin tokens
    # (batch_size, num_heads, num_groups, num_unpooled_tokens_per_group, channels)
    unpooled_margin_tokens = torch.gather(
        margin_tokens, dim=-2, 
        index=matching_results.unpooled_argsort_sim.unsqueeze(-1).expand(-1, -1, -1, -1, channels)
    )
    pooled_hidden_states = torch.cat(
        (center_tokens.flatten(2, 3), unpooled_margin_tokens.flatten(2, 3)), dim=-2)
    return pooled_hidden_states, matching_results


def unpool_sequence_by_similarity(
    pooled_hidden_states: torch.Tensor,
    lowres_group_info: LowresGroupInfo,
    matching_results: MatchingResults,
) -> torch.Tensor:
    """Reconstruct the original sequence of hidden states from the pooled representation.

    Args:
        pooled_hidden_states (torch.Tensor): Shape (batch_size, num_heads, pooled_sequence_length, channels).
        lowres_group_info (LowresGroupInfo): 
            center_indices (torch.Tensor): Shape (num_groups, 1).
            margin_indices (torch.Tensor): Shape (num_groups, group_size - 1).
            num_unpooled_tokens_per_group (int): Number of unpooled tokens per group.
        matching_results (MatchingResults): Precomputed similarity indices for margin tokens.
            unpooled_argsort_sim: (batch_size, num_heads, num_groups, num_unpooled_tokens_per_group)
            pooled_argsort_sim:   (batch_size, num_heads, num_groups, group_size - 1)

    Returns:
        torch.Tensor: Unpooled hidden states of shape (batch_size, num_heads, sequence_length, channels).
    """
    batch_size, num_heads, pooled_sequence_length, channels = pooled_hidden_states.shape
    num_center_tokens = num_groups = lowres_group_info.center_indices.shape[0]
    group_size = lowres_group_info.center_indices.shape[1] + lowres_group_info.margin_indices.shape[1]
    num_pooled_tokens_per_group = (
        lowres_group_info.margin_indices.shape[1] - lowres_group_info.num_unpooled_tokens_per_group)
    sequence_length = group_size * num_groups

    center_tokens, unpooled_margin_tokens = pooled_hidden_states.split_with_sizes(
        [num_center_tokens, pooled_sequence_length - num_groups], dim=-2)
    # copy center tokens to pooled margin tokens
    pooled_margin_tokens = torch.repeat_interleave(center_tokens, num_pooled_tokens_per_group, dim=-2)

    unpooled_margin_indices = torch.gather(
        lowres_group_info.margin_indices[None, None, :, :].expand(batch_size, num_heads, -1, -1), 
        dim=-1, index=matching_results.unpooled_argsort_sim,
    ).flatten(2, 3)
    pooled_margin_indices = torch.gather(
        lowres_group_info.margin_indices[None, None, :, :].expand(batch_size, num_heads, -1, -1), 
        dim=-1, index=matching_results.pooled_argsort_sim,
    ).flatten(2, 3)

    # scatter center, unpooled and pooled margin tokens to hidden states
    hidden_states = torch.zeros(
        batch_size, num_heads, sequence_length, channels,
        device=pooled_hidden_states.device, dtype=pooled_hidden_states.dtype
    )
    hidden_states.scatter_(
        dim=-2, index=unpooled_margin_indices[..., None].expand(-1, -1, -1, channels), 
        src=unpooled_margin_tokens
    )
    hidden_states.scatter_(
        dim=-2, index=pooled_margin_indices[..., None].expand(-1, -1, -1, channels), 
        src=pooled_margin_tokens
    )
    hidden_states.scatter_(
        dim=-2, index=lowres_group_info.center_indices[None, None].expand(batch_size, num_heads, -1, channels),
        src=center_tokens
    )
    return hidden_states



if __name__ == "__main__":
    # test the functions
    latent_video_shape = (4, 6, 4)
    compress_window_size = (2, 3, 2)

    print("latent_video_shape:", latent_video_shape)
    print("compress_window_size:", compress_window_size)
    hidden_states = torch.arange(
        latent_video_shape[0] * latent_video_shape[1] * latent_video_shape[2]).reshape(1, 1, -1, 1)
    print("hidden_states:\n", hidden_states.view(*latent_video_shape))

    group_info = get_group_info(
        latent_video_shape=latent_video_shape,
        compress_window_size=compress_window_size,
    )
    print(f"{group_info.num_unpooled_tokens_per_group=}")
    print('center indices:\n', group_info.center_indices.squeeze())
    print('margin indices:\n', group_info.margin_indices)

    pooled_hidden_states, matching_results = pool_sequence_by_similarity(
        hidden_states=hidden_states,
        lowres_group_info=group_info,
    )
    print(
        pooled_hidden_states.shape, 
        matching_results.unpooled_argsort_sim.shape, 
        matching_results.pooled_argsort_sim.shape
    )

    unpooled_hidden_states = unpool_sequence_by_similarity(
        pooled_hidden_states=pooled_hidden_states,
        lowres_group_info=group_info,
        matching_results=matching_results,
    )
    print(unpooled_hidden_states.shape)
    print(unpooled_hidden_states.squeeze().view(*latent_video_shape))
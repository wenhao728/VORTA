#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/12 19:17:36
@Desc    :   
    2025/03/14: QA & remove attention_mask
@Ref     :   Copy from https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/utils/communications.py
'''
from ..ulysses import SP_STATE, all_to_all


def prepare_sequence_parallel_data(
    hidden_states, 
    encoder_hidden_states, 
    encoder_attention_mask
):
    sp_size = SP_STATE.sp_size

    if sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            # attention_mask,
            encoder_attention_mask,
        )

    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
    encoder_hidden_states = all_to_all(encoder_hidden_states.repeat(1, sp_size, 1), scatter_dim=1, gather_dim=0)
    # attention_mask = all_to_all(attention_mask.repeat(1, sp_size, 1, 1), scatter_dim=1, gather_dim=0)
    if encoder_attention_mask is not None:
        encoder_attention_mask = all_to_all(encoder_attention_mask.repeat(1, sp_size), scatter_dim=1, gather_dim=0)

    return hidden_states, encoder_hidden_states, encoder_attention_mask


def sp_parallel_dataloader_wrapper(dataloader, device, train_batch_size, sp_size, train_sp_batch_size):
    if train_batch_size * sp_size < train_sp_batch_size:
        raise ValueError(f"{train_batch_size=} * {sp_size=} should be greater than {train_sp_batch_size=}")

    while True:
        for data_item in dataloader:
            latents, cond, cond_mask = data_item
            # latents, cond, attn_mask, cond_mask = data_item
            latents = latents.to(device)
            cond = cond.to(device)
            # attn_mask = attn_mask.to(device)
            if cond_mask is not None:
                cond_mask = cond_mask.to(device)

            frame = latents.shape[2]
            if frame == 1:
                yield latents, cond, cond_mask
                # yield latents, cond, attn_mask, cond_mask
            else:
                latents, cond, cond_mask = prepare_sequence_parallel_data(latents, cond, cond_mask)
                # latents, cond, attn_mask, cond_mask = prepare_sequence_parallel_data(
                #     latents, cond, attn_mask, cond_mask)

                for iter in range(train_batch_size * sp_size // train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size

                    yield (
                        latents[st_idx:ed_idx],
                        cond[st_idx:ed_idx],
                        # attn_mask[st_idx:ed_idx],
                        None if cond_mask is None else cond_mask[st_idx:ed_idx],
                    )
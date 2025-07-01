#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/19 22:38:34
@Desc    :   
    Wan 2.1: 81 frames -> 21 latent frames
    Bux: decord issue -> core dump
@Ref     :   
    https://github.com/dmlc/decord/issues/293
'''
import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import decord
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from diffusers.models.autoencoders import AutoencoderKLWan

project_dir = str(Path(__file__).resolve().parents[1])
if project_dir not in sys.path:
    sys.path.append(project_dir)

from vorta.utils import arg_to_json, setup_logging

decord.bridge.set_bridge('torch')
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--annotation_file', type=Path, default=Path('/raid/data/datasets/HD-Mixkit-resized-2k/videos2caption.json'))
    parser.add_argument(
        '--input_dir', type=Path, default=Path('/raid/data/datasets/HD-Mixkit-resized-2k/480p'))
    parser.add_argument(
        '--output_dir', type=Path, default=Path('/raid/data/datasets/HD-Mixkit-resized-2k/Finetune-Wan_2_1'))
    parser.add_argument('--pretrained_model_name_or_path', type=Path, default='Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


class VideoDataset(Dataset):

    def __init__(
        self,
        annotation_file,
        input_dir,
        max_num_frames=81,
    ):
        with open(annotation_file, 'r') as f:
            self.annotation = json.load(f)
        self.input_dir = input_dir
        self.max_num_frames = max_num_frames

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        item = self.annotation[idx]
        video_path = str(self.input_dir / item['video_path'])
        video_reader = decord.VideoReader(video_path)
        frames: torch.Tensor = video_reader.get_batch(range(len(video_reader)))
        frames = frames[:self.max_num_frames]  # (T, H, W, C), uint8
        frames = (frames.permute(3, 0, 1, 2).to(dtype=torch.float32) - 127.5) / 127.5  # (C, T, H, W), float32
        return {
            'frames': frames,
            'latent_paths': item['latent_path'],
        }


@torch.no_grad()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_file=args.output_dir / 'video_to_latents.log')
    args.output_dir = args.output_dir / 'latent'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(arg_to_json(args))

    # load dataset
    logger.info(f'Loading dataset from {args.annotation_file}')
    dataset = VideoDataset(args.annotation_file, args.input_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # collate_fn=VideoDataset.collate_fn,
        shuffle=False,
    )

    # load model
    logger.info(f'Loading model from {args.pretrained_model_name_or_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    vae = AutoencoderKLWan.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    vae.to(device=device, dtype=dtype)
    vae.eval()
    logger.info(f'Model loaded to {device} with dtype {dtype}')

    latent_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latent_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)

    # breakpoint()
    logger.info('Start extracting vae latents')
    for batch in tqdm(dataloader):
        frames = batch['frames'].to(device, dtype)
        latent_paths = batch['latent_paths']
        latent_exist = [(args.output_dir / latent_path).exists() for latent_path in latent_paths]
        if all(latent_exist):
            continue

        latents = vae.encode(frames).latent_dist.mode()
        latents = (latents - latent_mean) / latent_std

        for latent, latent_path in zip(latents, latent_paths):
            torch.save(latent, args.output_dir / latent_path)

    logger.info('Finished extracting latents')


if __name__ == '__main__':
    main()

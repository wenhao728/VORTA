#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2025/03/19 22:31:26
@Desc    :   
    480: 832x480
    720: 1280x720
    num_frames: 81 / fps: 15 = 5.4s

    Mixkit: 1080, 2634 samples
@Ref     :   
'''
import copy
import json
import logging
import random
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import decord
import numpy as np
from torchvision.io import write_video
from tqdm import tqdm

project_dir = str(Path(__file__).resolve().parents[1])
if project_dir not in sys.path:
    sys.path.append(project_dir)

from vorta.utils import arg_to_json, setup_logging

decord.bridge.set_bridge('torch')
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--annotation_file', type=Path, default=Path('/raid/data/datasets/HD-Mixkit-annotation/v1.1.0_HQ_part1.json'))
    parser.add_argument(
        '--input_dir', type=Path, default=Path('/raid/data/datasets/HD-Mixkit-raw'))
    parser.add_argument(
        '--output_dir', type=Path, default=Path('/raid/data/datasets/HD-Mixkit-resized-2k/480p'))
    parser.add_argument('--output_fps', type=float, default=15)
    parser.add_argument('--output_num_frames', type=int, default=81)
    parser.add_argument('--output_hw', type=int, nargs=2, default=[480, 832])
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


def prepare_attntions(args):
    with open(args.annotation_file, 'r') as f:
        annotations = json.load(f)
    logger.info(f'Number of videos: {len(annotations)}')

    annotations_ = []
    for annotation in annotations:
        src_video_path = annotation['path']

        # breakpoint()
        path = src_video_path.split('/')[-1].split('_')[0]

        src_num_frames = int(annotation['duration'] * annotation['fps'])
        frame_stride = annotation['fps'] / args.output_fps
        num_clips = src_num_frames // (frame_stride * args.output_num_frames)

        max_start_frame = int(src_num_frames - num_clips * frame_stride * args.output_num_frames)
        start_frame = random.randint(0, max_start_frame)

        num_captions=len(annotation['cap'])

        for j in range(int(num_clips)):
            start = start_frame + j * args.output_num_frames * frame_stride
            step = frame_stride
            end = start + args.output_num_frames * frame_stride
            input_frames = np.clip(np.arange(start, end, step), 0, src_num_frames - 1).astype(np.int32).tolist()

            caption_idx = random.randint(0, num_captions - 1)
            annotations_.append(dict(
                src_video_path=f"all_{src_video_path.split('_')[0]}.mp4",
                src_num_clips=int(num_clips),
                src_clip_idx=j,
                src_frames_idx=input_frames,
                length=args.output_num_frames,
                video_path=f"{path}_clip_{j}.mp4",
                latent_path=f"{path}_clip_{j}.pt",
                prompt_embed_path=f"{path}_cap_{caption_idx}.pt",
                caption=annotation['cap'][caption_idx],
            ))
    return annotations_


def resize_video(arg_tuple):
    input_video_path, output_video_path, src_frames_idx, output_hw, output_fps = arg_tuple
    if output_video_path.exists():
        return input_video_path, 'skipped', 'output exists'
    
    video_reader = decord.VideoReader(
        str(input_video_path.with_suffix('.mp4')), 
        num_threads=1, width=output_hw[1], height=output_hw[0]
    )
    video_frames = video_reader.get_batch(src_frames_idx)
    # breakpoint()
    write_video(
        output_video_path.with_suffix('.mp4'), 
        video_frames, 
        int(output_fps), 
        video_codec="libx264", 
        options={
            "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
            "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        }
    )
    return input_video_path, 'success', None


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_file=args.output_dir.parent / 'resize_videos.log')

    # Load annotations
    annotations = prepare_attntions(args)

    # Save annotations
    annotations_ = copy.deepcopy(annotations)
    for i in range(len(annotations_)):
        annotations_[i].pop('src_frames_idx')
    with open(args.output_dir.parent / 'videos2caption_temp.json', 'w') as f:
        json.dump(annotations_, f, indent=2)
    logger.info(f'Number of clips: {len(annotations)}')

    # Prepare arguments for parallel processing
    process_args = [
        (
            args.input_dir / annotation['src_video_path'], 
            args.output_dir / annotation['video_path'],
            annotation['src_frames_idx'],
            args.output_hw,
            args.output_fps,
        ) 
        for annotation in annotations
    ]

    # resize_video(process_args[0])
    # breakpoint()

    successful = 0
    skipped = 0
    failed = []

    with tqdm(total=len(process_args), desc="Converting videos", dynamic_ncols=True) as pbar:
        # Use max_workers as specified or default to CPU count
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(resize_video, arg): arg[0] for arg in process_args}

            # Process completed tasks
            for future in as_completed(future_to_file):
                filename, status, message = future.result()
                if status == "success":
                    successful += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed.append((filename, message))
                pbar.update(1)

    # Print final summary
    print(f"\nDone! Processed: {successful}, Skipped: {skipped}, Failed: {len(failed)}")
    if failed:
        print("Failed files:")
        for fname, error in failed:
            print(f"- {fname}: {error}")


if __name__ == '__main__':
    main()
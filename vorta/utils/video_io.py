import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import decord  # decord after torch, otherwise core dump: https://github.com/dmlc/decord/issues/293
from PIL import Image, ImageSequence
from torchvision.io import write_video

decord.bridge.set_bridge('native')
logger = logging.getLogger(__name__)
_supported_image_suffix = ['.jpg', '.jpeg', '.png']
_supported_video_suffix = ['.mp4', '.gif']


def _load_video_from_image_dir(video_dir: Path) -> List[Image.Image]:
    logger.debug(f'Loading video from image directory: {video_dir}')
    frames = []
    for file in sorted(video_dir.iterdir()):
        if file.suffix not in _supported_image_suffix:
            logger.debug(f'Skipping file: {file}')
            continue
        frame = Image.open(file).convert('RGB')
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f'No image found in {video_dir}, supported image suffix: {_supported_image_suffix}')
    
    return frames


def _load_video_from_video_file(video_file: Path) -> List[Image.Image]:
    logger.debug(f'Loading video from video file: {video_file}')
    if video_file.suffix == '.mp4':
        video_reader = decord.VideoReader(str(video_file), num_threads=1)
        frames = []
        for i in range(len(video_reader)):
            frames.append(Image.fromarray(video_reader[i].asnumpy()))
        return frames

    elif video_file.suffix == '.gif':
        frames = []
        for f in ImageSequence.Iterator(Image.open(video_file)):
            frame = f.convert('RGB')
            frames.append(frame)
        return frames

    else:
        raise NotImplementedError(
            f'Unsupported video file: {video_file}, supported suffix: {_supported_video_suffix}')


def load_video(video_file_or_dir: Path, start_frame: int = 0, stride: int = 1) -> List[Image.Image]:
    """
    Args:
        video_file_or_dir (Path): path to video file or directory containing images
        start_frame (int): start frame index
        stride (int): stride for frame sampling
    Returns:
        List[Image.Image]: list of frames, RGB
    """
    if not video_file_or_dir.exists():
        logger.debug(f'Video file or directory does not exist: {video_file_or_dir}, trying to find alternative')
        for suffix in _supported_video_suffix:
            if (video_file_or_dir.with_suffix(suffix)).exists():
                video_file_or_dir = video_file_or_dir.with_suffix(suffix)
                logger.debug(f'Found video file: {video_file_or_dir}')
                break
        else:
            raise FileNotFoundError(f'Reference video: {video_file_or_dir} does not exist')

    if video_file_or_dir.is_dir():
        buffer = _load_video_from_image_dir(video_file_or_dir)
    elif video_file_or_dir.is_file():
        buffer = _load_video_from_video_file(video_file_or_dir)
    else:
        # should not reach here
        raise NotImplementedError(f'{video_file_or_dir} is not a valid file or directory')
    
    video = buffer[start_frame::stride]
    logger.debug(f'Raw video frames: {len(buffer)}, sampled video frames: {len(video)}')
    logger.debug(f'Frame size: {video[0].size}')
    return video


def _format_video_pt(video_pt: torch.Tensor) -> torch.Tensor:
    assert video_pt.dim() == 4 and video_pt.size(1) == 3, \
        f"Video should have shape (T, 3, H, W), but got {video_pt.size()}"
    assert video_pt.max() <= 1.0 and video_pt.min() >= 0.0, \
        f"Video should be in range [0, 1], but got {video_pt.min()} - {video_pt.max()}"

    video_pt = (video_pt.cpu() * 255).to(torch.uint8).permute(0, 2, 3, 1)  # (T, H, W, C)
    return video_pt


def _format_video_pil(video_pil: List[Image.Image]) -> torch.Tensor:
    for i in range(len(video_pil)):
        video_pil[i] = np.array(video_pil[i].convert('RGB'))  # (H, W, C)

    return torch.from_numpy(np.stack(video_pil, axis=0))  # (T, H, W, C)


def _format_video_np(video_np: np.ndarray) -> torch.Tensor:
    assert video_np.ndim == 4 and video_np.shape[1] == 3, \
        f"Video should have shape (T, 3, H, W), but got {video_np.shape}"
    assert video_np.max() <= 1 and video_np.min() >= 0, \
        f"Video should be in range [0, 1], but got {video_np.min()} - {video_np.max()}"

    return torch.from_numpy((video_np * 255).astype(np.uint8)).permute(0, 2, 3, 1)  # (T, H, W, C)


def save_video(
    filename: os.PathLike, 
    video: Union[np.ndarray, torch.Tensor, List[Image.Image]], 
    fps: int,
) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(video, torch.Tensor):
        video = _format_video_pt(video)
    elif isinstance(video, list) and isinstance(video[0], Image.Image):
        video = _format_video_pil(video)
    elif isinstance(video, np.ndarray):
        video = _format_video_np(video)
    else:
        raise ValueError(f'Unsupported video type: {type(video)}')

    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }
    write_video(filename.with_suffix('.mp4'), video, fps, video_codec=video_codec, options=video_options)

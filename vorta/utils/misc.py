import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def prompt_to_file_name(text_prompt: str, prefix=None, suffix=None, max_str_len=20) -> str:
    file_name = re.sub(r'[^\w\s]', '', text_prompt).strip()
    file_name = re.sub(r'\s+', '-', file_name).strip()
    file_name = file_name.lower()

    if len(file_name) > max_str_len:
        file_name = file_name[:max_str_len]
    if prefix is not None:
        file_name = f'{prefix:03d}-{file_name}'
    if suffix is not None:
        file_name += f'-{suffix:02d}'
    return file_name


def get_cuda_memory_usage(device: torch.device) -> float:
    free, total = torch.cuda.mem_get_info(device)
    mem_used_gb = (total - free) / 1024 ** 3
    logger.debug(f"CUDA memory used: {mem_used_gb:.2f} GB")
    return mem_used_gb


def format_metrics_to_gb(item):
    g_gigabyte = 1024 ** 3
    return round(item / g_gigabyte, ndigits=4)


def parent_to_ckpt_dir(resume: str, ckpt_dir: Path) -> Tuple[Optional[Path], int]:
    if resume is None:
        return None, 0
    elif resume == 'latest':
        ckpts = sorted(ckpt_dir.glob('step-*'), key=lambda x: int(x.name.split('-')[1].split('.')[0]), reverse=True)
        if len(ckpts) == 0:
            logger.warning(f'No checkpoint found in {ckpt_dir}')
            return None, 0
        return ckpts[0], int(ckpts[0].name.split('-')[1].split('.')[0])
    else:
        ckpt = ckpt_dir / resume
        if not ckpt.exists():
            raise FileNotFoundError(f'Checkpoint {ckpt} does not exist')
        return ckpt, int(resume.split('-')[1].split('.')[0])


def str_to_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == 'fp32':
        return torch.float32
    elif dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return 'fp32'
    elif dtype == torch.float16:
        return 'fp16'
    elif dtype == torch.bfloat16:
        return 'bf16'
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def accumulate_loss(current_loss, new_loss):
    return new_loss if current_loss is None else current_loss + new_loss
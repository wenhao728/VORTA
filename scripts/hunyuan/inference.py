#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/12/18 22:51:33
@Desc    :   
@Ref     :   
'''
import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from PIL import Image
from tqdm.auto import tqdm

project_dir = str(Path(__file__).resolve().parents[2])
if project_dir not in sys.path:
    sys.path.append(project_dir)

from vorta.patch.modeling_hunyuan import (
    apply_sp_flashattn_transformer,
    apply_vorta_transformer,
)
from vorta.patch.pipeline_hunyuan import sp_pipeline_call, vorta_pipeline_call
from vorta.patch.utils import hunyuan_pixel2token, prepare_hunyuan_self_attn_kwargs
from vorta.utils import (
    arg_to_json,
    parent_to_ckpt_dir,
    prompt_to_file_name,
    setup_logging,
    str_to_dtype,
)

logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--log_level", type=lambda s: str(s).upper(), default="INFO")
    parser.add_argument("--pretrained_model_path", type=str, required=True)

    # data
    parser.add_argument('--val_data_json_file', type=Path, default=None, help='Path to the validation data json file')
    parser.add_argument("--output_dir", type=Path, help="Path to log file", required=True)
    parser.add_argument("--output_fps", type=int, default=24)

    # inference
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--enable_cpu_offload", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--cfg_scale", type=float, default=6.0,
        help="The embedded guidance scale as a condition for the model. The default value is 6.0."
    )
    parser.add_argument("--video_size", type=int, nargs=3, default=[117, 720, 1280])
    parser.add_argument('--noise_scheduler_shift', type=float, default=5.0, help='5.0 for 480p, 7.0 for 720p')

    # patch
    parser.add_argument('--resume_dir', type=Path, default=None, help='Path to the checkpoint to resume training')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume training, step-*')
    parser.add_argument("--native_attention", action="store_true", help="No patch")
    parser.add_argument('--tau_sparse', type=float, default=0.3, help='Sparse routing threshold')
    args = parser.parse_args()
    
    # model
    args.resume, _ = parent_to_ckpt_dir(args.resume, args.resume_dir)
    # output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.latent_shape = hunyuan_pixel2token(args.video_size)

    return args


def load_text_prompts(args: Namespace):
    with open(args.val_data_json_file, "r") as f:
        data_config = json.load(f)

    return data_config


def load_model(args: Namespace, device: torch.device):
    logger.info(f"Loading from {args.pretrained_model_path}")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipeline: HunyuanVideoPipeline = HunyuanVideoPipeline.from_pretrained(
        args.pretrained_model_path, transformer=transformer,
        # tokenizer=None, text_encoder=None, tokenizer_2=None, text_encoder_2=None,
        torch_dtype=torch.float16,
    )
    pipeline.scheduler._shift = args.noise_scheduler_shift
    pipeline.vae.enable_tiling()
    logger.info("Pipeline loaded")

    if args.native_attention:
        apply_sp_flashattn_transformer(pipeline.transformer)
        self_attention_kwargs = None
        pipeline.__class__.__call__ = sp_pipeline_call

    else:
        assert args.resume is not None, "Resume path is required for vorta attention"
        # training configuration
        with open(args.resume_dir / "config.json", "r") as f:
            model_config = json.load(f)

        apply_vorta_transformer(
            pipeline.transformer,
            checkpoint_file=args.resume / 'router.pt',
            router_dtype=torch.bfloat16,
        )
        self_attention_kwargs = prepare_hunyuan_self_attn_kwargs(
            model_config['self_attention_kwargs'], device, args.tau_sparse)
        pipeline.__class__.__call__ = vorta_pipeline_call

    return pipeline, self_attention_kwargs


def main():
    args = parse_args()
    setup_logging(
        log_level=args.log_level,
        output_file=(args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")).with_suffix('.log'),
    )
    logger.info(arg_to_json(args))
    logger.info(f"Visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")

    # load text prompts
    data_config = load_text_prompts(args)

    # load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline, self_attention_kwargs = load_model(args, device)
    if args.enable_cpu_offload:
        pipeline.enable_model_cpu_offload()
        # pipeline.enable_sequential_cpu_offload()
        generator = torch.Generator("cpu")  # use CPU generator if CPU offload is enabled
    else:
        pipeline = pipeline.to(device)
        generator = torch.Generator(device)
    logger.info(f"Model loaded to {pipeline.device}")

    for prompt_idx, config in enumerate(tqdm(data_config)):
        output_file = (args.output_dir / prompt_to_file_name(config["caption"], prefix=prompt_idx)).with_suffix(".mp4")
        if output_file.exists():
            logger.info(f"Skip {output_file}")
            continue

        if args.seed is not None:
            generator.manual_seed(args.seed)

        # inference
        with torch.no_grad():
            videos: List[List[Image.Image]]
            videos = pipeline(
                num_frames=args.video_size[0],
                height=args.video_size[1],
                width=args.video_size[2],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale,
                generator=generator,
                prompt=config["caption"],
                output_type="pil",
                return_dict=False,
                self_attention_kwargs=self_attention_kwargs,
            )[0]

        # save video
        for video_idx, video in enumerate(videos):
            export_to_video(video, output_file, fps=args.output_fps)
            logger.info(f"Video {prompt_idx}-{video_idx} saved: {output_file}")

    logger.info("Everything done!")


if __name__ == "__main__":
    main()
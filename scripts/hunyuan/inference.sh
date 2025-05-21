#!/bin/bash

pretrained_model_path=/raid/tao/models/HunyuanVideo  # local path to the pretrained model
# pretrained_model_path=hunyuanvideo-community/HunyuanVideo # path to the pretrained model in the Wan-AI repo
val_data_json_file=prompt.json  # path to the validation data JSON file


# CUDA_VISIBLE_DEVICES=7 python scripts/hunyuan/inference.py \
#     --pretrained_model_path $pretrained_model_path \
#     --val_data_json_file $val_data_json_file \
#     --output_dir results/hunyuan/baseline \
#     --native_attention \
#     --enable_cpu_offload \
#     --seed 1234


CUDA_VISIBLE_DEVICES=7 python scripts/hunyuan/inference.py \
    --pretrained_model_path $pretrained_model_path \
    --val_data_json_file $val_data_json_file \
    --output_dir results/hunyuan/vorta \
    --resume_dir results/hunyuan/train \
    --resume ckpt/step-000100 \
    --enable_cpu_offload \
    --seed 1234
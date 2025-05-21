#!/bin/bash

# model_variant=1.3B
model_variant=14B
pretrained_model_path=/raid/tao/models/Wan2.1-T2V-$model_variant-Diffusers  # local path to the pretrained model
# pretrained_model_path=Wan-AI/Wan2.1-T2V-$model_variant-Diffusers # path to the pretrained model in the Wan-AI repo
val_data_json_file=prompt.json  # path to the validation data JSON file


# CUDA_VISIBLE_DEVICES=6 python scripts/wan/inference.py \
#     --pretrained_model_path $pretrained_model_path \
#     --val_data_json_file $val_data_json_file \
#     --output_dir results/wan-$model_variant/baseline \
#     --native_attention \
#     --enable_cpu_offload \
#     --seed 1234


CUDA_VISIBLE_DEVICES=6 python scripts/wan/inference.py \
    --pretrained_model_path $pretrained_model_path \
    --val_data_json_file $val_data_json_file \
    --output_dir results/wan-$model_variant/vorta \
    --resume_dir results/wan-$model_variant/train \
    --resume ckpt/step-000100 \
    --enable_cpu_offload \
    --seed 1234
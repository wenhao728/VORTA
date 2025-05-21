#!/bin/bash

pretrained_model_path=/raid/tao/models/Wan2.1-T2V-1.3B-Diffusers  # local path to the pretrained model
# pretrained_model_path=Wan-AI/Wan2.1-T2V-1.3B-Diffusers # path to the pretrained model in the Wan-AI repo
val_data_json_file=prompt.json  # path to the validation data JSON file


CUDA_VISIBLE_DEVICES=7 python scripts/wan/inference.py \
    --pretrained_model_path $pretrained_model_path \
    --val_data_json_file $val_data_json_file \
    --output_dir results/wan-1.3B/baseline \
    --native_attention


CUDA_VISIBLE_DEVICES=7 python scripts/wan/inference.py \
    --pretrained_model_path $pretrained_model_path \
    --val_data_json_file $val_data_json_file \
    --output_dir results/wan-1.3B/vorta \
    --resume_dir results/wan-1.3B/train \
    --resume ckpt/step-000100
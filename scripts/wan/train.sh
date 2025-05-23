#!/bin/bash

END_POINT=localhost:29400

CUDA_VISIBLE_DEVICES=6,7 torchrun --nnodes 1 --nproc_per_node 2 --rdzv_endpoint=$END_POINT \
    scripts/wan/train.py \
    --exp_dir results/wan-14B/train \
    --data_json_file /dataset/video_datasets/HD-Mixkit-resized-2k/embed-Wan-720p-81f/video2caption.json \
    --dataloader_num_workers 24 \
    --sp_size 2 \
    --train_sp_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --video_size 77 720 1280 \
    --pretrained_model_path /raid/tao/models/Wan2.1-T2V-14B-Diffusers \
    --lowres_window_size 2 3 2 \
    --lowres_reduction_rate 0.5 \
    --sliding_attn_window_size 3 3 3 \
    --sliding_attn_tile_size 5 9 8 \
    --gradient_checkpointing \
    --learning_rate 1e-2 \
    --max_train_steps 100 \
    --diffusion_loss_weight 1.0 \
    --reg_loss_weight 0.02 \
    --last_layer_distill_loss_weight 20.0 \
    --report_interval 1 \
    --ckpt_interval 10
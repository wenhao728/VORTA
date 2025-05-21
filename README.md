<h1 align="center">
  VORTA: Efficient Video Diffusion via Routing Sparse Attention
</h1>

> [!TIP]
> **TL;DR** VORTA accelerates video diffusion transformers by sparse attention and dynamic routing, achieving up to 14.4× speedup with negligible quality loss.


<!-- ## 🎨 (WIP) Gallery -->


## 🔧 Setup
Install Pytorch, we have tested the code with PyTorch 2.6.0 and CUDA 12.6. But it should work with other versions as well. You can install PyTorch using the following command:
```
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

Install the dependencies:
```
python -m pip install -r requirements.txt
```


## 🚀 Quick Start
We use the genaral scripts to demonstrate the usage of our method. You can find the detailed scripts for each model in the `scripts` folder:
- HunyuanVideo: [scripts/hunyuan/inference.sh](scripts/hunyuan/inference.sh)
- Wan 2.1: [scripts/wan/inference.sh](scripts/wan/inference.sh)


Run the baseline model sampling without acceleration:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/<model_name>/inference.py \
    --pretrained_model_path <model_name_on_hf> \
    --pretrained_model_path $pretrained_model_path \
    --val_data_json_file prompt.json \
    --output_dir results/<model_name>/baseline \
    --native_attention \
    --seed 1234
```
> - You can edit the `prompts.json` or the `--val_data_json_file` option to change the text prompt.
> - See the source code `scripts/<model_name>/inference.py` or use `python scripts/<model_name>/inference.py --help` command for more detailed explanations of the arguments.


Run the video DiTs with VORTA for acceleration:
```diff
CUDA_VISIBLE_DEVICES=0 python scripts/<model_name>/inference.py \
    --pretrained_model_path <model_name_on_hf> \
    --pretrained_model_path $pretrained_model_path \
    --val_data_json_file prompt.json \
-    --output_dir results/<model_name>/baseline \
+    --output_dir results/<model_name>/vorta \
-    --native_attention \
+    --resume_dir results/<model_name>/train \
+    --resume ckpt/step-000100 \
    --seed 1234
```


## 🚧 TODO
- [ ] Attention kernel optimization for further hardware acceleration.
- [ ] Release the processed dataset.

## :hearts: Shout-out
Thanks to the authors of the following repositories for their great works and open-sourcing the code and models: [Diffusers](https://github.com/huggingface/diffusers), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [Wan 2.1](https://github.com/Wan-Video/Wan2.1), [FastVideo](https://github.com/hao-ai-lab/FastVideo)
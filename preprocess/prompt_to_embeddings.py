import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

project_dir = str(Path(__file__).resolve().parents[1])
if project_dir not in sys.path:
    sys.path.append(project_dir)

from vorta.utils import arg_to_json, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--annotation_file', type=Path, default=Path('/raid/data/datasets/HD-Mixkit-annotation/v1.1.0_HQ_part1.json'))
    parser.add_argument(
        '--output_dir', type=Path, default=Path('/raid/data/datasets/HD-Mixkit-resized-2k/Finetune-Wan_2_1'))
    parser.add_argument('--pretrained_model_name_or_path', type=Path, default='Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
    parser.add_argument('--max_sequence_length', type=int, default=226)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(
        self, 
        annotation_file, 
        tokenizer,
        max_sequence_length,
    ):
        self.annotation_file = annotation_file
        self.tokenizer = tokenizer
        # self.annotation = self.load_annotation()
        self.annotation = self.load_val_annotation()

        self.max_sequence_length = max_sequence_length

    def load_annotation(self):
        with open(self.annotation_file, 'r') as f:
            annotation_ = json.load(f)

        annotation = []
        for item in annotation_:
            file_name = item['path'].split('/')[-1].split('_')[0]
            captions = item['cap']
            for i, caption in enumerate(captions):
                annotation.append({'prompt_embed_path': f'{file_name}_cap_{i}.pt', 'caption': caption})
        return annotation

    def load_val_annotation(self):
        with open(self.annotation_file, 'r') as f:
            annotation_ = json.load(f)

        annotation = []
        for item in annotation_:
            caption = item['caption']
            prompt_embed_path = item['prompt_embed_path']
            annotation.append({'prompt_embed_path': prompt_embed_path, 'caption': caption})
        return annotation

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        item = self.annotation[idx]
        text_inputs = self.tokenizer(
            item['caption'],
            padding='max_length',
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        seq_lens = text_inputs['attention_mask'].gt(0).sum().long()
        
        return {
            'text_inputs': text_inputs.input_ids,
            'mask': text_inputs.attention_mask,
            'seq_lens': seq_lens,
            'prompt_embed_path': item['prompt_embed_path'],
        }
    
    @staticmethod
    def collate_fn(batch):
        text_inputs = torch.cat([item['text_inputs'] for item in batch], dim=0)
        mask = torch.cat([item['mask'] for item in batch], dim=0)
        seq_lens = torch.stack([item['seq_lens'] for item in batch], dim=0)
        prompt_embed_paths = [item['prompt_embed_path'] for item in batch]
        return text_inputs, mask, seq_lens, prompt_embed_paths


@torch.no_grad()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_file=args.output_dir / 'prompt_to_embeddings.log')
    args.output_dir = args.output_dir / 'prompt_embed'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(arg_to_json(args))

    # load dataset
    logger.info(f'Loading dataset from {args.annotation_file}')
    tokenizer = AutoTokenizer.from_pretrained(str(args.pretrained_model_name_or_path / 'tokenizer'))
    dataset = TextDataset(
        annotation_file=args.annotation_file,
        tokenizer=tokenizer,
        max_sequence_length=args.max_sequence_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=TextDataset.collate_fn,
        shuffle=False,
    )
    logger.info(f'Loaded {len(dataset)} samples')

    # load model
    logger.info(f'Loading model {args.pretrained_model_name_or_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dtype = torch.float32
    dtype = torch.bfloat16
    text_encoder = UMT5EncoderModel.from_pretrained(str(args.pretrained_model_name_or_path / 'text_encoder'))    
    # breakpoint()
    text_encoder.to(device=device, dtype=dtype)
    # breakpoint()
    text_encoder.eval()
    logger.info(f'Model loaded to {device} with dtype {dtype}')

    logger.info('Start extracting prompt embeddings')
    for batch in tqdm(dataloader):
        text_inputs = batch[0]
        mask = batch[1]
        seq_lens = batch[2]
        prompt_embed_paths = batch[3]
        prompts_exist = [(args.output_dir / prompt_embed_path).exists() for prompt_embed_path in prompt_embed_paths]
        # if all(prompts_exist):
        #     continue
        # breakpoint()
        text_embeds = text_encoder(text_inputs.to(device=device), mask.to(device=device)).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(args.max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )

        # breakpoint()
        for text_embed, prompt_embed_path in zip(text_embeds, prompt_embed_paths):
            torch.save(text_embed, args.output_dir / prompt_embed_path)

    logger.info('Finished extracting prompt embeddings')


if __name__ == '__main__':
    main()
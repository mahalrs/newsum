# Copyright 2023 The NewSum Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import lightning.pytorch as pl
import evaluate

from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoTokenizer
from tqdm import tqdm

from data import CNNDailyMailDataset
from newsum import NewSum

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default='./data/cnn_dailymail',
                    help='Directory containing processed dataset')
parser.add_argument('--log_dir',
                    default='./logs',
                    help='Directory to save logs and checkpoints')
parser.add_argument('--model_name',
                    default='google/pegasus-large',
                    help='Model to use for tokenizer and testing')
parser.add_argument(
    '--ckpt_path',
    default=None,
    help='Checkpoint to load for testing. If not None, Model name is ignored.')
parser.add_argument('--exp_name',
                    default='pegasus_prune',
                    help='Experiment name to use to save logs')
parser.add_argument('--prune_rate',
                    default=0.2,
                    type=float,
                    help='Amount to prune')
parser.add_argument('--test_batch', default=4, type=int, help='Test batch size')
parser.add_argument('--num_workers',
                    default=2,
                    type=int,
                    help='Number of dataloader workers to use')
parser.add_argument('--accelerator',
                    default='auto',
                    help='Accelerator to use for training')
parser.add_argument('--strategy',
                    default='auto',
                    help='Strategy to use for training')
parser.add_argument('--log_every_n_steps',
                    default=1,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--seed', default=123, type=int, help='Random seed to use')


def prune_model(model, prune_rate=0.2):
    model_to_prune = copy.deepcopy(model)

    # Apply structured pruning to the model's linear layers
    for _, module in model_to_prune.named_modules():
        if isinstance(module, nn.Linear):
            prune.ln_structured(module,
                                name='weight',
                                amount=prune_rate,
                                n=2,
                                dim=0)

    return model_to_prune


def eval(model, tokenizer, test_loader, device):
    # Define metrics
    metrics = [evaluate.load('rouge'), evaluate.load('bleu')]

    for batch in tqdm(test_loader):
        input_ids, attention_mask, targets = batch

        # Move to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate summaries
        summaries = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   num_beams=3,
                                   max_length=128,
                                   early_stopping=True,
                                   length_penalty=0.6)

        decoded_summaries = tokenizer.batch_decode(
            summaries,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            return_tensors='pt')

        decoded_targets = tokenizer.batch_decode(
            targets,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            return_tensors='pt')

        # Compute metrics
        for metric in metrics:
            metric.add_batch(predictions=decoded_summaries,
                             references=decoded_targets)

    # Compute scores
    scores = [metric.compute() for metric in metrics]
    return scores


def main():
    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    seed_everything(args.seed, workers=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    assert os.path.exists(args.data_dir), f'{args.data_dir} does not exist.'
    assert os.path.isdir(args.data_dir), f'{args.data_dir} is not a directory.'

    test_dataset = CNNDailyMailDataset(args.data_dir,
                                       split='test',
                                       tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    # Load config
    # Only model name is needed for testing; the rest of the config is ignored
    config = {
        'model_name': args.model_name,
        'learning_rate': 1e-5,
        'weight_decay': 0,
        'training_steps': 0,
        'warmup_steps': 0,
    }

    # Load model
    # If checkpoint path is provided, load that instead
    if args.ckpt_path is not None:
        assert os.path.exists(
            args.ckpt_path), f'{args.ckpt_path} does not exist.'
        assert os.path.isfile(
            args.ckpt_path), f'{args.ckpt_path} is not a file.'

        model = NewSum.load_from_checkpoint(args.ckpt_path)
    else:
        model = NewSum(config)

    # Prune model
    model = prune_model(model, prune_rate=args.prune_rate)

    # Define logger
    logger = TensorBoardLogger(args.log_dir, name=args.exp_name)

    # Define trainer
    trainer = pl.Trainer(accelerator=args.accelerator,
                         strategy=args.strategy,
                         devices=1,
                         num_nodes=1,
                         log_every_n_steps=args.log_every_n_steps,
                         logger=logger)

    # Test model
    trainer.test(model, dataloaders=test_loader)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model to device
    model = model.to(device)

    # Evaluate model
    scores = eval(model, tokenizer, test_loader, device)
    print(f'\nEvaluation results for {args.exp_name}:')
    print('  ROUGE:', scores[0])
    print('  BLEU:', scores[1]['bleu'])


if __name__ == '__main__':
    main()

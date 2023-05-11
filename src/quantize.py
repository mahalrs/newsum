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
import lightning.pytorch as pl

from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoTokenizer

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
                    default='pegasus_quant',
                    help='Experiment name to use to save logs')
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


def quantize(model):
    model_to_quantize = copy.deepcopy(model)
    quantized_model = torch.quantization.quantize_dynamic(
        model_to_quantize, {nn.Linear, nn.LayerNorm, nn.ReLU},
        dtype=torch.qint8)

    return quantized_model


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

    # Quantize model
    model = quantize(model)

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

    # Save Model
    trainer.save_checkpoint(os.path.join(args.log_dir, f'{args.exp_name}.ckpt'))


if __name__ == '__main__':
    main()

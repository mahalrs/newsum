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
import os

import lightning.pytorch as pl
import wandb

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
parser.add_argument('--run_name', default='pegasus', help='Sweep name to use')
parser.add_argument('--wandb_proj',
                    default='newsum',
                    help='Wandb project to use')
parser.add_argument('--model_name',
                    default='google/pegasus-large',
                    help='Model name to use for hyperparameter search')
parser.add_argument('--data_ratio',
                    default=0.1,
                    type=float,
                    help='Ratio of data to use for hyperparameter search')
parser.add_argument('--train_batch',
                    default=2,
                    type=int,
                    help='Train batch size')
parser.add_argument('--val_batch',
                    default=2,
                    type=int,
                    help='Validation batch size')
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
parser.add_argument('--devices',
                    default=1,
                    type=int,
                    help='Number of devices (cpu/gpu/tpu) to use for training.')
parser.add_argument('--epochs',
                    default=1,
                    type=int,
                    help='Number of epochs to train')
parser.add_argument('--run_cap',
                    default=20,
                    type=int,
                    help='Number of sweep runs')
parser.add_argument('--log_every_n_steps',
                    default=100,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--seed', default=123, type=int, help='Random seed to use')


def get_data_loaders(args, tokenizer):
    assert os.path.exists(args.data_dir), f'{args.data_dir} does not exist.'
    assert os.path.isdir(args.data_dir), f'{args.data_dir} is not a directory.'

    train_dataset = CNNDailyMailDataset(args.data_dir,
                                        split='train',
                                        tokenizer=tokenizer,
                                        ratio=args.data_ratio)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch,
                              shuffle=(not args.devices > 1),
                              num_workers=args.num_workers,
                              pin_memory=True)

    val_dataset = CNNDailyMailDataset(args.data_dir,
                                      split='validation',
                                      tokenizer=tokenizer,
                                      ratio=args.data_ratio)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.val_batch,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    return train_loader, val_loader


def get_sweep_config(run_name):
    sweep_config = {
        'method': 'random',
        'name': run_name,
        'run_cap': 20,
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'values': [1e-5, 5e-5, 1e-4]
            },
            'weight_decay': {
                'values': [0.0, 0.01, 0.1]
            },
            'warmup_steps_ratio': {
                'values': [0.0, 0.05, 0.1, 0.2]
            }
        }
    }

    return sweep_config


def main():
    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    seed_everything(args.seed, workers=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load data loaders
    train_loader, val_loader = get_data_loaders(args, tokenizer)

    # Get sweep config
    sweep_config = get_sweep_config(args.run_name)

    # Set up sweep
    def run_sweep(config=None, args=args):
        # Initialize a new wandb run
        with wandb.init(config=config):
            config = wandb.config

            # Calculate warmup and total steps
            assert args.devices > 0, 'Number of devices must be greater than 0.'
            total_steps = len(train_loader) // args.devices * args.epochs
            warmup_steps = int(total_steps * config.warmup_steps_ratio)

            # Initialize model config
            model_config = {
                'model_name': args.model_name,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'warmup_steps': warmup_steps,
                'training_steps': total_steps,
                'wandb': True
            }

            # Initialize model
            model = NewSum(model_config)

            # Initialize logger
            logger = TensorBoardLogger(args.log_dir, name=args.run_name)

            # Define trainer
            trainer = pl.Trainer(max_epochs=args.epochs,
                                 accelerator=args.accelerator,
                                 strategy=args.strategy,
                                 devices=args.devices,
                                 log_every_n_steps=args.log_every_n_steps,
                                 logger=logger,
                                 enable_checkpointing=False)

            # Train model
            trainer.fit(model,
                        train_dataloaders=train_loader,
                        val_dataloaders=val_loader)

    # Initialize sweep id
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_proj)

    # Run sweep
    wandb.agent(sweep_id, run_sweep, count=args.run_cap)


if __name__ == '__main__':
    main()

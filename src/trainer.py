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
import json
import os

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
parser.add_argument('--exp_name',
                    default='pegasus',
                    help='Experiment name to use to save logs')
parser.add_argument('--config',
                    default='./config/pegasus.json',
                    help='Path to config json file')
parser.add_argument('--train_batch',
                    default=4,
                    type=int,
                    help='Train batch size')
parser.add_argument('--val_batch',
                    default=4,
                    type=int,
                    help='Validation batch size')
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
parser.add_argument('--devices',
                    default=1,
                    type=int,
                    help='Number of devices (cpu/gpu/tpu) to use for training.')
parser.add_argument('--epochs',
                    default=1,
                    type=int,
                    help='Number of epochs to train')
parser.add_argument('--log_every_n_steps',
                    default=100,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--seed', default=123, type=int, help='Random seed to use')


def main():
    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    seed_everything(args.seed, workers=True)

    # Load config file
    assert os.path.exists(args.config), f'{args.config} does not exist.'
    assert os.path.isfile(args.config), f'{args.config} is not a file.'

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Load dataset
    assert os.path.exists(args.data_dir), f'{args.data_dir} does not exist.'
    assert os.path.isdir(args.data_dir), f'{args.data_dir} is not a directory.'

    train_dataset = CNNDailyMailDataset(args.data_dir,
                                        split='train',
                                        tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch,
                              shuffle=(not args.devices > 1),
                              num_workers=args.num_workers,
                              pin_memory=True)

    val_dataset = CNNDailyMailDataset(args.data_dir,
                                      split='validation',
                                      tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.val_batch,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    test_dataset = CNNDailyMailDataset(args.data_dir,
                                       split='test',
                                       tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.train_batch,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    # Calculate warmup and total steps
    assert args.devices > 0, 'Number of devices must be greater than 0.'
    total_steps = len(train_loader) // args.devices * args.epochs
    warmup_steps = int(total_steps * config['warmup_steps_ratio'])

    config['training_steps'] = total_steps
    config['warmup_steps'] = warmup_steps

    # Load model
    model = NewSum(config=config)

    # Define logger
    logger = TensorBoardLogger(args.log_dir, name=args.exp_name)

    # Define trainer
    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator=args.accelerator,
                         strategy=args.strategy,
                         devices=args.devices,
                         log_every_n_steps=args.log_every_n_steps,
                         logger=logger)

    # Train model
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # Test model
    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    main()

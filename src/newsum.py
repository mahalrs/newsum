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

import torch
import lightning.pytorch as pl
import wandb

from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup


class NewSum(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'])

        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.warmup_steps = config['warmup_steps']
        self.training_steps = config['training_steps']

        if config['wandb']:
            self.wandb_logger = True
        else:
            self.wandb_logger = False

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    def _shared_step(self, batch, phase):
        input_ids, attention_mask, targets = batch
        outputs = self(input_ids, attention_mask, targets)

        loss = outputs.loss

        if self.global_rank == 0 and self.wandb_logger:
            wandb.log({f'{phase}_loss': loss})

        self.log(f'{phase}_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)

        return {f'{phase}_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.training_steps)

        return [optimizer], [lr_scheduler]

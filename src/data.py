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

import json
import os

from torch.utils.data import Dataset


class CNNDailyMailDataset(Dataset):

    def __init__(self, data_dir, split, tokenizer, ratio=None, extended=False):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.extended = extended

        self.data = self.load_data()

        if ratio is not None:
            self.data = self.data[:int(len(self.data) * ratio)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.split,
                                 f'{self.data[idx]}.json')

        with open(file_path, 'r') as f:
            sample = json.load(f)
            article = sample['article_ner'] if self.extended else sample[
                'article']

            inputs = self.tokenizer(article,
                                    max_length=256,
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors='pt')
            targets = self.tokenizer(sample['summary'],
                                     max_length=128,
                                     truncation=True,
                                     padding='max_length',
                                     return_tensors='pt')

            return inputs['input_ids'].view(-1), inputs['attention_mask'].view(
                -1), targets['input_ids'].view(-1)

    def load_data(self):
        with open(os.path.join(self.data_dir, f'{self.split}.json'), 'r') as f:
            return json.load(f)

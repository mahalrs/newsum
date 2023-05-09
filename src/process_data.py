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

import torch

from datasets import load_dataset
from lightning.pytorch import seed_everything
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--datafiles',
                    default='../datafiles',
                    help='Directory containing data files')
parser.add_argument('--data_root',
                    default='./data',
                    help='Directory to save processed dataset')
parser.add_argument('--seed', default=123, type=int, help='Random seed to use')


def perform_ner(ner_pipeline, text):
    out = ner_pipeline(text)

    special_tokens = set()
    new_text = ''
    last_idx = 0

    for entity in out:
        new_text += text[last_idx:entity['start']]
        new_text += f"<{entity['entity']}>" + text[
            entity['start']:entity['end']] + f"</{entity['entity']}>"
        last_idx = entity['end']

        special_tokens.add(f"<{entity['entity']}>")
        special_tokens.add(f"</{entity['entity']}>")
    new_text += text[last_idx:]

    return new_text, special_tokens


def process_cnn_dailymail(datafiles, data_root, ner_pipe):
    out_dir = os.path.join(data_root, 'cnn_dailymail')
    for split in ['train', 'validation', 'test']:
        if not os.path.exists(os.path.join(out_dir, split)):
            os.makedirs(os.path.join(out_dir, split))

    dataset = load_dataset('cnn_dailymail', '3.0.0')
    special_tokens = set()

    for split in ['train', 'validation', 'test']:
        # Create a file containing all the ids for the split
        with open(os.path.join(out_dir, f'{split}.json'), 'w') as f:
            json.dump(dataset[split]['id'], f)

        # Create a file for each sample in the split
        for i in range(dataset[split].num_rows):
            sample = dataset[split][i]
            idx = sample['id']

            # Perform NER on the article
            article_ner, special_tokens_new = perform_ner(
                ner_pipe, sample['article'])
            special_tokens.update(special_tokens_new)

            with open(os.path.join(out_dir, split, f'{idx}.json'), 'w') as f:
                json.dump(
                    {
                        'id': idx,
                        'article': sample['article'],
                        'article_ner': article_ner,
                        'summary': sample['highlights']
                    }, f)

    # Save special tokens
    with open(os.path.join(out_dir, 'special_tokens.json'), 'w') as f:
        json.dump(list(special_tokens), f)


def main():
    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    seed_everything(args.seed, workers=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    print(device)

    assert os.path.exists(args.datafiles), f'{args.datafiles} does not exist.'
    assert os.path.isdir(
        args.datafiles), f'{args.datafiles} is not a directory.'

    # Create output directory
    if not os.path.exists(args.data_root):
        os.makedirs(args.data_root)

    # Load NER pipeline
    tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
    model = AutoModelForTokenClassification.from_pretrained(
        'dslim/bert-base-NER').to(device)
    ner_pipe = pipeline('ner',
                        model=model,
                        tokenizer=tokenizer,
                        framework='pt',
                        device=device)

    # Process CNN/DailyMail dataset
    process_cnn_dailymail(args.datafiles, args.data_root, ner_pipe)

    # Process XSum dataset
    # process_xsum(args.datafiles, args.data_root)


if __name__ == '__main__':
    main()

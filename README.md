# NewSum - News Summarization using Enhanced Content Features

This repository contains source code for various experiments to explore the feasibility of using enhanced content features such as news category and named entities to improve the quality and coherence of generated news summaries.


## Install dependencies

```sh
pip install -r requirements.txt
```


## Preprocess Data

Run the following commands to preprocess the data.

```sh
cd src

python process_data.py
```


## Fine-tune

To start fine-tuning, run the following command.

```sh
cd src

# Fine-tune pegasus for 1 epoch using 1 gpu
python trainer.py --exp_name my_exp1 --config ./config/pegasus.json --accelerator gpu --devices 1 --epochs 1
```

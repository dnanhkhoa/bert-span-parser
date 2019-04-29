# -*- coding: utf-8 -*-
import os
import random

import click
import torch

# from apex import amp
from loguru import logger
from tqdm import tqdm, trange

from eval import evalb
from label_encoder import LabelEncoder
from trees import load_trees

TAG_UNK = "UNK"

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
}


def eval():
    pass


@click.command()
@click.option("--train_file", required=True, type=click.Path())
@click.option("--dev_file", required=True, type=click.Path())
@click.option("--test_file", required=True, type=click.Path())
@click.option("--bert_model", required=True, type=click.Path())
@click.option("--batch_size", default=16, show_default=True, type=click.INT)
@click.option("--num_epochs", default=10, show_default=True, type=click.INT)
@click.option("--max_seq_length", default=256, show_default=True, type=click.INT)
@click.option("--learning_rate", default=3e-5, show_default=True, type=click.FLOAT)
@click.option("--warmup_proportion", default=0.1, show_default=True, type=click.FLOAT)
@click.option(
    "--gradient_accumulation_steps", default=1, show_default=True, type=click.INT
)
@click.option("--seed", default=42, show_default=True, type=click.INT)
@click.option("--device", default=0, show_default=True, type=click.INT)
@click.option("--fp16", is_flag=True)
@click.option("--do_eval", is_flag=True)
def main(*_, **kwargs):
    use_cuda = torch.cuda.is_available()
    device = torch.cuda.device("cuda:" + str(kwargs["device"]) if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.set_device(device)

    # For reproducibility
    os.environ["PYTHONHASHSEED"] = str(kwargs["seed"])
    random.seed(kwargs["seed"])
    torch.manual_seed(kwargs["seed"])
    torch.cuda.manual_seed_all(kwargs["seed"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Settings
    batch_size = kwargs["batch_size"] // kwargs["gradient_accumulation_steps"]


if __name__ == "__main__":
    main(
        [
            "--train_file=corpora/WSJ-PTB/02-21.10way.clean.train",
            "--dev_file=corpora/WSJ-PTB/22.auto.clean.dev",
            "--test_file=corpora/WSJ-PTB/23.auto.clean.test",
            "--bert_model=models/bert-base-multilingual-cased",
            # "--fp16",
            # "--do_eval",
        ]
    )

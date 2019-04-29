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
def main(*args, **kwargs):
    pass


if __name__ == "__main__":
    main([])

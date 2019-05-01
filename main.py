# -*- coding: utf-8 -*-
import os
import random

import click
import neptune
import torch

# from apex import amp
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from bert.tokenization import BertTokenizer
from eval import evalb
from label_encoder import LabelEncoder
from model import ChartParser
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


def create_data_loader(sentences, batch_size, max_seq_length, tokenizer, is_eval):
    features = []
    for sentence in sentences:
        tokens = []
        sections = []
        for token in sentence:
            subtokens = tokenizer.tokenize(BERT_TOKEN_MAPPING.get(token, token))
            tokens.extend(subtokens)
            sections.append(len(subtokens))

        num_tokens = len(tokens)

        # Account for [CLS] and [SEP] tokens
        if num_tokens > max_seq_length - 2:
            num_tokens = max_seq_length - 2
            tokens = tokens[:num_tokens]
            logger.warning("Too long sentence: {}", sentence)

        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        attention_mask = [1] * len(ids)

        # Zero-pad up to the maximum sequence length
        padding_size = max_seq_length - len(ids)

        ids += [0] * padding_size
        attention_mask += [0] * padding_size

        assert len(ids) == max_seq_length
        assert len(attention_mask) == max_seq_length

        features.append(
            {"ids": ids, "attention_mask": attention_mask, "sections": sections}
        )

    all_indices = torch.arange(len(features), dtype=torch.long)
    all_ids = torch.tensor([feature["ids"] for feature in features], dtype=torch.long)
    all_attention_masks = torch.tensor(
        [feature["attention_mask"] for feature in features], dtype=torch.long
    )
    all_sections = [feature["sections"] for feature in features]

    dataset = TensorDataset(all_indices, all_ids, all_attention_masks)

    sampler = SequentialSampler(dataset) if is_eval else RandomSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader, all_sections


def prepare_batch_input(
    indices, ids, attention_masks, sections, trees, sentences, device
):
    _ids = ids.to(device)
    _attention_masks = attention_masks.to(device)

    _sections = []
    _trees = []
    _sentences = []

    for _id in indices:
        _sections.append(sections[_id])
        _trees.append(trees[_id])
        _sentences.append(sentences[_id])

    return _ids, _attention_masks, _sections, _trees, _sentences


def eval():
    pass


@click.command()
@click.option("--train_file", required=True, type=click.Path())
@click.option("--dev_file", required=True, type=click.Path())
@click.option("--test_file", required=True, type=click.Path())
@click.option("--output_dir", required=True, type=click.Path())
@click.option("--bert_model", required=True, type=click.Path())
@click.option("--tag-embedding-dim", default=50, show_default=True, type=click.INT)
@click.option("--label-hidden-dim", default=250, show_default=True, type=click.INT)
@click.option("--dropout_prob", default=0.4, show_default=True, type=click.FLOAT)
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
            "--output_dir=outputs",
            "--bert_model=models/bert-base-multilingual-cased",
            # "--fp16",
            # "--do_eval",
        ]
    )

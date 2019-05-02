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

from bert.optimization import BertAdam
from bert.tokenization import BertTokenizer
from eval import evalb
from label_encoder import LabelEncoder
from model import ChartParser
from trees import InternalParseNode, load_trees

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

LABEL_MAPPING = {
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


def create_dataloader(sentences, batch_size, max_seq_length, tokenizer, is_eval):
    features = []
    for sentence in sentences:
        tokens = []
        sections = []
        for _, token in sentence:
            subtokens = tokenizer.tokenize(token)
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
    device = torch.device("cuda:" + str(kwargs["device"]) if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.set_device(device)

    # For reproducibility
    os.environ["PYTHONHASHSEED"] = str(kwargs["seed"])
    random.seed(kwargs["seed"])
    torch.manual_seed(kwargs["seed"])
    torch.cuda.manual_seed_all(kwargs["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare and load data
    tokenizer = BertTokenizer.from_pretrained(kwargs["bert_model"], do_lower_case=False)

    logger.info("Loading data...")
    train_treebank = load_trees(
        kwargs["train_file"],
        label_mapping=LABEL_MAPPING,
        word_mapping=BERT_TOKEN_MAPPING,
    )
    dev_treebank = load_trees(
        kwargs["dev_file"], label_mapping=LABEL_MAPPING, word_mapping=BERT_TOKEN_MAPPING
    )
    test_treebank = load_trees(
        kwargs["test_file"],
        label_mapping=LABEL_MAPPING,
        word_mapping=BERT_TOKEN_MAPPING,
    )
    logger.info(
        "Loaded {:,} train, {:,} dev, and {:,} test examples!",
        len(train_treebank),
        len(dev_treebank),
        len(test_treebank),
    )

    logger.info("Preprocessing data...")
    train_parse = [tree.convert() for tree in train_treebank]
    train_sentences = [
        [(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in train_parse
    ]
    dev_sentences = [
        [(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in dev_treebank
    ]
    test_sentences = [
        [(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in test_treebank
    ]
    logger.info("Data preprocessed!")

    logger.info("Preparing data for training...")

    tags = []
    labels = [()]

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, InternalParseNode):
                labels.append(node.label)
                nodes.extend(reversed(node.children))
            else:
                tags.append(node.tag)

    tag_encoder = LabelEncoder(unk_label=TAG_UNK)
    tag_encoder.fit(tags)

    label_encoder = LabelEncoder(unk_label=TAG_UNK)
    label_encoder.fit(labels)

    logger.info("Data prepared!")

    # Settings
    num_train_optimization_steps = kwargs["num_epochs"] * (
        (len(train_parse) - 1) // kwargs["batch_size"] + 1
    )
    kwargs["batch_size"] //= kwargs["gradient_accumulation_steps"]

    logger.info("Creating dataloaders for training...")
    train_dataloader, train_sections = create_dataloader(
        sentences=train_sentences,
        batch_size=kwargs["batch_size"],
        max_seq_length=kwargs["max_seq_length"],
        tokenizer=tokenizer,
        is_eval=False,
    )
    dev_dataloader, dev_sections = create_dataloader(
        sentences=dev_sentences,
        batch_size=kwargs["batch_size"],
        max_seq_length=kwargs["max_seq_length"],
        tokenizer=tokenizer,
        is_eval=True,
    )
    test_dataloader, test_sections = create_dataloader(
        sentences=test_sentences,
        batch_size=kwargs["batch_size"],
        max_seq_length=kwargs["max_seq_length"],
        tokenizer=tokenizer,
        is_eval=True,
    )
    logger.info("Dataloaders created!")

    # Initialize model
    model = ChartParser.from_pretrained(
        kwargs["bert_model"],
        tag_encoder=tag_encoder,
        label_encoder=label_encoder,
        tag_embedding_dim=kwargs["tag_embedding_dim"],
        label_hidden_dim=kwargs["label_hidden_dim"],
        dropout_prob=kwargs["dropout_prob"],
    )

    model.to(device)

    # Prepare optimizer
    param_optimizers = list(model.named_parameters())

    # Hack to remove pooler, which is not used thus it produce None grad that break apex
    param_optimizers = [n for n in param_optimizers if "pooler" not in n[0]]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizers if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizers if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=kwargs["learning_rate"],
        warmup=kwargs["warmup_proportion"],
        t_total=num_train_optimization_steps,
    )

    if kwargs["fp16"]:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if kwargs["do_eval"]:
        pass
    else:
        # Training phase
        global_steps = 0

        for epoch in trange(kwargs["num_epochs"], desc="Epoch"):
            model.train()


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

# -*- coding: utf-8 -*-
import json
import os
import random

import click
import neptune
import torch

# from apex import amp
from loguru import logger
from neptune.exceptions import NoExperimentContext
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from bert.optimization import BertAdam
from bert.tokenization import BertTokenizer
from eval import evalb
from label_encoder import LabelEncoder
from model import ChartParser
from trees import InternalParseNode, load_trees

MODEL_FILENAME = "model.bin"

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


def create_dataloader(sentences, batch_size, tag_encoder, tokenizer, is_eval):
    features = []
    for sentence in sentences:
        tokens = []
        tags = []
        sections = []
        for tag, token in sentence:
            subtokens = tokenizer.tokenize(BERT_TOKEN_MAPPING.get(token, token))
            tokens.extend(subtokens)
            tags.append(tag_encoder.transform(tag, unknown_label="[UNK]"))
            sections.append(len(subtokens))

        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        attention_mask = [1] * len(ids)

        features.append(
            {
                "ids": ids,
                "attention_mask": attention_mask,
                "tags": tags,
                "sections": sections,
            }
        )

    dataset = TensorDataset(torch.arange(len(features), dtype=torch.long))

    sampler = SequentialSampler(dataset) if is_eval else RandomSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader, features


def prepare_batch_input(indices, features, trees, sentences, tag_encoder, device):
    _ids = []
    _attention_masks = []
    _tags = []
    _sections = []

    _trees = []
    _sentences = []

    ids_padding_size = 0
    tags_padding_size = 0

    for _id in indices:
        _ids.append(features[_id]["ids"])
        _attention_masks.append(features[_id]["attention_mask"])
        _tags.append(features[_id]["tags"])
        _sections.append(features[_id]["sections"])

        _trees.append(trees[_id])
        _sentences.append(sentences[_id])

        ids_padding_size = max(ids_padding_size, len(features[_id]["ids"]))
        tags_padding_size = max(tags_padding_size, len(features[_id]["tags"]))

    # Zero-pad
    for _id, _attention_mask, _tag in zip(_ids, _attention_masks, _tags):
        padding_size = ids_padding_size - len(_id)
        _id += [0] * padding_size
        _attention_mask += [0] * padding_size
        _tag += [tag_encoder.transform("[PAD]")] * (tags_padding_size - len(_tag))

    _ids = torch.tensor(_ids, dtype=torch.long).to(device)
    _attention_masks = torch.tensor(_attention_masks, dtype=torch.long).to(device)
    _tags = torch.tensor(_tags, dtype=torch.long).to(device)

    return _ids, _attention_masks, _tags, _sections, _trees, _sentences


def eval(
    model,
    eval_dataloader,
    eval_features,
    eval_trees,
    eval_sentences,
    tag_encoder,
    device,
):
    # Evaluation phase
    model.eval()

    all_predicted_trees = []

    for indices, *_ in tqdm(eval_dataloader, desc="Iteration"):
        ids, attention_masks, tags, sections, _, sentences = prepare_batch_input(
            indices=indices,
            features=eval_features,
            trees=eval_trees,
            sentences=eval_sentences,
            tag_encoder=tag_encoder,
            device=device,
        )

        with torch.no_grad():
            predicted_trees, _ = model(
                ids=ids,
                attention_masks=attention_masks,
                tags=tags,
                sections=sections,
                sentences=sentences,
                gold_trees=None,
            )

            for predicted_tree in predicted_trees:
                all_predicted_trees.append(predicted_tree.convert())

    return evalb(eval_trees, all_predicted_trees)


@click.command()
@click.option("--train_file", required=True, type=click.Path())
@click.option("--dev_file", required=True, type=click.Path())
@click.option("--test_file", required=True, type=click.Path())
@click.option("--output_dir", required=True, type=click.Path())
@click.option("--bert_model", required=True, type=click.Path())
@click.option("--tag_embedding_dim", default=50, show_default=True, type=click.INT)
@click.option("--dropout_prob", default=0.1, show_default=True, type=click.FLOAT)
@click.option("--batch_size", default=32, show_default=True, type=click.INT)
@click.option("--num_epochs", default=20, show_default=True, type=click.INT)
@click.option("--learning_rate", default=3e-5, show_default=True, type=click.FLOAT)
@click.option("--warmup_proportion", default=0.1, show_default=True, type=click.FLOAT)
@click.option(
    "--gradient_accumulation_steps", default=1, show_default=True, type=click.INT
)
@click.option("--seed", default=42, show_default=True, type=click.INT)
@click.option("--device", default=0, show_default=True, type=click.INT)
@click.option("--fp16", is_flag=True)
@click.option("--freeze_bert", is_flag=True)
@click.option("--do_eval", is_flag=True)
def main(*_, **kwargs):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(kwargs["device"]) if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.set_device(device)

    kwargs["use_cuda"] = use_cuda

    neptune.create_experiment(
        name="bert-span-parser",
        upload_source_files=[],
        params={k: str(v) if isinstance(v, bool) else v for k, v in kwargs.items()},
    )

    logger.info("Settings: {}", json.dumps(kwargs, indent=2, ensure_ascii=False))

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

    train_treebank = load_trees(kwargs["train_file"])
    dev_treebank = load_trees(kwargs["dev_file"])
    test_treebank = load_trees(kwargs["test_file"])

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
    labels = []

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, InternalParseNode):
                labels.append(node.label)
                nodes.extend(reversed(node.children))
            else:
                tags.append(node.tag)

    tag_encoder = LabelEncoder()
    tag_encoder.fit(tags, reserved_labels=["[PAD]", "[UNK]"])

    label_encoder = LabelEncoder()
    label_encoder.fit(labels, reserved_labels=[()])

    logger.info("Data prepared!")

    # Settings
    num_train_optimization_steps = kwargs["num_epochs"] * (
        (len(train_parse) - 1) // kwargs["batch_size"] + 1
    )
    kwargs["batch_size"] //= kwargs["gradient_accumulation_steps"]

    logger.info("Creating dataloaders for training...")

    train_dataloader, train_features = create_dataloader(
        sentences=train_sentences,
        batch_size=kwargs["batch_size"],
        tag_encoder=tag_encoder,
        tokenizer=tokenizer,
        is_eval=False,
    )
    dev_dataloader, dev_features = create_dataloader(
        sentences=dev_sentences,
        batch_size=kwargs["batch_size"],
        tag_encoder=tag_encoder,
        tokenizer=tokenizer,
        is_eval=True,
    )
    test_dataloader, test_features = create_dataloader(
        sentences=test_sentences,
        batch_size=kwargs["batch_size"],
        tag_encoder=tag_encoder,
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
        dropout_prob=kwargs["dropout_prob"],
    )

    model.to(device)

    # Prepare optimizer
    param_optimizers = list(model.named_parameters())

    if kwargs["freeze_bert"]:
        for p in model.bert.parameters():
            p.requires_grad = False
        param_optimizers = [(n, p) for n, p in param_optimizers if p.requires_grad]

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
        pretrained_model_file = os.path.join(kwargs["output_dir"], MODEL_FILENAME)

        assert os.path.isfile(
            pretrained_model_file
        ), "Pretrained model file does not exist!"

        logger.info("Loading pretrained model from {}", pretrained_model_file)

        # Load model from file
        params = torch.load(pretrained_model_file, map_location=device)
        model.load_state_dict(params["model"])

        logger.info(
            "Loaded pretrained model (Epoch: {:,}, Fscore: {:.2f})",
            params["epoch"],
            params["fscore"],
        )

        eval_score = eval(
            model=model,
            eval_dataloader=test_dataloader,
            eval_features=test_features,
            eval_trees=test_treebank,
            eval_sentences=test_sentences,
            tag_encoder=tag_encoder,
            device=device,
        )

        neptune.send_metric("test_eval_precision", eval_score.precision())
        neptune.send_metric("test_eval_recall", eval_score.recall())
        neptune.send_metric("test_eval_fscore", eval_score.fscore())

        tqdm.write("Evaluation score: {}".format(str(eval_score)))
    else:
        pretrained_model_file = os.path.join(kwargs["output_dir"], MODEL_FILENAME)
        assert not os.path.isfile(
            pretrained_model_file
        ), "Please remove or move the pretrained model file to another place!"

        # Training phase
        global_steps = 0
        best_dev_fscore = 0

        for epoch in trange(kwargs["num_epochs"], desc="Epoch"):
            model.train()

            train_loss = 0
            num_train_steps = 0

            for step, (indices, *_) in enumerate(
                tqdm(train_dataloader, desc="Iteration")
            ):
                ids, attention_masks, tags, sections, trees, sentences = prepare_batch_input(
                    indices=indices,
                    features=train_features,
                    trees=train_parse,
                    sentences=train_sentences,
                    tag_encoder=tag_encoder,
                    device=device,
                )

                _, losses = model(
                    ids=ids,
                    attention_masks=attention_masks,
                    tags=tags,
                    sections=sections,
                    sentences=sentences,
                    gold_trees=trees,
                )

                loss = torch.stack(losses).mean()

                if kwargs["gradient_accumulation_steps"] > 1:
                    loss /= kwargs["gradient_accumulation_steps"]

                if kwargs["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                train_loss += loss.item()

                num_train_steps += 1

                if (step + 1) % kwargs["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_steps += 1

            # Write logs
            neptune.send_metric("train_loss", epoch, train_loss / num_train_steps)
            neptune.send_metric("global_steps", epoch, global_steps)

            tqdm.write(
                "Epoch: {:,} - Train loss: {:.4f} - Global steps: {:,}".format(
                    epoch, train_loss / num_train_steps, global_steps
                )
            )

            # Evaluate
            eval_score = eval(
                model=model,
                eval_dataloader=dev_dataloader,
                eval_features=dev_features,
                eval_trees=dev_treebank,
                eval_sentences=dev_sentences,
                tag_encoder=tag_encoder,
                device=device,
            )

            neptune.send_metric("eval_precision", epoch, eval_score.precision())
            neptune.send_metric("eval_recall", epoch, eval_score.recall())
            neptune.send_metric("eval_fscore", epoch, eval_score.fscore())

            tqdm.write(
                "Epoch: {:,} - Evaluation score: {}".format(epoch, str(eval_score))
            )

            # Save best model
            if eval_score.fscore() > best_dev_fscore:
                best_dev_fscore = eval_score.fscore()

                tqdm.write("** Saving model...")

                os.makedirs(kwargs["output_dir"], exist_ok=True)

                torch.save(
                    {
                        "epoch": epoch,
                        "fscore": best_dev_fscore,
                        "model": (
                            model.module if hasattr(model, "module") else model
                        ).state_dict(),
                    },
                    pretrained_model_file,
                )

            tqdm.write("** Best evaluation fscore: {:.2f}".format(best_dev_fscore))


if __name__ == "__main__":
    neptune.init(project_qualified_name=os.getenv("NEPTUNE_PROJECT_NAME"))
    try:
        # main(
        #     [
        #         "--train_file=corpora/WSJ-PTB/02-21.10way.clean.train",
        #         "--dev_file=corpora/WSJ-PTB/22.auto.clean.dev",
        #         "--test_file=corpora/WSJ-PTB/23.auto.clean.test",
        #         "--output_dir=outputs",
        #         "--bert_model=models/bert-base-multilingual-cased",
        #         "--batch_size=32",
        #         "--num_epochs=20",
        #         "--learning_rate=3e-5",
        #         # "--fp16",
        #         # "--do_eval",
        #     ]
        # )

        # main(
        #     [
        #         "--train_file=corpora/Small-WSJ-PTB/small",
        #         "--dev_file=corpora/Small-WSJ-PTB/small",
        #         "--test_file=corpora/Small-WSJ-PTB/small",
        #         "--output_dir=outputs",
        #         "--bert_model=models/bert-base-multilingual-cased",
        #         "--batch_size=32",
        #         "--num_epochs=20",
        #         "--learning_rate=3e-5",
        #         # "--fp16",
        #         # "--do_eval",
        #     ]
        # )

        main()
    finally:
        try:
            neptune.stop()
        except NoExperimentContext:
            pass

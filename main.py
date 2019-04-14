import argparse
import itertools
import os.path
import time
import re

import dynet_config
dynet_config.set_gpu()

import dynet as dy
import numpy as np

import evaluate
import parse
import trees
import vocabulary

# For BERT
import torch
from bert.tokenization import BertTokenizer
from bert.modeling import BertModel


def prepare_tensors(sentences, tokenizer, seq_length):
    """Loads a data file into a list of `InputFeatures`."""

    features = []
    for sentence in sentences:
        tokens = []
        input_subword_mask = []

        for phrase_idx, phrase in enumerate(sentence):
            for token in re.split(r"(?<=[^\W_])_(?=[^\W_])", phrase, flags=re.UNICODE):
                subwords = tokenizer.tokenize(token)
                for subword in subwords:
                    tokens.append(subword)
                    input_subword_mask.append(phrase_idx)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > seq_length - 2:
            tokens = tokens[0 : (seq_length - 2)]
            input_subword_mask = input_subword_mask[0 : (seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens.insert(0, "[CLS]")
        input_subword_mask.insert(0, -1)
        tokens.append("[SEP]")
        input_subword_mask.append(-1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_subword_mask.append(-1)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_subword_mask) == seq_length

        features.append({
            "input_ids": input_ids,
            "input_mask": input_mask,
            "input_subword_mask": input_subword_mask
        })

    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f["input_mask"] for f in features], dtype=torch.long)
    all_input_subword_mask = [f["input_subword_mask"] for f in features]

    return all_input_ids, all_input_mask, all_input_subword_mask


def get_embeddings(sentences, model, tokenizer, device, seq_length=256):
    all_input_ids, all_input_mask, all_input_subword_mask = prepare_tensors(sentences, tokenizer, seq_length)
    all_input_ids = all_input_ids.to(device)
    all_input_mask = all_input_mask.to(device)

    subword_embeddings, _ = model(
        all_input_ids,
        token_type_ids=None,
        attention_mask=all_input_mask,
        output_all_encoded_layers=False,
    )
    subword_embeddings = subword_embeddings.detach().cpu().numpy()

    embeddings = []
    for sentence_index, subword_embedding in enumerate(subword_embeddings):
        input_subword_mask = np.asarray(all_input_subword_mask[sentence_index])

        # Create word embedding by averaging its subword embeddings
        word_embeddings = []
        for pos in range(input_subword_mask.max() + 1):
            word_embeddings.append(np.mean(subword_embedding[input_subword_mask == pos], axis=0))

        embeddings.append(word_embeddings)

    return embeddings


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading BERT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("models/bert-base-multilingual-cased/vocab", do_lower_case=False)

    bert_model = BertModel.from_pretrained("models/bert-base-multilingual-cased")
    bert_model.to(device)
    bert_model.eval()  # Use BERT as feature extractor

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(parse.START)
    tag_vocab.index(parse.STOP)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {parse.START, parse.STOP, parse.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)

    print("Initializing model...")
    model = dy.ParameterCollection()
    if args.parser_type == "top-down":
        parser = parse.TopDownParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.split_hidden_dim,
            args.dropout,
        )
    else:
        parser = parse.ChartParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.dropout,
        )
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        for tree in dev_treebank:
            dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
            words = [leaf.word for leaf in tree.leaves()]
            word_embeddings = get_embeddings([words], bert_model, tokenizer, device)[0]
            predicted, _ = parser.parse(sentence, word_embeddings)
            dev_predicted.append(predicted.convert())

        dev_fscore = evaluate.evalb(dev_treebank, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore() > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore()
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore())
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            dy.renew_cg()
            batch_losses = []
            for tree in train_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                words = [leaf.word for leaf in tree.leaves()]
                word_embeddings = get_embeddings([words], bert_model, tokenizer, device)[0]
                if args.parser_type == "top-down":
                    _, loss = parser.parse(sentence, word_embeddings, tree, args.explore)
                else:
                    _, loss = parser.parse(sentence, word_embeddings, tree)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()


def run_test(args):
    print("Loading BERT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("models/bert-base-multilingual-cased/vocab", do_lower_case=False)

    bert_model = BertModel.from_pretrained("models/bert-base-multilingual-cased")
    bert_model.to(device)
    bert_model.eval()  # Use BERT as feature extractor

    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")

    start_time = time.time()

    test_predicted = []
    for tree in test_treebank:
        dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
        words = [leaf.word for leaf in tree.leaves()]
        word_embeddings = get_embeddings([words], bert_model, tokenizer, device)[0]
        predicted, _ = parser.parse(sentence, word_embeddings)
        test_predicted.append(predicted.convert())

    test_fscore = evaluate.evalb(test_treebank, test_predicted)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )


def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--parser-type", choices=["top-down", "chart"], default="chart")
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=768)  # default=100
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--explore", action="store_true")
    subparser.add_argument("--model-path-base", default="outputs")
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", default="corpora/WSJ-PTB/02-21.10way.clean.train")
    subparser.add_argument("--dev-path", default="corpora/WSJ-PTB/22.auto.clean.dev")
    subparser.add_argument("--batch-size", type=int, default=10)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="corpora/WSJ-PTB/23.auto.clean.test")

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()

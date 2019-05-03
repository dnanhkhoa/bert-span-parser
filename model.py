# -*- coding: utf-8 -*-
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from bert.modeling import BertModel, BertPreTrainedModel
from trees import InternalParseNode, LeafParseNode


class ChartParser(BertPreTrainedModel):
    def __init__(
        self, config, tag_encoder, label_encoder, tag_embedding_dim, dropout_prob
    ):
        super(ChartParser, self).__init__(config)

        self.tag_encoder = tag_encoder
        self.label_encoder = label_encoder

        self.bert = BertModel(config)

        self.tag_embeddings = nn.Embedding(
            num_embeddings=tag_encoder.size,
            embedding_dim=tag_embedding_dim,
            padding_idx=tag_encoder.transform("[PAD]"),
        )
        self.label_classifier = nn.Linear(
            in_features=config.hidden_size + tag_embedding_dim,
            out_features=label_encoder.size - 1,  # Skip label ()
        )
        self.dropout = nn.Dropout(dropout_prob)

        self.apply(self.init_bert_weights)

    def __parse(self, embeddings, sentence, gold_tree=None):
        @lru_cache(maxsize=None)
        def get_span_encoding(left, length):
            span_embedding = embeddings.narrow(0, left, length)
            return span_embedding.mean(dim=0)

        @lru_cache(maxsize=None)
        def get_label_scores(left, length):
            label_scores = self.label_classifier(get_span_encoding(left, length))

            # The original code did not use Softmax
            # label_scores = F.softmax(label_scores)

            # Force zero score for label NULL ()
            label_scores = torch.cat(
                [torch.zeros(1, device=label_scores.device), label_scores]
            )
            return label_scores

        def helper(force_gold):
            if force_gold:
                assert gold_tree

            chart = {}

            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length

                    label_scores = get_label_scores(left, length)

                    if gold_tree:
                        oracle_label = gold_tree.oracle_label(left, right)
                        oracle_label_index = self.label_encoder.transform(oracle_label)

                    if force_gold:
                        label = oracle_label
                        label_score = label_scores[oracle_label_index]
                    else:
                        if gold_tree:
                            # Augment scores
                            label_scores += 1
                            label_scores[oracle_label_index] -= 1

                        argmax_label_index = (
                            label_scores.argmax()
                            if length < len(sentence)
                            else label_scores[1:].argmax() + 1
                        ).item()
                        argmax_label = self.label_encoder.inverse_transform(
                            argmax_label_index
                        )
                        label = argmax_label
                        label_score = label_scores[argmax_label_index]

                    if length == 1:
                        tag, word = sentence[left]
                        tree = LeafParseNode(left, tag, word)
                        if label:
                            tree = InternalParseNode(label, [tree])
                        chart[left, right] = [tree], label_score
                        continue

                    if force_gold:
                        oracle_splits = gold_tree.oracle_splits(left, right)
                        best_split = min(oracle_splits)
                    else:
                        best_split = max(
                            range(left + 1, right),
                            key=lambda split: chart[left, split][1]
                            + chart[split, right][1],
                        )

                    left_trees, left_score = chart[left, best_split]
                    right_trees, right_score = chart[best_split, right]

                    children = left_trees + right_trees
                    if label:
                        children = [InternalParseNode(label, children)]

                    chart[left, right] = (
                        children,
                        label_score + left_score + right_score,
                    )

            children, score = chart[0, len(sentence)]
            assert len(children) == 1
            return children[0], score

        tree, score = helper(force_gold=False)

        if gold_tree:
            oracle_tree, oracle_score = helper(force_gold=True)

            linearized_gold_tree = gold_tree.convert().linearize()

            assert oracle_tree.convert().linearize() == linearized_gold_tree

            loss = (
                torch.zeros(1, device=embeddings.device)
                if tree.convert().linearize() == linearized_gold_tree
                else score - oracle_score
            )
            return tree, loss

        return tree, score

    def forward(self, ids, attention_masks, tags, sections, sentences, gold_trees=None):
        subtoken_embeddings, _ = self.bert(
            input_ids=ids,
            attention_mask=attention_masks,
            output_all_encoded_layers=False,
        )

        tag_embeddings = self.tag_embeddings(tags)

        predicted_trees = []
        losses = []

        # Loop over each sample in a mini-batch
        for (
            idx,
            (_subtoken_embeddings, _tag_embeddings, section, sentence),
        ) in enumerate(zip(subtoken_embeddings, tag_embeddings, sections, sentences)):
            gold_tree = gold_trees and gold_trees[idx]

            num_tokens = len(section)
            num_subtokens = sum(section)

            # Remove paddings
            _tag_embeddings = _tag_embeddings.narrow(0, 0, num_tokens)
            _subtoken_embeddings = _subtoken_embeddings.narrow(0, 0, num_subtokens)

            # Merge subtoken embeddings to form a single token embedding
            token_embeddings = []
            for _token_embeddings in _subtoken_embeddings.split(section, dim=0):
                token_embeddings.append(_token_embeddings.mean(dim=0, keepdim=True))

            token_embeddings = torch.cat(token_embeddings, dim=0)

            embeddings = torch.cat([token_embeddings, _tag_embeddings], dim=-1)

            embeddings = self.dropout(embeddings)

            predicted_tree, loss = self.__parse(embeddings, sentence, gold_tree)

            predicted_trees.append(predicted_tree)
            losses.append(loss)

        return predicted_trees, losses

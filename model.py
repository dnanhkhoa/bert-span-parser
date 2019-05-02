# -*- coding: utf-8 -*-
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from bert.modeling import BertModel, BertPreTrainedModel


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
            if span_embedding.size(0) > 1:
                return span_embedding.mean(dim=0, keepdim=True)
            return span_embedding

        @lru_cache(maxsize=None)
        def get_label_scores(left, length):
            label_scores = self.label_classifier(get_span_encoding(left, length))

            # The original code did not use Softmax
            label_scores = F.softmax(label_scores, dim=-1)

            # Force zero score for label NULL ()
            label_scores = torch.cat(
                [
                    torch.zeros((label_scores.size(0), 1), device=label_scores.device),
                    label_scores,
                ],
                dim=-1,
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

        tree, score = helper(False)

        if gold_tree:
            oracle_tree, oracle_score = helper(True)

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

            token_embeddings = []
            for _token_embeddings in _subtoken_embeddings.split(section, dim=0):
                if _token_embeddings.size(0) > 1:
                    token_embeddings.append(_token_embeddings.mean(dim=0, keepdim=True))
                    # token_embeddings.append(_token_embeddings.sum(dim=0, keepdim=True))
                else:
                    token_embeddings.append(_token_embeddings)

            token_embeddings = torch.cat(token_embeddings, dim=0)

            embeddings = torch.cat([token_embeddings, _tag_embeddings], dim=1)

            embeddings = self.dropout(embeddings)

            predicted_tree, loss = self.__parse(embeddings, sentence, gold_tree)

            predicted_trees.append(predicted_tree)
            losses.append(loss)

        return predicted_trees, losses

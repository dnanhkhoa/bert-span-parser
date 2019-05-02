# -*- coding: utf-8 -*-
from functools import lru_cache

import torch
from torch import functional as F
from torch import nn

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
        def get_span_encoding(left, right):
            pass

        @lru_cache(maxsize=None)
        def get_label_scores(left, right):
            pass

        return 0, 0

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

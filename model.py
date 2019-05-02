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
            out_features=label_encoder.size - 1,  # DO NOT INCLUDE LABEL ()
            bias=True,
        )
        self.dropout = nn.Dropout(dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, ids, attention_masks, tags, sections, sentences, gold_trees=None):
        @lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            pass

        @lru_cache(maxsize=None)
        def get_label_scores(left, right):
            pass

        return 0, 0

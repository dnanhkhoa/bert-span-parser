# -*- coding: utf-8 -*-
from functools import lru_cache

import torch
from torch import functional as F
from torch import nn

from bert.modeling import BertModel, BertPreTrainedModel


class ChartParser(BertPreTrainedModel):
    def __init__(self, config):
        super(ChartParser, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.label_classifier = nn.Linear(config.hidden_size, 5)

        self.apply(self.init_bert_weights)

    def forward(self):
        pass

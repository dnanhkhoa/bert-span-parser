# -*- coding: utf-8 -*-
import torch
from torch import nn

from bert.modeling import ACT2FN, BertModel, BertPreTrainedModel
from trees import InternalParseNode, LeafParseNode


class ChartParser(BertPreTrainedModel):
    def __init__(
        self,
        config,
        tag_encoder,
        label_encoder,
        lstm_layers,
        lstm_dim,
        tag_embedding_dim,
        label_hidden_dim,
        dropout_prob,
    ):
        super(ChartParser, self).__init__(config)

        self.tag_encoder = tag_encoder
        self.label_encoder = label_encoder

        self.bert = BertModel(config)

        self.tag_embeddings = nn.Embedding(
            num_embeddings=len(tag_encoder),
            embedding_dim=tag_embedding_dim,
            padding_idx=tag_encoder.transform("[PAD]"),
        )

        self.hidden_dense = nn.Linear(
            in_features=config.hidden_size + tag_embedding_dim, out_features=250
        )

        self.intermediate_act_fn = ACT2FN[config.hidden_act]

        self.label_classifier = nn.Linear(
            in_features=250, out_features=len(label_encoder) - 1  # Skip label ()
        )
        self.dropout = nn.Dropout(dropout_prob)

        self.apply(self.init_bert_weights)

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

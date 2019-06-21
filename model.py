# -*- coding: utf-8 -*-
import numpy as np
import pyximport
import torch
import torch.nn.functional as F
from torch import nn

from bert.modeling import ACT2FN, BertLayerNorm, BertModel, BertPreTrainedModel
from trees import InternalParseNode, LeafParseNode

pyximport.install(
    build_dir=".caches", setup_args={"include_dirs": np.get_include()}, language_level=3
)

import chart_decoder


class Attention(nn.Module):
    def __init__(self, config, embedding_dim, hidden_dim, dropout_prob):
        super().__init__()

        self.hidden_dense_1 = nn.Linear(
            in_features=embedding_dim, out_features=hidden_dim
        )

        # self.hidden_dense_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        self.score = nn.Linear(in_features=hidden_dim, out_features=1)

        self.intermediate_act_fn = ACT2FN[config.hidden_act]

        self.layer_norm = BertLayerNorm(hidden_dim, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, embeddings):
        # return self.score(
        #     self.dropout(
        #         self.layer_norm(
        #             self.intermediate_act_fn(
        #                 self.hidden_dense_2(
        #                     self.dropout(
        #                         self.layer_norm(
        #                             self.intermediate_act_fn(
        #                                 self.hidden_dense_1(embeddings)
        #                             )
        #                         )
        #                     )
        #                 )
        #             )
        #         )
        #     )
        # )

        return self.score(
            self.dropout(
                self.layer_norm(
                    self.intermediate_act_fn(self.hidden_dense_1(embeddings))
                )
            )
        )


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

        self.bert = BertModel(config)

        self.tag_encoder = tag_encoder
        self.label_encoder = label_encoder

        self.lstm = nn.LSTM(
            input_size=tag_embedding_dim + config.hidden_size,
            hidden_size=lstm_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True,
        )

        self.tag_embeddings = nn.Embedding(
            num_embeddings=len(tag_encoder),
            embedding_dim=tag_embedding_dim,
            padding_idx=tag_encoder.transform("[PAD]"),
        )

        self.attention = Attention(
            config=config,
            embedding_dim=lstm_dim * 2,
            hidden_dim=150,
            dropout_prob=dropout_prob,
        )

        self.hidden_dense = nn.Linear(
            in_features=3 * lstm_dim * 2, out_features=label_hidden_dim
        )

        self.label_classifier = nn.Linear(
            in_features=label_hidden_dim,
            out_features=len(label_encoder) - 1,  # Skip label ()
        )

        self.intermediate_act_fn = ACT2FN[config.hidden_act]

        self.layer_norm = BertLayerNorm(label_hidden_dim, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, ids, attention_masks, tags, sections, sentences, gold_trees=None):
        is_training = gold_trees is not None

        subtoken_embeddings, _ = self.bert(
            input_ids=ids,
            attention_mask=attention_masks,
            output_all_encoded_layers=False,
        )

        tag_embeddings = self.tag_embeddings(tags)

        sentence_sections = [0]
        span_embeddings = []

        # Loop over each sample in a mini-batch
        for _subtoken_embeddings, _tag_embeddings, section, sentence in zip(
            subtoken_embeddings, tag_embeddings, sections, sentences
        ):
            num_tokens = len(section)
            num_subtokens = sum(section)

            # Remove paddings
            _tag_embeddings = _tag_embeddings.narrow(dim=0, start=0, length=num_tokens)
            _subtoken_embeddings = _subtoken_embeddings.narrow(
                dim=0, start=1, length=num_subtokens
            )

            # Merge subtoken embeddings to form a single token embedding
            token_embeddings = []
            for _token_embeddings in _subtoken_embeddings.split(section, dim=0):
                # V0: Use mean of all subtoken embeddings as word embedding
                # token_embeddings.append(_token_embeddings.mean(dim=0, keepdim=True))

                # V1: Use the first subtoken embedding as word embedding
                token_embeddings.append(
                    _token_embeddings.narrow(dim=0, start=0, length=1)
                )

                # V2: Use the last subtoken embedding as word embedding
                # token_embeddings.append(
                #     _token_embeddings.narrow(dim=0, start=-1, length=1)
                # )

                # V3: Use difference between the last and first subtoken embedding as word embedding
                # => Failed
                # token_embeddings.append(
                #     _token_embeddings.narrow(dim=0, start=-1, length=1)
                #     - _token_embeddings.narrow(dim=0, start=0, length=1)
                # )

                # V4: Use mean of the first and last subtoken embedding as word embedding
                # token_embeddings.append(
                #     torch.cat(
                #         (
                #             _token_embeddings.narrow(dim=0, start=0, length=1),
                #             _token_embeddings.narrow(dim=0, start=-1, length=1),
                #         )
                #     ).mean(dim=0, keepdim=True)
                # )

            token_embeddings = torch.cat(token_embeddings, dim=0)

            token_embeddings = torch.cat([_tag_embeddings, token_embeddings], dim=-1)

            token_embeddings = self.dropout(token_embeddings)

            lstm_embeddings, _ = self.lstm(token_embeddings.unsqueeze(dim=0))

            lstm_embeddings = lstm_embeddings.squeeze(dim=0)

            # Generate spans
            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length

                    left_embedding = lstm_embeddings[left]

                    right_embedding = lstm_embeddings[right - 1]

                    # V0: unweighted average
                    # average_embedding = lstm_embeddings.narrow(
                    #     dim=0, start=left, length=length
                    # ).mean(dim=0)

                    # span_embeddings.append(
                    #     torch.cat([left_embedding, average_embedding, right_embedding])
                    # )

                    # V1: Use attention as weighted average
                    span_embedding = lstm_embeddings.narrow(
                        dim=0, start=left, length=length
                    )
                    attentions_weights = self.attention(span_embedding)

                    attentions_weights = F.softmax(attentions_weights, dim=0)

                    weighted_span_embedding = torch.mul(
                        span_embedding, attentions_weights
                    ).sum(dim=0)

                    span_embeddings.append(
                        torch.cat(
                            [left_embedding, weighted_span_embedding, right_embedding]
                        )
                    )

            sentence_sections.append(len(span_embeddings))

        span_embeddings = torch.stack(span_embeddings)

        label_scores = self.label_classifier(
            self.dropout(
                self.layer_norm(
                    self.intermediate_act_fn(self.hidden_dense(span_embeddings))
                )
            )
        )

        # Add score 0 for NULL label
        label_scores = torch.cat(
            (label_scores.new_zeros(label_scores.size(0), 1), label_scores), dim=-1
        )

        total_augmented_amount = 0.0

        all_pred_trees = []

        all_pred_indices = []
        all_pred_labels = []

        all_gold_indices = []
        all_gold_labels = []

        for sentence_idx, np_label_scores in enumerate(
            np.split(label_scores.detach().cpu().numpy(), sentence_sections[1:-1])
        ):
            decoder_args = {
                "sentence_len": len(sections[sentence_idx]),
                "num_previous_indices": sentence_sections[sentence_idx],
                "label_scores": np_label_scores,
                "is_training": is_training,
                "gold_tree": gold_trees and gold_trees[sentence_idx],
                "label_encoder": self.label_encoder,
            }

            _, pred_included_i, pred_included_j, pred_included_indices, pred_included_labels, pred_augmented_amount = chart_decoder.decode(
                False, **decoder_args
            )

            all_pred_indices.append(pred_included_indices)
            all_pred_labels.append(pred_included_labels)

            total_augmented_amount += pred_augmented_amount

            if is_training:
                _, _, _, gold_included_indices, gold_included_labels, _ = chart_decoder.decode(
                    True, **decoder_args
                )

                all_gold_indices.append(gold_included_indices)
                all_gold_labels.append(gold_included_labels)
            else:
                stack_idx = -1

                def make_tree():
                    nonlocal stack_idx
                    stack_idx += 1

                    pos_i, pos_j, label_index = (
                        pred_included_i[stack_idx],
                        pred_included_j[stack_idx],
                        pred_included_labels[stack_idx],
                    )

                    pos_i = pos_i.item()
                    pos_j = pos_j.item()

                    pred_label = self.label_encoder.inverse_transform(label_index)

                    if pos_i + 1 >= pos_j:
                        tag, word = sentences[sentence_idx][pos_i]
                        tree = LeafParseNode(pos_i, tag, word)
                        if pred_label:
                            tree = InternalParseNode(pred_label, [tree])
                        return [tree]
                    else:
                        left_trees = make_tree()
                        right_trees = make_tree()
                        children = left_trees + right_trees
                        if pred_label:
                            return [InternalParseNode(pred_label, children)]
                        else:
                            return children

                all_pred_trees.append(make_tree()[0])

        # Is training
        if is_training:
            all_pred_indices = ids.new_tensor(np.concatenate(all_pred_indices))
            all_pred_labels = ids.new_tensor(np.concatenate(all_pred_labels))

            all_gold_indices = ids.new_tensor(np.concatenate(all_gold_indices))
            all_gold_labels = ids.new_tensor(np.concatenate(all_gold_labels))

            loss = (
                label_scores[all_pred_indices, all_pred_labels].sum()
                - label_scores[all_gold_indices, all_gold_labels].sum()
                + total_augmented_amount
            )

            return loss

        return all_pred_trees

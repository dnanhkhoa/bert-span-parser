# -*- coding: utf-8 -*-
# This code is adapted from https://github.com/junekihong/beam-span-parser
from collections import defaultdict


# Juneki: Mitchell Stern's original code calls the EVALB script.
# It turns out, this script cannot handle particularly longs sentences.
# Which was a problem for our Discourse experiments.
# We have changed this to use our own faithful implementation of EVALB.
def evalb(gold_trees, predicted_trees):
    assert len(gold_trees) == len(predicted_trees)
    score = FScore()
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        gold_phrase_tree = PhraseTree.parse(gold_tree.linearize())
        predicted_phrase_tree = PhraseTree.parse(predicted_tree.linearize())
        score += predicted_phrase_tree.compare(gold_phrase_tree)
    return score


class FScore(object):
    def __init__(self, correct=0, pred_count=0, gold_count=0):
        self.correct = correct  # correct brackets
        self.pred_count = pred_count  # total predicted brackets
        self.gold_count = gold_count  # total gold brackets

    def precision(self):
        if self.pred_count > 0:
            return (100.0 * self.correct) / self.pred_count
        else:
            return 0.0

    def recall(self):
        if self.gold_count > 0:
            return (100.0 * self.correct) / self.gold_count
        else:
            return 0.0

    def fscore(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) > 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0

    def __str__(self):
        return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f})".format(
            self.recall(), self.precision(), self.fscore()
        )

    def __iadd__(self, other):
        self.correct += other.correct
        self.pred_count += other.pred_count
        self.gold_count += other.gold_count
        return self


class PhraseTree(object):

    puncs = [",", ".", ":", "``", "''", "PU"]  ## (COLLINS.prm)

    def __init__(self, symbol=None, children=[], sentence=[], leaf=None):
        self.symbol = symbol  # label at top node
        self.children = children  # list of PhraseTree objects
        self.sentence = sentence
        self.leaf = leaf  # word at bottom level else None
        self._str = None

    @staticmethod
    def parse(line):
        _, tree = PhraseTree._parse(line + " ", 0, [])
        return tree

    @staticmethod
    def _parse(line, index, sentence):
        assert line[index] == "(", "Invalid tree string {} at {}".format(line, index)

        index += 1
        symbol = None
        children = []
        leaf = None

        while line[index] != ")":
            if line[index] == "(":
                index, tree = PhraseTree._parse(line, index, sentence)
                children.append(tree)
            else:
                if symbol is None:
                    rpos = min(line.find(" ", index), line.find(")", index))
                    symbol = line[index:rpos]
                else:
                    rpos = line.find(")", index)
                    word = line[index:rpos]
                    sentence.append((word, symbol))
                    leaf = len(sentence) - 1

                index = rpos

            if line[index] == " ":
                index += 1

        assert line[index] == ")", "Invalid tree string {} at {}".format(line, index)

        tree = PhraseTree(
            symbol=symbol, children=children, sentence=sentence, leaf=leaf
        )

        return (index + 1), tree

    def left_span(self):
        try:
            return self._left_span
        except AttributeError:
            if self.leaf is None:
                self._left_span = self.children[0].left_span()
            else:
                self._left_span = self.leaf
            return self._left_span

    def right_span(self):
        try:
            return self._right_span
        except AttributeError:
            if self.leaf is None:
                self._right_span = self.children[-1].right_span()
            else:
                self._right_span = self.leaf
            return self._right_span

    def brackets(self, advp_prt=True, counts=None):
        if counts is None:
            counts = defaultdict(int)

        if self.leaf is not None:
            return {}

        nonterm = self.symbol
        if advp_prt and nonterm == "PRT":
            nonterm = "ADVP"

        left = self.left_span()
        right = self.right_span()

        # ignore punctuation
        while left < len(self.sentence) and self.sentence[left][1] in PhraseTree.puncs:
            left += 1
        while right > 0 and self.sentence[right][1] in PhraseTree.puncs:
            right -= 1

        if left <= right and nonterm != "TOP":
            counts[(nonterm, left, right)] += 1

        for child in self.children:
            child.brackets(advp_prt=advp_prt, counts=counts)

        return counts

    def compare(self, gold, advp_prt=True):
        pred_bracks = self.brackets(advp_prt)
        gold_bracks = gold.brackets(advp_prt)

        correct = 0
        for gb in gold_bracks:
            if gb in pred_bracks:
                correct += min(gold_bracks[gb], pred_bracks[gb])

        pred_total = sum(pred_bracks.values())
        gold_total = sum(gold_bracks.values())

        return FScore(correct, pred_total, gold_total)

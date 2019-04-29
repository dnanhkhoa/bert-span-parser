# -*- coding: utf-8 -*-
# This code is adapted from https://github.com/junekihong/beam-span-parser

import math
from collections import defaultdict


# Juneki: Mitchell Stern's original code calls the EVALB script.
# It turns out, this script cannot handle particularly longs sentences.
# Which was a problem for our Discourse experiments.
# We have changed this to use our own faithful implementation of EVALB.
def evalb(gold_trees, predicted_trees):
    result = FScore()
    assert len(gold_trees) == len(predicted_trees)
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        gold_phrase_tree = PhraseTree.parse(str(gold_tree.linearize()))
        predicted_phrase_tree = PhraseTree.parse(str(predicted_tree.linearize()))
        result += predicted_phrase_tree.compare(gold_phrase_tree)
    return result


class FScore(object):
    def __init__(self, correct=0, predcount=0, goldcount=0):
        self.correct = correct  # correct brackets
        self.predcount = predcount  # total predicted brackets
        self.goldcount = goldcount  # total gold brackets

    def precision(self):
        if self.predcount > 0:
            return (100.0 * self.correct) / self.predcount
        else:
            return 0.0

    def recall(self):
        if self.goldcount > 0:
            return (100.0 * self.correct) / self.goldcount
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
        self.predcount += other.predcount
        self.goldcount += other.goldcount
        return self


class PhraseTree(object):

    puncs = [",", ".", ":", "``", "''", "PU"]  ## (COLLINS.prm)

    def __init__(self, symbol=None, children=[], sentence=[], leaf=None):
        self.symbol = symbol  # label at top node
        self.children = children  # list of PhraseTree objects
        self.sentence = sentence
        self.leaf = leaf  # word at bottom level else None

        self._str = None

    def __str__(self):
        if True or self._str is None:
            children = [
                child
                for child in self.children
                if (len(child.children) > 0 or child.leaf is not None)
            ]

            if len(children) != 0:
                childstr = " ".join(str(c) for c in children)
                self._str = "({} {})".format(self.symbol, childstr)
            elif self.leaf is not None:
                self._str = "({} {})".format(
                    # self.sentence[self.leaf][1],
                    self.symbol,
                    self.sentence[self.leaf][0],
                )
            else:
                self._str = ""
        return self._str

    @staticmethod
    def parse(line):
        """
        Loads a tree from a tree in PTB parenthetical format.
        """
        line += " "
        sentence = []
        _, t = PhraseTree._parse(line, 0, sentence)

        # if t.symbol == 'TOP' and len(t.children) == 1:
        #    t = t.children[0]

        return t

    @staticmethod
    def _parse(line, index, sentence):
        "((...) (...) w/t (...)). returns pos and tree, and carries sent out."

        assert line[index] == "(", "Invalid tree string {} at {}".format(line, index)
        index += 1
        symbol = None
        children = []
        leaf = None
        while line[index] != ")":
            if line[index] == "(":
                index, t = PhraseTree._parse(line, index, sentence)
                children.append(t)
            else:
                if symbol is None:
                    # symbol is here!
                    rpos = min(line.find(" ", index), line.find(")", index))
                    # see above N.B. (find could return -1)

                    symbol = line[index:rpos]  # (word, tag) string pair

                    index = rpos
                else:
                    rpos = line.find(")", index)
                    word = line[index:rpos]
                    sentence.append((word, symbol))
                    leaf = len(sentence) - 1
                    index = rpos

            if line[index] == " ":
                index += 1

        assert line[index] == ")", "Invalid tree string %s at %d" % (line, index)

        t = PhraseTree(symbol=symbol, children=children, sentence=sentence, leaf=leaf)

        return (index + 1), t

    def left_span(self):
        try:
            return self._left_span
        except AttributeError:
            if self.leaf is not None:
                self._left_span = self.leaf
            else:
                self._left_span = self.children[0].left_span()
            return self._left_span

    def right_span(self):
        try:
            return self._right_span
        except AttributeError:
            if self.leaf is not None:
                self._right_span = self.leaf
            else:
                self._right_span = self.children[-1].right_span()
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
        """
        returns (Precision, Recall, F-measure)
        """
        predbracks = self.brackets(advp_prt)
        goldbracks = gold.brackets(advp_prt)

        correct = 0
        for gb in goldbracks:
            if gb in predbracks:
                correct += min(goldbracks[gb], predbracks[gb])

        pred_total = sum(predbracks.values())
        gold_total = sum(goldbracks.values())

        return FScore(correct, pred_total, gold_total)

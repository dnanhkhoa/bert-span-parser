# -*- coding: utf-8 -*-
from functools import lru_cache


class TreebankNode:
    pass


class LeafTreebankNode(TreebankNode):
    __slots__ = ["tag", "word"]

    def __init__(self, tag, word):
        self.tag = tag
        self.word = word

    @lru_cache(maxsize=None)
    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    @lru_cache(maxsize=None)
    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)


class InternalTreebankNode(TreebankNode):
    __slots__ = ["label", "children"]

    def __init__(self, label, children):
        self.label = label
        self.children = children

    @lru_cache(maxsize=None)
    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children)
        )

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    @lru_cache(maxsize=None)
    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
            tree.children[0], InternalTreebankNode
        ):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(sublabels, children)


class ParseNode:
    pass


class LeafParseNode(ParseNode):
    __slots__ = ["tag", "word", "left", "right"]

    def __init__(self, index, tag, word):
        self.tag = tag
        self.word = word
        self.left = index
        self.right = index + 1

    @lru_cache(maxsize=None)
    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    @lru_cache(maxsize=None)
    def convert(self):
        return LeafTreebankNode(self.tag, self.word)


class InternalParseNode(ParseNode):
    __slots__ = ["label", "children", "left", "right"]

    def __init__(self, label, children):
        self.label = label
        self.children = children
        self.left = children[0].left
        self.right = children[-1].right

    @lru_cache(maxsize=None)
    def linearize(self):
        return "({} {})".format(
            "->".join(self.label),
            " ".join(child.linearize() for child in self.children),
        )

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    @lru_cache(maxsize=None)
    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    @lru_cache(maxsize=None)
    def enclosing(self, left, right):
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    @lru_cache(maxsize=None)
    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()  # Leaf nodes

    @lru_cache(maxsize=None)
    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]


def load_trees(fn, strip_top=True):
    with open(fn, "r", encoding="UTF-8") as f:
        tokens = f.read().replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            parent_count = 0
            while tokens[index] == "(":
                index += 1
                parent_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while parent_count > 0:
                assert tokens[index] == ")"
                index += 1
                parent_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label in ("TOP", "ROOT"):
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    return trees

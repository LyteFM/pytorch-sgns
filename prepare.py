#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:16:00 2020

@author: fynn
"""
import os
import pickle
from typing import List, Dict

from nltk.corpus import treebank
from nltk.util import everygrams


# only include actual words & punctuation, write a small training set
words = [p[0] for p in treebank.tagged_words() if p[1] not in {'-NONE-', '-LRB-', '-RRB-'}]

with open('data/corpus.txt', 'w') as f:
    f.write(' '.join(words[:1000]))


class NGramHandler:
    """
    - Constructs the ngrams for every word in a given dictionary.
    - Computes the score function given a word $w_t$ and its context words $c ∈ C_t$
    „Ultimately, a word is represented by its index in the word dictionary and the set of hashed n-grams it contains.”
    """

    def __init__(self):
        self.idx2word: List = pickle.load(open(os.path.join('data', 'idx2word.dat'), 'rb'))
        self.word2idx: Dict = pickle.load(open(os.path.join('data', 'word2idx.dat'), 'rb'))

        # extract all the n-grams for n greater or equal to 3 and smaller or equal to 6, add '<'/ '>' in start/end
        self.idx2ngrams = [list(map(''.join, everygrams(w, min_len=3, max_len=6))) for w in (f'<{x}>' for x in self.idx2word)]
        for i, ngrams in enumerate(self.idx2ngrams):
            full = f'<{self.idx2word[i]}>'
            if ngrams[-1] != full:
                ngrams.append(full)
        pass

    def score(self, w_idx, c_idx):
        """
        Given a single word and context, look up the vector representation of the word's n-grams $z_g ∈ G_w$ and the
        context vector $v_c$ and compute the score $Σ_{g ∈ G_w} z_g^T v_c$
        - this means: only the n-grams of the _current_ word are considered, $v_c$ is still just the vector for $c$.
        :param w_idx:
        :param c_idx:
        :return:
        """
        pass


if __name__ == '__main__':
    test = NGramHandler()

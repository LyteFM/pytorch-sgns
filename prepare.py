#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:16:00 2020

@author: fynn
"""
from nltk.corpus import treebank

# only include actual words & punctuation, write a small training set
words = [p[0] for p in treebank.tagged_words() if p[1] not in {'-NONE-', '-LRB-', '-RRB-'}]

with open('data/corpus.txt', 'w') as f:
    f.write(' '.join(words[:1000]))

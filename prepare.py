#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:16:00 2020

@author: fynn
"""
from nltk.corpus import treebank

# only include actual words & punctuation, write a small training set
# todo: actually, use sents!!!
sents = []
for s in treebank.tagged_sents():
    sents.append([p[0] for p in s if p[1] not in {'-NONE-', '-LRB-', '-RRB-', '``', "''"}])
    pass
print('got corpus with sentences: ', len(sents))

with open('data/corpus.txt', 'w') as f:
    for s in sents:
        f.write(' '.join(s) + '\n')

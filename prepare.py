#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:16:00 2020

@author: fynn
"""
from nltk.corpus import treebank
import numpy as np
import requests

# Change here. todo: build CLI
selected_corpus = 'ud' # treebank
ud_lang = 'de' # 'en' , 'it
ud_mode = 'dev'  # 'train', 'test'


sents = []
if selected_corpus == 'treebank':
    for s in treebank.tagged_sents():
        sents.append([p[0] for p in s if p[1] not in {'-NONE-', '-LRB-', '-RRB-'}]) # '``', "''"
else:
    filename = f'{ud_lang}-ud-{ud_mode}.conllu'
    try:
        data = np.loadtxt('data/' + filename, dtype=str, delimiter='\t', comments=None)
    except OSError:
        r = requests.get(f'https://raw.githubusercontent.com/ufal/rh_nntagging/master/data/ud-1.2/{ud_lang}/'+filename)
        with open('data/'+filename, 'w') as f:
            f.write(r.text)
        data = np.loadtxt('data/' + filename, dtype=str, delimiter='\t', comments=None)

    curr = []
    for w in data:
        if w[0] == '1':
            if len(curr) != 0:
                sents.append(curr)
            curr = []
        # neither the unknown character (I'm using |) nor the nonexisting char (I'm using *) may be in corpus.
        if '*' not in w[1] and '|' not in w[1] and (len(w[1]) == 1 or any(c.isalnum() for c in w[1])):
            curr.append(w[1])
print('got corpus with sentences: ', len(sents))

with open('data/corpus.txt', 'w') as f:
    for s in sents:
        f.write(' '.join(s) + '\n')

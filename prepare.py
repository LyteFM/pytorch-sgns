#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:16:00 2020

@author: fynn
"""

import numpy as np
import requests
import os


# Change here. todo: build CLI
ud_lang = 'de' # 'en' # 'it' #
ud_mode = 'train' #  'test' # 'dev' #

comm = {'de': None, 'en': None, 'it':'#'}


if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists(f'data/{ud_lang}'):
    os.mkdir(f'data/{ud_lang}')


sents = []

filename = f'{ud_lang}-ud-{ud_mode}.conllu'
try:
    data = np.loadtxt(f'data/{ud_lang}/' + filename, dtype=str, delimiter='\t', comments=comm[ud_lang])
except OSError:
    r = requests.get(f'https://raw.githubusercontent.com/ufal/rh_nntagging/master/data/ud-1.2/{ud_lang}/'+filename)
    with open(f'data/{ud_lang}/'+filename, 'w') as f:
        f.write(r.text)
    data = np.loadtxt(f'data/{ud_lang}/' + filename, dtype=str, delimiter='\t', comments=comm[ud_lang])

curr = []
for w in data:
    if w[0] == '1':
        if len(curr) != 0:
            sents.append(curr)
        curr = []
    # the unknown character (I'm using |) may not be in corpus.
    if '|' not in w[1] and (len(w[1]) == 1 or any(c.isalnum() for c in w[1])):
        curr.append(w[1])
print('got corpus with sentences: ', len(sents))


with open(f'data/{ud_lang}/corpus_{ud_mode}.txt', 'w') as f:
    for s in sents:
        f.write(' '.join(s) + '\n')

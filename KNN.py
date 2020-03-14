#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import argparse
import matplotlib
import numpy as np

from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lan', type=str, default='en', help="language initials")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./qualitative analysis/', help="analysis results directory path")
    parser.add_argument('--freq_min', type=int, default=1, help="minimum frequence of rare words")
    parser.add_argument('--freq_max', type=int, default=1, help="maximum frequence of rare words")
    parser.add_argument('--k', type=int, default=3, help="number of nearest neighbors")
    return parser.parse_args()

def IdxKNN(pcs, k):
    idx, _ = zip(*sorted(list(enumerate(pcs)), key=itemgetter(1), reverse=True))
    return list(idx[1:k+1])

def RareWords(wc, freq_max = 1, freq_min = 1):
    return sorted(filter(lambda x: wc[x] <=freq_max & wc[x] >= freq_min, wc), key = wc.get, reverse=False)

def GenerateKNNs(args):
    # dictionary = { word: occurrencies }
    word_counts = pickle.load(open(os.path.join(args.data_dir, args.lan, 'wc.dat'), 'rb'))
    # dictionary = { word: index }
    word2idx = pickle.load(open(os.path.join(args.data_dir, args.lan, 'word2idx.dat'), 'rb'))
    # list (len = vocabulary_size). Element i = word with index i
    idx2word = pickle.load(open(os.path.join(args.data_dir, args.lan, 'idx2word.dat'), 'rb'))
    # ndarray ( vocabulary_size x embedding_size ). Row i = embedding WITHOUT SUBWORDS' INFO of word with index i.
    idx2vec = pickle.load(open(os.path.join(args.data_dir, args.lan, 'idx2vec.dat'), 'rb'))
    # ndarray ( vocabulary_size x embedding_size ). Row i = embedding WITH SUBWORDS' INFO of word with index i.
    idx2vec_ngrams = pickle.load(open(os.path.join(args.data_dir, args.lan, 'idx2vec_ngrams.dat'), 'rb'))

    rare_words = RareWords(word_counts, args.freq_max, args.freq_min)
    print('We are analyzing {} words.'.format(len(rare_words)))

    # Matrix of paiwise cosine similarity
    # NO SUBWORDS' INFO
    PCS = cosine_similarity(idx2vec)
    # WITH SUBWORDS' INFO
    PCS_ngrams = cosine_similarity(idx2vec_ngrams)

    KNNs = dict()
    for rw in rare_words:
        knn = IdxKNN(PCS[word2idx[rw],:], args.k)
        knn_ngrams = IdxKNN(PCS_ngrams[word2idx[rw],:], args.k)
        KNNs[rw] = {'sg': [idx2word[i] for i in knn], 'sgsi':[idx2word[i] for i in knn_ngrams]}

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.isdir(os.path.join(args.save_dir, args.lan)):
        os.mkdir(os.path.join(args.save_dir, args.lan))

    pickle.dump(KNNs, open(os.path.join(args.save_dir, args.lan, 'KNNs.dat'), 'wb'))

if __name__=='__main__':
    GenerateKNNs(parse_args())

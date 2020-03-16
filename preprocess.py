# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
from collections import defaultdict

from nltk.util import everygrams


def parse_args(lan, dataset):
    """
    lan: language, e.g. 'de', 'en', 'it'
    dataset: set of data, e.g. 'dev', 'test', 'train'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=f'./data/{lan}/', help="data directory path")
    parser.add_argument('--vocab', type=str, default=f'./data/{lan}/corpus_{dataset}.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default=f'./data/{lan}/corpus_{dataset}.txt', help="corpus path")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token, should be a single character when using ngrams")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000,
                        help="maximum size of vocab. Infrequent words will be stripped when exceeding the size.")
    parser.add_argument('--min_freq', type=int, default=1,
                        help="Minimum word frequency for inclusion in vocabulary")
    parser.add_argument('--ngrams', action='store_true', help="for n-gram based training")
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, window=5, unk='<UNK>', data_dir='./data/', ngrams=False):
        self.window = window
        self.unk = unk  # must be the first char of the vocab, will always be zero in the embeddings.
        self.data_dir = data_dir
        self.use_ngrams = ngrams

        self.wc = {self.unk: 0}
        self.idx2word = []
        self.word2idx = dict()
        self.vocab = set()

        self.ngram_idx2ngram = []
        self.ngram2ngramCounts = defaultdict(int)
        self.ngram2ngram_idx = dict()
        self.word_idx2ngrams = []
        self.word_idx2corresp_ngram = []

    def skipgram(self, sentence, i):
        """
        returs the context of the specified window size (possibly random) for the word at position i in a sentence
        """
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self, filepath, max_vocab=20000, min_freq=1):
        print("building vocab...")
        step = 0

        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                sent = line.split()
                for word in sent:
                    self.wc[word] = self.wc.get(word, 0) + 1
        print("initially, vocab has", len(self.wc)+1, "words.")
        self.idx2word = [self.unk] + sorted(filter(lambda x: self.wc[x] >= min_freq, self.wc),
                                            key=self.wc.get, reverse=True)[:max_vocab - 1]
        print("with max size", max_vocab, "and min freq", min_freq, "vocab now has", len(self.idx2word), "words.")
        if self.wc[self.unk] > 0:
            print('WARN: unknown character may not be included in corpus!')
        else:
            self.wc[self.unk] = 1  # avoid 0-div error
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])

        if self.use_ngrams:
            self.ngram_idx2ngram = []
            self.ngram2ngram_idx = dict()
            self.word_idx2ngrams = [list(map(''.join, everygrams(w, min_len=3, max_len=6))) for w in (f'<{x}>' for x in self.idx2word)]
            self.word_idx2corresp_ngram = []

            for i, ngrams in enumerate(self.word_idx2ngrams):
                full = f'<{self.idx2word[i]}>'
                if ngrams[-1] != full:
                    ngrams.append(full)
                for ngram in ngrams:
                    self.ngram2ngramCounts[ngram] += 1
                    if ngram not in self.ngram2ngram_idx:
                        curr_ngram_idx = len(self.ngram_idx2ngram)
                        self.ngram2ngram_idx[ngram] = curr_ngram_idx
                        self.ngram_idx2ngram.append(ngram)
                        if ngram == full:
                            self.word_idx2corresp_ngram.append(curr_ngram_idx)
            self.word_idx2ngram_indices = []
            for i, ngrams in enumerate(self.word_idx2ngrams):
                self.word_idx2ngram_indices.append([self.ngram2ngram_idx[x] for x in ngrams])

            pickle.dump(self.ngram2ngramCounts, open(os.path.join(self.data_dir, 'ngram2ngramCounts.dat'), 'wb'))
            pickle.dump(self.word_idx2ngrams, open(os.path.join(self.data_dir, 'word_idx2ngrams.dat'), 'wb'))
            pickle.dump(self.ngram2ngram_idx, open(os.path.join(self.data_dir, 'ngram2ngram_idx.dat'), 'wb'))
            pickle.dump(self.ngram_idx2ngram, open(os.path.join(self.data_dir, 'ngram_idx2ngram.dat'), 'wb'))
            pickle.dump(self.word_idx2ngram_indices, open(os.path.join(self.data_dir, 'word_idx2ngram_indices.dat'), 'wb'))
            pickle.dump(self.word_idx2corresp_ngram, open(os.path.join(self.data_dir, 'word_idx2corresp_ngram.dat'), 'wb'))

        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'wc.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        print("build done")

    def convert(self, filepath):
        print("converting corpus...")
        step = 0
        data = []
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                sent = []
                for word in line.split():
                    if word in self.vocab:
                        sent.append(word)
                    else:
                        sent.append(self.unk)
                for i in range(len(sent)):
                    iword, owords = self.skipgram(sent, i)
                    data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
        print("")
        pickle.dump(data, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        print("conversion done")


if __name__ == '__main__':
    for lan in ['de', 'en', 'it']:
        dataset = 'train'
        args = parse_args(lan, dataset)
        preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir, ngrams=args.ngrams)
        preprocess.build(args.vocab, max_vocab=args.max_vocab, min_freq=args.min_freq)
        preprocess.convert(args.corpus)

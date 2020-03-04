# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None, ngram_list=None, ss_t=0):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        self.ngrams = None
        self.ss_t = ss_t
        if weights is not None:
            if ngram_list is None:
                wf = np.power(weights, 0.75)
                wf = wf / wf.sum()
                self.weights = FT(wf)
            else:
                self.weights = FT(weights)

        if ngram_list is not None:
            self.ngrams = ngram_list
            self.largest = len(max(ngram_list, key=lambda x: len(x)))
            print('init SGNS with max len: ', self.largest)

    def forward(self, iword, owords):
        """
        - retrieve indices for negative samples at random from the vocabulary, optionally using weights and a rejection threshold
        - forward the current, context and negative words to the embeddings layer
        - return mean of the LogSigmoid-loss for the two independent classifications of positive examples and negative
          samples, evaluating the dot-product via batch matrix multiplication
        todo: logistic loss instead of LogSigmoid; use weights based on frequency

        Parameters
        ----------
        iword : FloatTensor
            minibatch of current words $w_t$
        owords : FloatTensor
            minibatch of context words $w_c$

        Returns
        -------
        FloatTensor
            minibatch of the loss

        """
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        # don't just use iword, get the indices of the n-grams. Problem: not same size anymore, can't minibatch properly
        # solution: fill with zeros, len as for the one with most embeddings.
        if self.ngrams is not None:
            ivectors = t.zeros((batch_size, self.embedding.embedding_size, self.largest))
            for i, w_idx in enumerate(iword):
                i_fw = self.embedding.forward_i(self.ngrams[w_idx])  # tensor of size (num_ngrams, 300)
                ivectors[i, :, :len(self.ngrams[w_idx])] = i_fw.t()
        else:
            ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        #print('ovectors:', ovectors.shape, 'nvectors:', nvectors.shape, 'ivectors:', ivectors.shape)
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log()
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log()
        #print('oloss:', oloss.shape, 'nloss:', nloss.shape)
        if self.ngrams is None:
            return -(oloss.mean(1) + nloss.view(-1, context_size, self.n_negs).sum(2).mean(1)).mean()
        else:
            # what exactly do I want???
            # 1. for each word w_t, t ∈ batch_size:
            #      compute the sum over scores for each context VEC (oloss) / negative word VEC (nloss):
            #        where the score is sum (over all n-gram-vec) of the DOT product between the n-gram-vec and that VEC
            #        -> already have these 59 values.
            # -> and return the mean of the whole thing
            ol_scores = oloss.sum(2).sum(1)
            nl_scores = nloss.sum(2).sum(1)
            return - (ol_scores+nl_scores).mean()

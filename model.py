# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch import logsumexp
from torch.nn.functional import logsigmoid


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

    def __init__(self, embedding, vocab_size, word_freqs, ss_t, n_negs=20, use_weights=False, ngrams_list=None, corresp_ngram=None, loss='sigmoid'):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        self.word_idx2ngram_indices = None
        if loss == 'sigmoid':
            self.loss = logsigmoid
        elif loss == 'logistic':
            self.loss = lambda x: logsumexp(t.stack((t.zeros_like(x), x.neg())), 0)
        self.word_idx2ngram_indices = ngrams_list
        self.word_idx2ngram_idx = LT(corresp_ngram) if ngrams_list else None
        self.word_freqs = word_freqs
        self.discard_probs = np.clip(1 - np.sqrt(ss_t / self.word_freqs), 0, 1)
        self.neg_corpus = None
        self.use_weights = use_weights
        self.use_ngrams = ngrams_list is not None
        self.sample_neg_corpus(True)

    def sample_neg_corpus(self, verbose=False):
        neg_mask = np.random.rand(self.vocab_size) > self.discard_probs
        self.neg_corpus = t.from_numpy(np.arange(self.vocab_size)[neg_mask])
        if self.use_weights:
            weights = self.word_freqs[neg_mask]
            # Raising unigram frequency to power of 3/4 seems to yield best results, subwords paper used sqrt instead
            wf = np.power(weights, 0.75)  # p.sqrt(weights) if self.use_ngrams else
            wf = wf / wf.sum()
            self.weights = t.from_numpy(wf)
        if verbose:
            print('kept', len(self.neg_corpus), 'words of', len(self.word_freqs), 'for subsampling.')

    def _sample_context(self, batch_size, context_size):
        if self.use_weights:
            neg_choice = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            neg_choice = FT(batch_size, context_size * self.n_negs).uniform_(0, self.neg_corpus - 1).long()
        neg_words = self.neg_corpus[neg_choice]
        return self.self.word_idx2ngram_idx[neg_words] if self.use_ngrams else neg_words

    def forward(self, iword, owords):
        """
        - retrieve indices for negative samples at random from the vocabulary, optionally using weights and a rejection threshold
        - forward the current, context and negative words to the embeddings layer
        - return mean of the LogSigmoid-loss for the two independent classifications of positive examples and negative
          samples, evaluating the dot-product via batch matrix multiplication
        NOTE: it's important that the first entry of index 0 is the unknown  character.

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
        nwords = self._sample_context(batch_size, context_size)
        # don't just use iword, get the indices of the n-grams. Problem: not same size anymore, can't minibatch properly
        # solution: fill with zeros, len as for the one with most embeddings.
        if self.use_ngrams:
            max_wordidx = max(iword, key=lambda ii: len(self.word_idx2ngram_indices[ii]))
            curr_largest = len(self.word_idx2ngram_indices[max_wordidx])
            ivector_ngramindices = t.zeros((batch_size, curr_largest), dtype=t.long)
            for i, w_idx in enumerate(iword):
                ivector_ngramindices[i, :len(self.word_idx2ngram_indices[w_idx])] = LT(self.word_idx2ngram_indices[w_idx])
            ivectors = self.embedding.forward_i(ivector_ngramindices).permute(0, 2, 1)
        else:
            ivectors = self.embedding.forward_i(iword).unsqueeze(2)

        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = self.loss(t.bmm(ovectors, ivectors).squeeze())
        nloss = self.loss(t.bmm(nvectors, ivectors).squeeze())

        if self.use_ngrams:
            ol_scores = oloss.unsqueeze(2).sum(2).mean(1)
            nl_scores = nloss.unsqueeze(2).sum(2).mean(1)
            return - (ol_scores + nl_scores).mean()
        else:
            return -(oloss.mean(1) + nloss.view(-1, context_size, self.n_negs).sum(2).mean(1)).mean()


# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch as t
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--ngrams', action='store_true', default=False, help="use ngrams for training")
    return parser.parse_args()


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(args):
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    if args.ngrams:
        word_idx2ngram_idx = pickle.load(open(os.path.join(args.data_dir, 'word_idx2ngram_indices.dat'), 'rb'))
        ngram_idx2ngram = pickle.load(open(os.path.join(args.data_dir, 'ngram_idx2ngram.dat'), 'rb'))
        word_idx2corresp_ngram = pickle.load(open(os.path.join(args.data_dir, 'word_idx2corresp_ngram.dat'), 'rb'))
        vocab_size = len(ngram_idx2ngram)
        print('training with', vocab_size, 'ngrams of', len(idx2word), 'words')
    else:
        vocab_size = len(idx2word)
        word_idx2ngram_idx = None
    if args.weights:
        if args.ngrams:
            ng_counts = pickle.load(open(os.path.join(args.data_dir, 'ngram2ngramCounts.dat'), 'rb'))
            wf = np.array([ng_counts[ngram] for ngram in ngram_idx2ngram])
        else:
            wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
            wf = np.array([wc[word] for word in idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(args.ss_t / wf)
        weights = np.clip(ws, 0, 1)
        print( np.count_nonzero(weights), 'entries of', len(wf), 'ngrams are nonzero with subsampling threshold', args.ss_t)
    else:
        weights = None
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, weights=weights, ngram_list=word_idx2ngram_idx)
    if os.path.isfile(modelpath) and args.conti:
        sgns.load_state_dict(t.load(modelpath))
    if args.cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters())
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and args.conti:
        optim.load_state_dict(t.load(optimpath))
    for epoch in range(1, args.epoch + 1):
        # todo: I want the corpus to be randomized, but sorted by ngram length asc
        dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'))
        # I also need indices of the original words both for i- and owords :)
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
    if args.ngrams:
        # todo: only want the actual words :)
        ngram_idx2vec = model.ivectors.weight.data.cpu().numpy()
        idx2vec = ngram_idx2vec[word_idx2corresp_ngram]
    else:
        idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(args.data_dir, 'idx2vec.dat'), 'wb'))
    t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))


if __name__ == '__main__':
    train(parse_args())

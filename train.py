# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch as t
import numpy as np
import time

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS


def parse_args(lan):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default=f'./data/{lan}/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default=f'./pts/{lan}/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--rand_window', action='store_true', default=False, help="Random window size for each epoch")
    parser.add_argument('--ngrams', action='store_true', default=False, help="use ngrams for training")
    parser.add_argument('--loss', type=str, default='sigmoid', help="specify loss function: sigmoid or logistic")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    return parser.parse_args()


class PermutedSubsampledCorpus(Dataset):
    """
    If weigths are passed, a word is only loaded if the random number (between 0/1) is larger than its weight.
    The each entry in data consists of a word index and the indices of its context words.
    If trained in ngram mode, pass an ngram_list containing the ngram indices for each word;
    Then the corpus is permuted and then sorted by ngram length to speed up training.
    """

    def __init__(self, datapath, ws=None, ngram_list=None, rand_window=False):
        data: list = [pair for pair in pickle.load(open(datapath, 'rb')) if pair[0] != 0]
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data
        if ngram_list is not None:
            random.shuffle(self.data)
            self.data.sort(key=lambda x: len(ngram_list[x[0]]))
        if rand_window:
            window_size = len(self.data[0][1])/2
            clips = np.random.randint(0, window_size, size=len(self.data))
            for i, clip in enumerate(clips):
                if clip > 0:
                    self.data[i][1][:clip] = [0] * clip
                    self.data[i][1][-clip:] = [0] * clip

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(args):
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    vocab_size = len(idx2word)
    word_counts = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    word_freqs = np.array([word_counts[word] for word in idx2word])
    word_freqs = word_freqs / word_freqs.sum()

    if args.ngrams:
        word_idx2ngram_indices = pickle.load(open(os.path.join(args.data_dir, 'word_idx2ngram_indices.dat'), 'rb'))
        ngram_idx2ngram = pickle.load(open(os.path.join(args.data_dir, 'ngram_idx2ngram.dat'), 'rb'))
        word_idx2corresp_ngram = pickle.load(open(os.path.join(args.data_dir, 'word_idx2corresp_ngram.dat'), 'rb'))
        embeds_rows = len(ngram_idx2ngram)
        print('training with', embeds_rows, 'ngrams of', len(idx2word), 'words')
    else:
        embeds_rows = vocab_size
        word_idx2ngram_indices = None
        word_idx2corresp_ngram = None
    if not os.path.isdir(args.save_dir[:6]):
        os.mkdir(args.save_dir[:6])
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    model = Word2Vec(vocab_size=embeds_rows, embedding_size=args.e_dim)
    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))
    sgns = SGNS(model, vocab_size, word_freqs, args.ss_t, n_negs=args.n_negs, use_weights=args.weights,
                ngrams_list=word_idx2ngram_indices, corresp_ngram=word_idx2corresp_ngram)
    if os.path.isfile(modelpath) and args.conti:
        sgns.load_state_dict(t.load(modelpath))
    if args.cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters(), lr=args.lr)
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and args.conti:
        optim.load_state_dict(t.load(optimpath))
    start = time.time()
    for epoch in range(1, args.epoch + 1):
        total_loss = 0
        sgns.sample_neg_corpus()
        dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'), ngram_list=word_idx2ngram_indices,
                                           rand_window=args.rand_window)
        # it's also fine to shuffle here - just saving a few empty multiplications this way
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=word_idx2ngram_indices is None)
        # TOCHECK: is total_batches useful?
        #total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), total=total_loss)
    print('training took', (time.time()-start)//60, 'minutes.')
    if args.ngrams:
        # only want the actual words :)
        ngram_idx2vec = model.ivectors.weight.data.cpu().numpy()
        idx2vec = ngram_idx2vec[word_idx2corresp_ngram]
    else:
        idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(args.data_dir, 'idx2vec.dat'), 'wb'))
    t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))


if __name__ == '__main__':
    lan = 'de'
    train(parse_args(lan))

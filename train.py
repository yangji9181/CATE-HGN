import pickle
from os import path

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

#from data import Dataset
from data import preprocess
from module import Encoder, UniformAggregator, AttentionAggregator, ClusteredAttentionAggregator, UniformCombinator, AttentionCombinator

class Catm_hgn(nn.Module):

    def __init__(self, num_classes, enc):
        super(Catm_hgn, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc.forward(nodes)
        scores = self.weight.mm(embeds.t())
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load():
    if not path.exists("dblp/data.pkl"):
        preprocess()
    with open('dblp/data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def train(
        cuda=False,
        k=10,
        enc_dim=128,
        agg1_dim=64,
        agg2_dim=32,
        final_dim=16,
        max_iter=100,
        batch_size=256,
        trainable=True,
        lr=0.1
        ):
    np.random.seed(1)
    random.seed(1)
    data = load()
    
    paper_feat = nn.Embedding(data['n_papers'], data['n_feat'])
    author_feat = nn.Embedding(data['n_authors'], data['n_feat'])
    venue_feat = nn.Embedding(data['n_venues'], data['n_feat'])
    term_feat = nn.Embedding(data['n_terms'], data['n_feat'])

    # load real feature
    #paper_feat_data = torch.FloatTensor(data['paper_feat'])
    #author_feat_data = torch.FloatTensor(data['author_feat'])
    #venue_feat_data = torch.FloatTensor(data['venue_feat'])
    #term_feat_data = torch.FloatTensor(data['term_feat'])

    # random initialize feature
    paper_feat_data = nn.Parameter(torch.FloatTensor(data['n_papers'], data['n_feat']))
    author_feat_data = nn.Parameter(torch.FloatTensor(data['n_authors'], data['n_feat']))
    venue_feat_data = nn.Parameter(torch.FloatTensor(data['n_venues'], data['n_feat']))
    term_feat_data = nn.Parameter(torch.FloatTensor(data['n_terms'], data['n_feat']))
    init.xavier_uniform_(paper_feat_data)
    init.xavier_uniform_(author_feat_data)
    init.xavier_uniform_(venue_feat_data)
    init.xavier_uniform_(term_feat_data)

    paper_feat.weight = nn.Parameter(paper_feat_data, requires_grad=trainable)
    author_feat.weight = nn.Parameter(author_feat_data, requires_grad=trainable)
    venue_feat.weight = nn.Parameter(venue_feat_data, requires_grad=trainable)
    term_feat.weight = nn.Parameter(term_feat_data, requires_grad=trainable)

    # two-layer self-clustering parameters
    layer1_center = nn.Parameter(torch.FloatTensor(k, enc_dim))
    layer2_center = nn.Parameter(torch.FloatTensor(k, agg1_dim))
    layer1_mask = nn.Parameter(torch.FloatTensor(k, enc_dim))
    layer2_mask = nn.Parameter(torch.FloatTensor(k, agg1_dim))
    init.xavier_uniform_(layer1_center)
    init.xavier_uniform_(layer2_center)
    init.xavier_uniform_(layer1_mask)
    init.xavier_uniform_(layer2_mask)

    if cuda:
        paper_feat.cuda()
        author_feat.cuda()
        venue_feat.cuda()
        term_feat.cuda()

        layer1_center.cuda()
        layer2_center.cuda()
        layer1_mask.cuda()
        layer2_mask.cuda()

    # feature encoding
    #paper_enc = Encoder(paper_feat, data['n_feat'], enc_dim, cuda)
    #author_enc = Encoder(author_feat, data['n_feat'], enc_dim, cuda)
    #venue_enc = Encoder(venue_feat, data['n_feat'], enc_dim, cuda)
    #term_enc = Encoder(term_feat, data['n_feat'], enc_dim, cuda)

    # no feature encoding
    enc_dim = data['n_feat']
    paper_enc = paper_feat
    author_enc = author_feat
    venue_enc = venue_feat
    term_enc = term_feat

    #paper_agg = UniformAggregator(lambda nodes: paper_enc(nodes), lambda nodes: paper_enc(nodes), data['paper_paper_adj'], enc_dim, agg1_dim, cuda, self_loop=True)
    #author_agg = UniformAggregator(lambda nodes: author_enc(nodes), lambda nodes: paper_enc(nodes), data['author_paper_adj'], enc_dim, agg1_dim, cuda)
    #venue_agg = UniformAggregator(lambda nodes: venue_enc(nodes), lambda nodes: paper_enc(nodes), data['venue_paper_adj'], enc_dim, agg1_dim, cuda)
    #term_agg = UniformAggregator(lambda nodes: term_enc(nodes), lambda nodes: paper_enc(nodes), data['term_paper_adj'], enc_dim, agg1_dim, cuda)

    #paper_paper_agg = UniformAggregator(lambda nodes: paper_agg(nodes), lambda nodes: paper_agg(nodes), data['paper_paper_adj'], agg1_dim, agg2_dim, cuda, self_loop=True)
    #paper_author_agg = UniformAggregator(lambda nodes: paper_agg(nodes), lambda nodes: author_agg(nodes), data['paper_author_adj'], agg1_dim, agg2_dim, cuda)
    #paper_venue_agg = UniformAggregator(lambda nodes: paper_agg(nodes), lambda nodes: venue_agg(nodes), data['paper_venue_adj'], agg1_dim, agg2_dim, cuda)
    #paper_term_agg = UniformAggregator(lambda nodes: paper_agg(nodes), lambda nodes: term_agg(nodes), data['paper_term_adj'], agg1_dim, agg2_dim, cuda)

    paper_agg = AttentionAggregator(lambda nodes: paper_enc(nodes), lambda nodes: paper_enc(nodes), data['paper_paper_adj'], enc_dim, agg1_dim, cuda, self_loop=True)
    author_agg = AttentionAggregator(lambda nodes: author_enc(nodes), lambda nodes: paper_enc(nodes), data['author_paper_adj'], enc_dim, agg1_dim, cuda)
    venue_agg = AttentionAggregator(lambda nodes: venue_enc(nodes), lambda nodes: paper_enc(nodes), data['venue_paper_adj'], enc_dim, agg1_dim, cuda)
    term_agg = AttentionAggregator(lambda nodes: term_enc(nodes), lambda nodes: paper_enc(nodes), data['term_paper_adj'], enc_dim, agg1_dim, cuda)

    paper_paper_agg = AttentionAggregator(lambda nodes: paper_agg(nodes), lambda nodes: paper_agg(nodes), data['paper_paper_adj'], agg1_dim, agg2_dim, cuda, self_loop=True)
    paper_author_agg = AttentionAggregator(lambda nodes: paper_agg(nodes), lambda nodes: author_agg(nodes), data['paper_author_adj'], agg1_dim, agg2_dim, cuda)
    paper_venue_agg = AttentionAggregator(lambda nodes: paper_agg(nodes), lambda nodes: venue_agg(nodes), data['paper_venue_adj'], agg1_dim, agg2_dim, cuda)
    paper_term_agg = AttentionAggregator(lambda nodes: paper_agg(nodes), lambda nodes: term_agg(nodes), data['paper_term_adj'], agg1_dim, agg2_dim, cuda)

    #paper_agg = ClusteredAttentionAggregator(layer1_center, layer1_mask, lambda nodes: paper_enc(nodes), lambda nodes: paper_enc(nodes), data['paper_paper_adj'], enc_dim, agg1_dim, cuda, self_loop=True)
    #author_agg = ClusteredAttentionAggregator(layer1_center, layer1_mask, lambda nodes: author_enc(nodes), lambda nodes: paper_enc(nodes), data['author_paper_adj'], enc_dim, agg1_dim, cuda)
    #venue_agg = ClusteredAttentionAggregator(layer1_center, layer1_mask, lambda nodes: venue_enc(nodes), lambda nodes: paper_enc(nodes), data['venue_paper_adj'], enc_dim, agg1_dim, cuda)
    #term_agg = ClusteredAttentionAggregator(layer1_center, layer1_mask, lambda nodes: term_enc(nodes), lambda nodes: paper_enc(nodes), data['term_paper_adj'], enc_dim, agg1_dim, cuda)

    #paper_paper_agg = ClusteredAttentionAggregator(layer2_center, layer2_mask, lambda nodes: paper_agg(nodes), lambda nodes: paper_agg(nodes), data['paper_paper_adj'], agg1_dim, agg2_dim, cuda, self_loop=True)
    #paper_author_agg = ClusteredAttentionAggregator(layer2_center, layer2_mask, lambda nodes: paper_agg(nodes), lambda nodes: author_agg(nodes), data['paper_author_adj'], agg1_dim, agg2_dim, cuda)
    #paper_venue_agg = ClusteredAttentionAggregator(layer2_center, layer2_mask, lambda nodes: paper_agg(nodes), lambda nodes: venue_agg(nodes), data['paper_venue_adj'], agg1_dim, agg2_dim, cuda)
    #paper_term_agg = ClusteredAttentionAggregator(layer2_center, layer2_mask, lambda nodes: paper_agg(nodes), lambda nodes: term_agg(nodes), data['paper_term_adj'], agg1_dim, agg2_dim, cuda)

    paper_final = AttentionCombinator(lambda nodes: paper_paper_agg(nodes), lambda nodes: paper_author_agg(nodes), lambda nodes: paper_venue_agg(nodes), lambda nodes: paper_term_agg(nodes), agg2_dim, final_dim, cuda)

    #TODO: implement term mining
    catm_hgn = Catm_hgn(2, paper_final)
    if cuda:
        catm_hgn.cuda()

    test = data['test']
    val = data['val']
    train = list(data['train'])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, catm_hgn.parameters()), lr=lr)
    times = []
    for batch in range (max_iter):
        batch_nodes = train[:batch_size]
        random.shuffle(train)
        #val = np.array(batch_nodes, dtype=np.dtype(int))
        start_time = time.time()
        optimizer.zero_grad()
        loss = catm_hgn.loss(batch_nodes, Variable(torch.LongTensor(data['paper_label'][np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print('batch %d, loss: %f' % (batch, loss.item()))

        val_output = catm_hgn.forward(val)
        print('validation F1: %f' % f1_score(data['paper_label'][val], val_output.data.numpy().argmax(axis=1), average='micro'))
    print('average batch time: %f' % np.mean(times))


if __name__ == "__main__":
    train()
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, feat, feat_dim, embed_dim, cuda):
        super(Encoder, self).__init__()

        self.feat = feat
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        index = torch.LongTensor(nodes)
        if self.cuda:
            index = index.cuda()
        embed = F.relu(self.weight.mm(self.feat(index).t()))
        return embed.t()

class UniformAggregator(nn.Module):
    def __init__(self, self_feat, neigh_feat, adj_lists, feat_dim, embed_dim, cuda, self_loop=False, num_sample=10):
        super(UniformAggregator, self).__init__()

        self.self_feat = self_feat
        self.neigh_feat = neigh_feat
        self.adj_lists = adj_lists
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.self_loop = self_loop
        self.num_sample = num_sample

        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, feat_dim*2))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        if self.self_loop:
            to_neighs = [self.adj_lists[int(node)] | set([int(node)]) for node in nodes]
        else:
            to_neighs = [self.adj_lists[int(node)] for node in nodes]
        _set = set
        _sample = random.sample
        sample_neighs = [_set(_sample(to_neigh, self.num_sample)) if len(to_neigh) >= self.num_sample else to_neigh for to_neigh in to_neighs]
        unique_nodes_list = list(set.union(*sample_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(sample_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for sample_neigh in sample_neighs for n in sample_neigh]   
        row_indices = [i for i in range(len(sample_neighs)) for j in range(len(sample_neighs[i]))]
        mask[row_indices, column_indices] = 1
        index = torch.LongTensor(unique_nodes_list)
        if self.cuda:
            mask = mask.cuda()
            index = index.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        unique_node_feat = self.neigh_feat(index)
        neigh_feats = mask.mm(unique_node_feat)

        index = torch.LongTensor(nodes)
        if self.cuda:
            index = index.cuda()
        self_feats = self.self_feat(index)
        combined_feats = torch.cat([self_feats, neigh_feats], dim=1)
        embed = F.relu(self.weight.mm(combined_feats.t()))
        return embed.t()

class AttentionAggregator(nn.Module):
    def __init__(self, self_feat, neigh_feat, adj_lists, feat_dim, embed_dim, cuda, self_loop=False, num_sample=10):
        super(AttentionAggregator, self).__init__()

        self.self_feat = self_feat
        self.neigh_feat = neigh_feat
        self.adj_lists = adj_lists
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.self_loop = self_loop
        self.num_sample = num_sample

        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, feat_dim*2))
        init.xavier_uniform_(self.weight)

        self.alpha = nn.Parameter(torch.FloatTensor(feat_dim*2, 1))
        init.xavier_uniform_(self.alpha)

    def forward(self, nodes):
        if self.self_loop:
            to_neighs = [self.adj_lists[int(node)] | set([int(node)]) for node in nodes]
        else:
            to_neighs = [self.adj_lists[int(node)] for node in nodes]
        _set = set
        _sample = random.sample
        sample_neighs = [_set(_sample(to_neigh, self.num_sample)) if len(to_neigh) >= self.num_sample else to_neigh for to_neigh in to_neighs]
        unique_nodes_list = list(set.union(*sample_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(sample_neighs), self.num_sample, len(unique_nodes)))     
        column_indices = [unique_nodes[n] for sample_neigh in sample_neighs for n in sample_neigh]
        row_indices = [i for i in range(len(sample_neighs)) for j in range(len(sample_neighs[i]))]
        neigh_indices = [j for i in range(len(sample_neighs)) for j in range(len(sample_neighs[i]))]
        mask[row_indices, neigh_indices, column_indices] = 1
        unique_index = torch.LongTensor(unique_nodes_list)
        node_index = torch.LongTensor(nodes)
        if self.cuda:
            mask = mask.cuda()
            unique_index = unique_index.cuda()
            node_index = node_index.cuda()
        unique_node_feat = self.neigh_feat(unique_index)
        self_feat = self.self_feat(node_index)
        neigh_feat_all = mask.matmul(unique_node_feat)
        self_feat_all = self_feat.unsqueeze(-1).expand(len(nodes), self.feat_dim, self.num_sample).transpose(1, 2)
        att_weight = torch.exp(F.relu(torch.cat([self_feat_all, neigh_feat_all], dim=2).matmul(self.alpha).squeeze()))
        att_weight_sum = att_weight.sum(dim=1, keepdim=True)
        att_weight_norm = att_weight.div(att_weight_sum)
        
        neigh_feat = neigh_feat_all.mul(att_weight_norm.unsqueeze(-1).expand(len(nodes), self.num_sample, self.feat_dim)).sum(dim=1).squeeze()
        combined_feat = torch.cat([self_feat, neigh_feat], dim=1)
        embed = F.relu(self.weight.mm(combined_feat.t()))

        return embed.t()

class ClusteredAttentionAggregator(nn.Module):
    def __init__(self, center, mask, self_feat, neigh_feat, adj_lists, feat_dim, embed_dim, cuda, self_loop=False, num_sample=10):
        super(ClusteredAttentionAggregator, self).__init__()

        self.center = center
        self.mask = mask
        self.self_feat = self_feat
        self.neigh_feat = neigh_feat
        self.adj_lists = adj_lists
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.self_loop = self_loop
        self.num_sample = num_sample

        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, feat_dim*2))
        init.xavier_uniform_(self.weight)

        self.alpha = nn.Parameter(torch.FloatTensor(feat_dim*2, 1))
        init.xavier_uniform_(self.alpha)

    def forward(self, nodes):
        if self.self_loop:
            to_neighs = [self.adj_lists[int(node)] | set([int(node)]) for node in nodes]
        else:
            to_neighs = [self.adj_lists[int(node)] for node in nodes]
        _set = set
        _sample = random.sample
        sample_neighs = [_set(_sample(to_neigh, self.num_sample)) if len(to_neigh) >= self.num_sample else to_neigh for to_neigh in to_neighs]
        unique_nodes_list = list(set.union(*sample_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(sample_neighs), self.num_sample, len(unique_nodes)))     
        column_indices = [unique_nodes[n] for sample_neigh in sample_neighs for n in sample_neigh]
        row_indices = [i for i in range(len(sample_neighs)) for j in range(len(sample_neighs[i]))]
        neigh_indices = [j for i in range(len(sample_neighs)) for j in range(len(sample_neighs[i]))]
        mask[row_indices, neigh_indices, column_indices] = 1
        unique_index = torch.LongTensor(unique_nodes_list)
        node_index = torch.LongTensor(nodes)
        if self.cuda:
            mask = mask.cuda()
            unique_index = unique_index.cuda()
            node_index = node_index.cuda()
        unique_node_feat = self.neigh_feat(unique_index)
        self_feat = self.self_feat(node_index)
        neigh_feat_all = mask.matmul(unique_node_feat)
        self_feat_all = self_feat.unsqueeze(-1).expand(len(nodes), self.feat_dim, self.num_sample).transpose(1, 2)
        att_weight = torch.exp(F.relu(torch.cat([self_feat_all, neigh_feat_all], dim=2).matmul(self.alpha).squeeze()))
        att_weight_sum = att_weight.sum(dim=1, keepdim=True)
        att_weight_norm = att_weight.div(att_weight_sum)

        k = self.mask.shape[0]
        q = torch.pow(torch.pow(neigh_feat_all.unsqueeze(2).expand(len(nodes), self.num_sample, k, self.feat_dim) - self.center.unsqueeze(0).unsqueeze(2).expand(len(nodes), k, self.num_sample, self.feat_dim), 2).sum(dim=3)+1, -1)
        q_sum = q.sum(dim=2, keepdim=True)
        q_norm = q.div(q_sum)
        neigh_feat_all_clustered = (neigh_feat_all.unsqueeze(2).expand(len(nodes), self.num_sample, k, self.feat_dim) * q.unsqueeze(3).expand(len(nodes), self.num_sample, k, self.feat_dim) * self.mask.unsqueeze(0).unsqueeze(2).expand(len(nodes), k, self.num_sample, self.feat_dim)).sum(2).squeeze()

        neigh_feat = neigh_feat_all_clustered.mul(att_weight_norm.unsqueeze(-1).expand(len(nodes), self.num_sample, self.feat_dim)).sum(dim=1).squeeze()
        combined_feat = torch.cat([self_feat, neigh_feat], dim=1)
        embed = F.relu(self.weight.mm(combined_feat.t()))

        return embed.t()

class UniformCombinator(nn.Module):
    def __init__(self, feat1, feat2, feat3, feat4, feat_dim, embed_dim, cuda):
        super(UniformCombinator, self).__init__()

        self.feat1 = feat1
        self.feat2 = feat2
        self.feat3 = feat3
        self.feat4 = feat4
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.cuda = cuda

        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, feat_dim*4))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        index = torch.LongTensor(nodes)
        if self.cuda:
            index = index.cuda()
        combined_feats = torch.cat([self.feat1(index), self.feat2(index), self.feat3(index), self.feat4(index)], dim=1)
        embed = F.relu(self.weight.mm(combined_feats.t()))
        return embed.t()


class AttentionCombinator(nn.Module):
    def __init__(self, feat1, feat2, feat3, feat4, feat_dim, embed_dim, cuda):
        super(AttentionCombinator, self).__init__()

        self.feat1 = feat1
        self.feat2 = feat2
        self.feat3 = feat3
        self.feat4 = feat4
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.cuda = cuda

        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, feat_dim*4))
        init.xavier_uniform_(self.weight)

        self.alpha = nn.Parameter(torch.FloatTensor(feat_dim, 1))
        init.xavier_uniform_(self.alpha)

    def forward(self, nodes):
        index = torch.LongTensor(nodes)
        if self.cuda:
            index = index.cuda()
        feats = torch.stack([self.feat1(index), self.feat2(index), self.feat3(index), self.feat4(index)], dim=1)
        att_weight = torch.exp(F.relu(feats.matmul(self.alpha).squeeze()))
        att_weight_sum = att_weight.sum(dim=1, keepdim=True)
        att_weight_norm = att_weight.div(att_weight_sum)
        combined_feats = feats.mul(att_weight.unsqueeze(-1).expand(len(nodes), 4, self.feat_dim)).view(len(nodes), -1).contiguous()
        embed = F.relu(self.weight.mm(combined_feats.t()))
        return embed.t()

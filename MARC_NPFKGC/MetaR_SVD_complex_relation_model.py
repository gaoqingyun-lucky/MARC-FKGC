from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import kl_divergence

from flow import Flow
from relational_path_gnn import RelationalPathGNN
# from embedding import *


class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1, dropout=0.5):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        self.lstm = nn.LSTM(self.embed_size * 2, self.n_hidden, self.layers, bidirectional=True, dropout=self.dropout)
        # self.gru = nn.GRU(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True)
        self.out = nn.Linear(self.n_hidden * 2 * self.layers, self.out_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2).cuda()
        # batchnorm = nn.BatchNorm1d(5, affine=False).cuda()
        # attn_weight = batchnorm(attn_weight)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weight)
        context = context.view(-1, self.n_hidden * 2 * self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(self.layers * 2, size[0], self.n_hidden)).cuda()
        cell_state = Variable(torch.zeros(self.layers * 2, size[0], self.n_hidden)).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))  # LSTM
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_cell_state)  # change log

        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)



class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)
class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


def save_grad(grad):
    global grad_norm
    grad_norm = grad


class NPFKGC(nn.Module):
    def __init__(self, g, dataset, parameter, num_symbols, embed=None):
        super(NPFKGC, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.rel2id = dataset['rel2id']
        self.num_rel = len(self.rel2id)
        self.few = parameter['few']
        self.dropout = nn.Dropout(0.5)
        self.num_hidden1 = 500
        self.num_hidden2 = 200
        self.lstm_dim = parameter['lstm_hiddendim']
        self.lstm_layer = parameter['lstm_layers']
        self.np_flow = parameter['flow']
        self.latent_num = parameter['latent_num']


        self.r_path_gnn = RelationalPathGNN(g, dataset['ent2id'], len(dataset['rel2emb']), parameter)

        if parameter['dataset'] == 'Wiki-One':
            self.r_dim = self.z_dim = 50
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=50, num_hidden1=250,
                                                        num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=50, num_hidden1=250,
                                                num_hidden2=100, r_dim=self.z_dim, dropout_p=self.dropout_p)
            # self.embedding_learner = EmbeddingLearner(self.embed_dim, self.z_dim, self.embed_dim)
            self.relation_latent_encoder = Relation_Latent_Encoder(r_dim=self.embed_dim, num_heads=2, latent_num=self.embed_dim)
            self.complex_relation_attention = Complex_Relation_Attention(self.r_dim, num_heads=2)

        elif parameter['dataset'] == 'NELL-One':
            self.r_dim = self.z_dim = 100
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=100, num_hidden1=500,
                                                num_hidden2=200, r_dim=self.z_dim, dropout_p=self.dropout_p)
            # self.embedding_learner = EmbeddingLearner(self.embed_dim, self.z_dim, self.embed_dim)
            self.relation_latent_encoder = Relation_Latent_Encoder(r_dim=self.embed_dim, num_heads=4, latent_num=self.embed_dim)
            self.complex_relation_attention = Complex_Relation_Attention(self.r_dim, num_heads=4)
        elif parameter['dataset'] == 'FB15K-One':
            self.r_dim = self.z_dim = 100
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=100, num_hidden1=500,
                                                num_hidden2=200, r_dim=self.z_dim, dropout_p=self.dropout_p)
            # self.embedding_learner = EmbeddingLearner(self.embed_dim, self.z_dim, self.embed_dim)
            self.relation_latent_encoder = Relation_Latent_Encoder(r_dim = self.embed_dim, num_heads = 4, latent_num = self.embed_dim)
            self.complex_relation_attention = Complex_Relation_Attention(self.r_dim, num_heads=4)
        if self.np_flow != 'none':
            self.flows = Flow(self.z_dim, parameter['flow'], parameter['K'])

        self.xy_to_mu_sigma = MuSigmaEncoder(self.r_dim, self.z_dim)

        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.embedding_learner = EmbeddingLearner()

    def eval_reset(self):
        self.eval_query = None
        self.eval_z = None
        self.eval_rel = None
        self.is_reset = True

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def eval_support(self, support, support_negative, query):
        support, support_negative, query = self.r_path_gnn(support), self.r_path_gnn(support_negative), self.r_path_gnn(
            query)
        support_few = support.view(support.shape[0], self.few, 2, self.embed_dim)
        support_pos_r = self.latent_encoder(support, 1)
        support_neg_r = self.latent_encoder(support_negative, 0)
        target_r = torch.cat([support_pos_r, support_neg_r], dim=1)
        target_dist = self.xy_to_mu_sigma(target_r)
        latent_vector = self.relation_latent_encoder(support_pos_r)
        Q, R = torch.linalg.qr(latent_vector)
        Or_latent_vector = Q[:self.latent_num, :].unsqueeze(0).expand(support_pos_r.shape[0], -1, -1)
        latent_vector_support = self.complex_relation_attention(Or_latent_vector, support_pos_r)
        z = target_dist.sample()
        if self.np_flow != 'none':
            z, _ = self.flows(z, target_dist)
        rel = self.relation_learner(support_few) + latent_vector_support.unsqueeze(1).unsqueeze(2)
        return query, z, rel

    def eval_forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        support, support_negative, query, negative = task
        negative = self.r_path_gnn(negative)
        if self.is_reset:
            query, z, rel = self.eval_support(support, support_negative, query)
            self.eval_query = query
            self.eval_z = z
            self.eval_rel = rel
            self.is_reset = False
        else:
            query = self.eval_query
            z = self.eval_z
            rel = self.eval_rel
        num_q = query.shape[1]  # num of query
        num_n = negative.shape[1]  # num of query negative
        rel_q = rel.expand(-1, num_q + num_n, -1, -1)
        z_q = z.unsqueeze(1).expand(-1, num_q + num_n, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, z_q)
        return p_score, n_score

    def forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        latent_orthogonal_loss = 0
        # latent_l2_loss = 0
        # transfer task string into embedding
        support, support_negative, query, negative = [self.r_path_gnn(t) for t in task]
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        support_few = support.view(support.shape[0], self.few, 2, self.embed_dim)

        # Encoder
        if iseval or istest:
            support_pos_r = self.latent_encoder(support, 1)
            support_neg_r = self.latent_encoder(support_negative, 0)
            target_r = torch.cat([support_pos_r, support_neg_r], dim=1)
            target_dist = self.xy_to_mu_sigma(target_r)
            latent_vector = self.relation_latent_encoder(support_pos_r)
            Q, R = torch.linalg.qr(latent_vector)
            Or_latent_vector  = Q[:self.latent_num,:].unsqueeze(0).expand(support_pos_r.shape[0], -1, -1)
            latent_vector_support = self.complex_relation_attention(Or_latent_vector, support_pos_r)
            # z = target_dist.sample()
            # if self.np_flow != 'none':
            #     z, _ = self.flows(z, target_dist)
        else:
            query_pos_r = self.latent_encoder(query, 1)
            query_neg_r = self.latent_encoder(negative, 0)
            support_pos_r = self.latent_encoder(support, 1)
            support_neg_r = self.latent_encoder(support_negative, 0)
            context_r = torch.cat([support_pos_r, support_neg_r], dim=1)
            target_r = torch.cat([support_pos_r, support_neg_r, query_pos_r, query_neg_r], dim=1)
            context_dist = self.xy_to_mu_sigma(context_r)
            target_dist = self.xy_to_mu_sigma(target_r)
            latent_vector = self.relation_latent_encoder(support_pos_r)
            Q, R = torch.linalg.qr(latent_vector)
            Or_latent_vector  = Q[:self.latent_num,:].unsqueeze(0).expand(support_pos_r.shape[0], -1, -1)
            ZZT = torch.mm(Q[:self.latent_num,:], Q[:self.latent_num,:].T)  # 潜在变量Z*Z^T
            n = ZZT.shape[0]
            I = torch.eye(n, device=ZZT.device)
            latent_orthogonal_loss = torch.norm(ZZT - I, p='fro') ** 2  # Frobenius 范数的平方
            # latent_l2_loss = torch.norm(Q[:self.latent_num,:], p=2)
            latent_vector_support = self.complex_relation_attention(Or_latent_vector, support_pos_r)

            # z = target_dist.rsample()
            # if self.np_flow != 'none':
            #     z, kld = self.flows(z, target_dist, context_dist)
            # else:
            #     kld = kl_divergence(target_dist, context_dist).sum(-1)

        # rel = self.relation_learner(support_few) + latent_vector_support.unsqueeze(1).unsqueeze(2)
        rel = self.relation_learner(support) + latent_vector_support.unsqueeze(1).unsqueeze(2)
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)

                y = torch.Tensor([1]).to(self.device)
                y = y.unsqueeze(0).expand(p_score.shape)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)


        return p_score, n_score


class LatentEncoder(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, r_dim=100, dropout_p=0.5):
        super(LatentEncoder, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(2 * embed_size + 1, num_hidden1)),
            # ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden1, num_hidden2)),
            # ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, r_dim)),
            # ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs, y):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)  # (B, few, dim * 2)
        if y == 1:
            label = torch.ones(size[0], size[1], 1).to(inputs)
        else:
            label = torch.zeros(size[0], size[1], 1).to(inputs)
        x = torch.cat([x, label], dim=-1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)

        return x  # (B, few, r_dim)


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def aggregate(self, r):
        return torch.mean(r, dim=1)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, few, r_dim)
        """
        r = self.aggregate(r)
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return torch.distributions.Normal(mu, sigma)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        residual = x
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class Relation_Latent_Encoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, num_heads, latent_num, qkv_bias=True, drop=0., norm_layer=nn.LayerNorm,
                 drop_path=0.):
        super(Relation_Latent_Encoder, self).__init__()

        self.r_dim = r_dim
        self.latent_num = latent_num
        # self.r_to_hidden = nn.Linear(r_dim, r_dim)
        # self.hidden_to_mu = nn.Linear(r_dim, r_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, r_dim)
        self.attn = Attention(r_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0.2, proj_drop=drop)
        self.norm1 = norm_layer(r_dim)
        # self.norm2 = norm_layer(r_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def aggregate(self, r):
        return torch.mean(r, dim=1)

    # 施加正交性（使用 Gram-Schmidt 正交化方法）
    def gram_schmidt(self,vectors):
        # 对输入的 vectors 进行正交化处理
        orthogonal_vectors = []
        for v in vectors:
            for u in orthogonal_vectors:
                v = v - torch.matmul(v, u) / torch.matmul(u, u) * u
            orthogonal_vectors.append(v)
        return torch.stack(orthogonal_vectors)


    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, few, r_dim)
        """
        # r = self.aggregate(r).unsqueeze(1)
        r, _ = self.attn(self.norm1(r))
        r = r + self.drop_path(r)
        # r = self.drop_path(self.mlp(self.norm2(r)))
        r = self.aggregate(r)
        mu = torch.mean(r, dim=0, keepdim=True)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(mu))
        latent_normal_dist = torch.distributions.Normal(mu, sigma)
        latent_vector = latent_normal_dist.sample((self.latent_num,)).squeeze(1)
        # latent_vector = self.gram_schmidt(latent_vector)
        return latent_vector

class Complex_Relation_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads = 4, norm_layer=nn.LayerNorm):
        super(Complex_Relation_Attention, self).__init__()

        # embed_dim 是每个头的维度
        # num_heads 是头的数量
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # 确保 embed_dim 可以被 num_heads 整除
        assert embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # 每个头的维度
        self.head_dim = embed_dim // self.num_heads
        self.norm1 = norm_layer(embed_dim)

        # 线性变换层，用于 Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.scale = embed_dim ** -0.5
        # self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 输出层
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, Q, K, mask=None):
        batch_size = Q.size(0)
        B, N, C = Q.shape

        # Q, K, V 的形状是 (batch_size, seq_len, embed_dim)
        # 将 Q, K, V 进行线性变换
        Q = self.q_linear(Q)  # (batch_size, seq_len_q, embed_dim)
        K = self.k_linear(K)  # (batch_size, seq_len_k, embed_dim)
        V = K  # (batch_size, seq_len_v, embed_dim)

        # 分成多个头
        Q = Q.view(B, -1, self.num_heads, self.head_dim)  # (batch_size, seq_len_q, num_heads, head_dim)
        K = K.view(B, -1, self.num_heads, self.head_dim)  # (batch_size, seq_len_k, num_heads, head_dim)
        V = V.view(B, -1, self.num_heads, self.head_dim)  # (batch_size, seq_len_v, num_heads, head_dim)

        # 转置，使得 head 维度是第一个维度
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len_v, head_dim)

        attn = (Q @ K.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ V).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return torch.mean(x, dim=1)

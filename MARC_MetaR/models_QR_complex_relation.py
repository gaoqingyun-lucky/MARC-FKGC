from embedding import *
from collections import OrderedDict
import torch


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

# ---------------------------------------add------------------------------------------
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
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
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

        self.num_heads = num_heads
        self.embed_dim = embed_dim

        assert embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // self.num_heads
        self.norm1 = norm_layer(embed_dim)

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.scale = embed_dim ** -0.5

        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, Q, K, mask=None):
        batch_size = Q.size(0)
        B, N, C = Q.shape
        Q = self.q_linear(Q)  # (batch_size, seq_len_q, embed_dim)
        K = self.k_linear(K)  # (batch_size, seq_len_k, embed_dim)
        V = K  # (batch_size, seq_len_v, embed_dim)

        Q = Q.view(B, -1, self.num_heads, self.head_dim)  # (batch_size, seq_len_q, num_heads, head_dim)
        K = K.view(B, -1, self.num_heads, self.head_dim)  # (batch_size, seq_len_k, num_heads, head_dim)
        V = V.view(B, -1, self.num_heads, self.head_dim)  # (batch_size, seq_len_v, num_heads, head_dim)

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


# ------------------------------------------------------------------------------------


class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.embedding = Embedding(dataset, parameter)
        self.latent_num = parameter['latent_num']

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=50, num_hidden1=250,
                                                        num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=self.embed_dim, num_hidden1=500,
                                                num_hidden2=self.embed_dim * 2, r_dim=self.embed_dim, dropout_p=self.dropout_p)
            self.relation_latent_encoder = Relation_Latent_Encoder(r_dim=self.embed_dim, num_heads=2, latent_num=self.embed_dim)
            self.complex_relation_attention = Complex_Relation_Attention(self.embed_dim, num_heads = 2)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=self.embed_dim, num_hidden1=500,
                                                num_hidden2=self.embed_dim * 2, r_dim=self.embed_dim, dropout_p=self.dropout_p)
            self.relation_latent_encoder = Relation_Latent_Encoder(r_dim=self.embed_dim, num_heads=4, latent_num=self.embed_dim)
            self.complex_relation_attention = Complex_Relation_Attention(self.embed_dim, num_heads = 4)
        elif parameter['dataset'] == 'FB15K':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=self.embed_dim, num_hidden1=500,
                                                num_hidden2=self.embed_dim * 2, r_dim=self.embed_dim, dropout_p=self.dropout_p)
            self.relation_latent_encoder = Relation_Latent_Encoder(r_dim=self.embed_dim, num_heads=4, latent_num=self.embed_dim)
            self.complex_relation_attention = Complex_Relation_Attention(self.embed_dim, num_heads = 4)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]

        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        # -----------------------------add------------------------------
        support_pos_r = self.latent_encoder(support, 1)
        # latent_vector = self.relation_latent_encoder(support_pos_r).unsqueeze(0).expand(support_pos_r.shape[0], -1, -1)
        latent_vector = self.relation_latent_encoder(support_pos_r)
        Q, R = torch.linalg.qr(latent_vector)
        Or_latent_vector = Q[:self.latent_num, :].unsqueeze(0).expand(support_pos_r.shape[0], -1, -1)
        ZZT = torch.mm(Q[:self.latent_num, :], Q[:self.latent_num, :].T)  # 潜在变量Z*Z^T
        n = ZZT.shape[0]
        I = torch.eye(n, device=ZZT.device)
        latent_orthogonal_loss = torch.norm(ZZT - I, p='fro') ** 2  # Frobenius 范数的平方
        latent_vector_support = self.complex_relation_attention(Or_latent_vector, support_pos_r)
        # --------------------------------------------------------------

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

        return p_score, n_score, latent_orthogonal_loss


import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.distributions import Independent

import copy


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    # embd_pdrop = 0.1
    # resid_pdrop = 0.1
    # attn_pdrop = 0.1
    embd_pdrop = 0.
    resid_pdrop = 0.
    attn_pdrop = 0.

    def __init__(self, state_size, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.state_size = state_size
        for k, v in kwargs.items():
            setattr(self, k, v)

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class G_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., type='layernorm'):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.normal_type = type

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.eps = 1e-5

        self.register_norm(self.eps)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        seq_len = q.size(-2)
        if self.normal_type == 'instance':
            k, v = k.transpose(-2, -1), v.transpose(-2, -1)

        k = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_K, (k[:, i, ...] for i in range(self.heads)))], dim=1)
        v = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_V, (v[:, i, ...] for i in range(self.heads)))], dim=1)

        if self.normal_type == 'instance':
            k, v = k.transpose(-2, -1), v.transpose(-2, -1)

        scores = torch.matmul(k.transpose(-2, -1), v) / seq_len

        out = torch.matmul(q, scores)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def register_norm(self, eps):
        if self.normal_type == 'instance':
            self.norm_K = self._get_instancenorm(self.dim_head, self.heads,
                                                 eps=eps,
                                                 affine=True)
            self.norm_V = self._get_instancenorm(self.dim_head, self.heads,
                                                 eps=eps,
                                                 affine=True)
        else:
            self.norm_K = self._get_layernorm(self.dim_head, self.heads,
                                              eps=eps)
            self.norm_V = self._get_layernorm(self.dim_head, self.heads,
                                              eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., att_type='ScaleDot'):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.att_type = att_type
        if self.att_type == 'ScaleDot':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))
        elif self.att_type == 'Galerkin':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    G_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    G_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MaskAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., max_block =256):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.register_buffer("mask", rearrange(torch.tril(torch.ones(max_block+1, max_block+1)), 'a b -> 1 1 a b'))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # batch timestep context_dim
        B, T, C = x.size()
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # batch * time * head * dim
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # TODO: Check the mask implementation
        # TODO:block should > 3* n_agent + 1
        dots = dots.masked_fill(self.mask[:,:, :T, :T] == 0, float('-inf'))
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Masked_G_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., type='layernorm'):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.normal_type = type

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.eps = 1e-7

        self.register_norm(self.eps)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        bsz, n_head, seq_len, d_k = q.shape
        dtype = q.dtype

        # k = k / seq_len

        if self.normal_type == 'instance':
            k, v = k.transpose(-2, -1), v.transpose(-2, -1)

        k = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_K, (k[:, i, ...] for i in range(self.heads)))], dim=1)
        v = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_V, (v[:, i, ...] for i in range(self.heads)))], dim=1)

        if self.normal_type == 'instance':
            k, v = k.transpose(-2, -1), v.transpose(-2, -1)

        b_q, b_k, b_v = [x.reshape(bsz, n_head, -1, 1, d_k) for x in (q, k, v)]

        b_k_sum = b_k.sum(dim=-2)
        b_k_cumsum = b_k_sum.cumsum(dim=-2).type(dtype)

        scores = torch.einsum('bhund,bhune->bhude', b_k, b_v)
        scores = scores.cumsum(dim=-3).type(dtype) / seq_len


        D_inv = 1. / torch.einsum('bhud,bhund->bhun', b_k_cumsum + self.eps, b_q)
        out = torch.einsum('bhund,bhude,bhun->bhune', b_q, scores, D_inv)

        # out = out.reshape(q.shape)
        # out = rearrange(out, 'b h n d -> b n (h d)')

        out = rearrange(out, 'b h u n e -> b u (h e n)')


        return self.to_out(out)

    def register_norm(self, eps):
        if self.normal_type == 'instance':
            self.norm_K = self._get_instancenorm(self.dim_head, self.heads,
                                                 eps=eps,
                                                 affine=True)
            self.norm_V = self._get_instancenorm(self.dim_head, self.heads,
                                                 eps=eps,
                                                 affine=True)
        else:
            self.norm_K = self._get_layernorm(self.dim_head, self.heads,
                                              eps=eps)
            self.norm_V = self._get_layernorm(self.dim_head, self.heads,
                                              eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])

class maskedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., att_type='ScaleDot'):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.att_type = att_type
        if self.att_type == 'ScaleDot':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, MaskAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))
        elif self.att_type == 'Galerkin':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Masked_G_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Masked_G_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# class maskedTransformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, MaskAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x

class LinearDecoder(nn.Module):
    def __init__(self, in_dim, med_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, med_dim),
            Swish(),
            nn.Linear(med_dim, out_dim)
        )

    def forward(self, in_feature):
        out_feature = self.net(in_feature)

        return out_feature


class TransformerQAgent(nn.Module):
    def __init__(
        self,
        config,
        decoder_dim_head = 64
    ):
        super().__init__()
        # construct encoder
        self.config = config
        self.n_agent = config.context_len # self.n_agent means the max input block
        self.n_ensemble = config.n_ensemble
        self.action_dim = config.action_space
        self.obs_dim = config.obs_space
        self.device = config.device
        # self.action_max = config.action_max
        # self.action_max.requires_grad = True
        self.sigma_min = -5
        self.sigma_max = 0.5

        self.time_step = self.n_agent
        self.device = config.device
        self.mode = config.mode

        self.embed_dim = config.n_embed
        self.encoder_head = config.n_head
        self.encoder_dim = config.n_embed
        self.encoder_layer = config.n_layer
        #self.mode = config.mode #'oa' 'oar' 'oaro'
        self.encoder = Transformer(self.encoder_dim,
                                   self.encoder_layer,
                                   self.encoder_head,
                                   decoder_dim_head,
                                   self.encoder_dim * 4) # TODO:dropout, dim_head

        # token [TODO: all the same or different?]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        # self.sigma_param = nn.Parameter(torch.ones(self.action_dim)*-2)#.to(self.device)
        self.register_parameter(name='sigma_param', param=nn.Parameter(torch.ones(self.action_dim)*-2))
        # self.register_parameter(name='sigma_param', param=nn.Parameter(torch.tensor([-4.0, -4.0])))
        # self.sigma_param = nn.Parameter(torch.tensor([-1.0, -1.0]))  # .to(self.device)

        # position embeding
        self.obs_pos_embedding = nn.Parameter(torch.randn(1, self.time_step, self.embed_dim))
        self.act_pos_embedding = nn.Parameter(torch.randn(1, self.time_step, self.embed_dim))
        self.timestep_embeding = nn.Embedding(self.time_step, self.embed_dim)

        # embed [TODO: is ReLu necessary?]
        self.to_obs_embed = nn.Sequential(
            nn.Linear(self.obs_dim, self.embed_dim)
        )
        self.to_action_embed = nn.Sequential(
            nn.Linear(self.action_dim, self.embed_dim)
        )

        # self.embed_to_action = nn.Sequential(
        #     nn.Linear(self.embed_dim * 2, self.embed_dim),
        #     Swish(),
        #     nn.Linear(self.embed_dim, 1)
        # )
        self.decoder_list1 = []
        self.decoder_list2 = []
        for i in range(self.n_ensemble):
            self.decoder_list1.append(LinearDecoder(self.embed_dim * 2, self.embed_dim, 1).to(self.device))
            self.decoder_list2.append(LinearDecoder(self.embed_dim * 2, self.embed_dim, 1).to(self.device))

        # learn parameters
        self.optimizer = self.configure_optimizers()

        self.parameter_number = sum(p.numel() for p in self.parameters())
        print("number of parameters: %e", self.parameter_number)


    def forward(self, obs=None, action=None, train=False):
        """
        :param obs: [batch * n_agent/n_timestep * dim]
        :param action:
        :param reward:
        :param obs_next:
        :return: o,a,r after reconstruction
        """
        device = obs.device
        batch_num, n_timestep, _ = obs.shape

        # cal position embedding
        obs_pos_embedding = repeat(self.obs_pos_embedding, 'b n d -> (b b_repeat) n d', b_repeat=batch_num)
        act_pos_embedding = repeat(self.act_pos_embedding, 'b n d -> (b b_repeat) n d', b_repeat=batch_num)

        # if mask, should use the specific token to replace the origin input
        # o-observation
        obs_token = self.to_obs_embed(obs)
        act_token = self.to_action_embed(action)

        # TODO: check weather agent_postion_embeding is neccessary
        obs_token += obs_pos_embedding
        act_token += act_pos_embedding
        cls_token = repeat(self.cls_token, 'b t d -> (b b_repeat) t d', b_repeat=batch_num)

        # if input the current info or the info should as a results, should feed into the network.
        # TODO:[0809] cheak classfication
        #tokens_ls = [cls_token, obs_token]
        # tokens_ls = [obs_token]
        tokens_ls = [obs_token, act_token]
        tokens_ls = rearrange(tokens_ls, 'n b t d-> b (t n) d')
        # tokens = torch.cat(tokens_ls, dim=1)

        tokens = tokens_ls

        # get the patches to be masked for the final reconstruction loss
        # attend with vision transformer [TODO: in CV, mlp head is used to merge infomation]
        encoded_tokens = self.encoder(tokens)

        # get the first token
        first_o_token = encoded_tokens[:, -2]
        first_a_token = encoded_tokens[:, -1]
        # -1, -2 affecting final performance?

        first_token = torch.cat((first_o_token, first_a_token), dim=1)
        # action = self.embed_to_action(first_token)
        action_list1 = []
        action_list2 = []
        for i in range(self.n_ensemble):
            action_list1.append(self.decoder_list1[i].forward(first_token))
            action_list2.append(self.decoder_list2[i].forward(first_token))

        return action_list1, action_list2

    def reset_optimizer(self):
        self.optimizer = self.configure_optimizers()
        return self.optimizer

    def configure_optimizers(self):
        config = self.config
        #TODO: update optimaizer setting, identify the actor and critic
        # learning rate schedular

        optimizer = torch.optim.AdamW(self.parameters(), lr=config.q_lr)

        return optimizer

    def getValue(self, obs, action):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
            action = rearrange(action, "b t c d-> (b t) c d")

        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)
        action = action[:, -self.n_agent:, :].to(device=self.device)

        value1, value2 = self.forward(obs, action)
        value = torch.min(value1[0], value2[0])
        if re_flag == True:
            value = rearrange(value, "(b t) 1 -> b t 1", b=b)
        return value

    def getEnsembleValue(self, obs, action):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
            action = rearrange(action, "b t c d-> (b t) c d")

        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)
        action = action[:, -self.n_agent:, :].to(device=self.device)

        value1, value2 = self.forward(obs, action)
        value_list = []
        if re_flag == True:
            for i in range(self.n_ensemble):
                value_list.append(rearrange(torch.min(value1[i], value2[i]), "(b t) 1 -> b t 1", b=b))

        return value_list

    def getActionDistribution(self, obs):
        pass

    def loss(self):
        pass
        # calculate reconstruction loss

class TransformerAgent(nn.Module):
    def __init__(
        self,
        config,
        decoder_dim_head = 64
    ):
        super().__init__()
        # construct encoder
        self.config = config
        self.n_agent = config.context_len # self.n_agent means the max input block
        self.n_ensemble = config.n_ensemble
        self.action_dim = config.action_space if config.mode == 'actor' else 1# since mean and std are concatenated
        self.obs_dim = (config.obs_space + config.action_space) if config.mode == 'Q' else config.obs_space
        self.device = config.device
        # self.action_max = config.action_max
        self.action_max_old = config.action_max
        self.action_max_0 = config.action_max_0
        self.action_max_1 = config.action_max_1
        self.action_max = torch.tensor([self.action_max_old] * self.action_dim).to(device=self.device)
        # self.action_max = torch.tensor([self.action_max_old, self.action_max_old]).to(device=self.device)
        # self.action_max = torch.tensor([1.0, 0.3]).to(device=self.device)
        self.sigma_scaler = torch.tensor([self.action_max_old * 2.0] * self.action_dim).to(device=self.device)
        # self.action_max.requires_grad = True
        self.sigma_min = -5
        self.sigma_max = 0.5

        self.time_step = self.n_agent
        self.device = config.device
        self.mode = config.mode

        self.embed_dim = config.n_embed
        self.encoder_head = config.n_head
        self.encoder_dim = config.n_embed
        self.encoder_layer = config.n_layer
        #self.mode = config.mode #'oa' 'oar' 'oaro'
        self.encoder = Transformer(self.encoder_dim,
                                   self.encoder_layer,
                                   self.encoder_head,
                                   decoder_dim_head,
                                   self.encoder_dim * 4) # TODO:dropout, dim_head

        # token [TODO: all the same or different?]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        # self.sigma_param = nn.Parameter(torch.ones(self.action_dim)*-2)#.to(self.device)
        self.register_parameter(name='sigma_param', param=nn.Parameter(torch.ones(self.action_dim)*-2))
        # self.register_parameter(name='sigma_param', param=nn.Parameter(torch.tensor([-4.0, -4.0])))
        # self.sigma_param = nn.Parameter(torch.tensor([-1.0, -1.0]))  # .to(self.device)

        # position embeding
        self.obs_pos_embedding = nn.Parameter(torch.randn(1, self.time_step, self.embed_dim))
        self.timestep_embeding = nn.Embedding(self.time_step, self.embed_dim)

        # embed [TODO: is ReLu necessary?]
        self.to_obs_embed = nn.Sequential(
            nn.Linear(self.obs_dim, self.embed_dim)
        )

        # self.embed_to_action = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     Swish(),
        #     nn.Linear(self.embed_dim, self.action_dim)
        # )
        self.decoder_list = []
        for i in range(self.n_ensemble):
            self.decoder_list.append(LinearDecoder(self.embed_dim, self.embed_dim, self.action_dim).to(self.device))

        # learn parameters
        self.optimizer = self.configure_optimizers()

        self.parameter_number = sum(p.numel() for p in self.parameters())
        print("number of parameters: %e", self.parameter_number)


    def forward(self, obs=None, train=False):
        """
        :param obs: [batch * n_agent/n_timestep * dim]
        :param action:
        :param reward:
        :param obs_next:
        :return: o,a,r after reconstruction
        """
        device = obs.device
        batch_num, n_timestep, _ = obs.shape

        # cal position embedding
        obs_pos_embedding = repeat(self.obs_pos_embedding, 'b n d -> (b b_repeat) n d', b_repeat=batch_num)

        # if mask, should use the specific token to replace the origin input
        # o-observation
        obs_token = self.to_obs_embed(obs)

        # TODO: check weather agent_postion_embeding is neccessary
        obs_token += obs_pos_embedding
        cls_token = repeat(self.cls_token, 'b t d -> (b b_repeat) t d', b_repeat=batch_num)

        # if input the current info or the info should as a results, should feed into the network.
        # TODO:[0809] cheak classfication
        #tokens_ls = [cls_token, obs_token]
        tokens_ls = [obs_token]
        tokens = torch.cat(tokens_ls, dim=1)

        # get the patches to be masked for the final reconstruction loss
        # attend with vision transformer [TODO: in CV, mlp head is used to merge infomation]
        encoded_tokens = self.encoder(tokens)

        # get the first token
        first_token = encoded_tokens[:, -1]

        action_list = []
        for i in range(self.n_ensemble):
            action_list.append(self.decoder_list[i].forward(first_token))

        # action = self.embed_to_action(first_token)

        return action_list

    def reset_optimizer(self):
        self.optimizer = self.configure_optimizers()
        return self.optimizer

    def configure_optimizers(self):
        config = self.config
        #TODO: update optimaizer setting, identify the actor and critic
        # learning rate schedular
        if self.mode == 'actor':
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.a_lr)#, betas=train_config.betas)
        elif self.mode == 'critic':
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.c_lr)
        elif self.mode == 'Q':
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.q_lr)

        return optimizer


    def getVecAction(self, obs, train=True):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)[0]

        a_ = torch.mul(torch.tanh(a), self.action_max)


        if re_flag == True:
            a_ = rearrange(a_, '(b t) d -> b t d', b=b)

        return a_.detach().cpu().numpy()


    def getEnsembleAction(self, obs, train=True):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)

        action_list = []

        for i in range(self.n_ensemble):
            a_ = torch.mul(torch.tanh(a[i]), self.action_max)

            if re_flag == True:
                a_ = rearrange(a_, '(b t) d -> b t d', b=b)

            action_list.append(a_.detach().cpu().numpy())

        return action_list

    def imitate_action(self, obs):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)
        # a = rearrange(a, 'b (s d)-> (b s) d', s=2)
        # mu, sigma = (a[:, 0], a[:, 1])
        mu, sigma = a, repeat(self.sigma_param, 'd -> b d', b=a.shape[0]).exp()
        sigma = torch.mul(sigma, self.sigma_scaler)
        # new_mu = torch.tanh(mu)
        # new_mu[0] *= self.action_max_0
        # new_mu[1] *= self.action_max_1
        # mu = new_mu
        mu = torch.mul(torch.tanh(mu), self.action_max)
        # mu = torch.clamp(mu, min=-self.action_max, max=self.action_max)
        # sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()
        a_dis = Independent(torch.distributions.Normal(mu, sigma), 1)
        a_ = a_dis.sample()
        self.entropy = a_dis.entropy().mean().item()

        if re_flag == True:
            a_ = rearrange(a_, '(b t) d -> b t d', b=b)
            mu = rearrange(mu, '(b t) d -> b t d', b=b)

        return a_, mu

    def getActionDistribution(self, obs):
        pass

    def loss(self):
        pass





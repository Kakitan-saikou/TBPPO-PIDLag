import copy

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.distributions import Independent
from model.mae import Transformer, maskedTransformer
from typing import Dict, List, Union, Tuple, Optional

from model.spectral_layers import SpectralRegressor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleLinearDecoder(nn.Module):
    def __init__(self, n_ensemble, in_dim, hid_dim, out_dim, device):
        super().__init__()
        self.w1 = torch.randn(n_ensemble, in_dim, hid_dim).to(device)
        self.b1 = torch.randn(n_ensemble, 1, hid_dim).to(device)
        self.w2 = torch.randn(n_ensemble, hid_dim, out_dim).to(device)
        self.b2 = torch.randn(n_ensemble, 1, out_dim).to(device)

        # Initialize
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        self.b1.zero_()
        self.b2.zero_()

    def swish_func(self, x):
        return x * torch.sigmoid(x)

    def forward(self, in_feature):
        if len(in_feature.shape) == 2:
            # Common 2-dim input(batch * dim)
            hidden = self.swish_func((torch.einsum('bi,nid->nbd', in_feature, self.w1) + self.b1))
            out_feature = torch.einsum('nbd,ndo->nbo', hidden, self.w2) + self.b2
        else:
            batch_len = in_feature.shape[0]
            in_feature_d = rearrange(in_feature, 'b c i -> (b c) i')
            hidden = self.swish_func((torch.einsum('si,nid->nsd', in_feature_d, self.w1) + self.b1))
            out_feature = torch.einsum('nsd,ndo->nso', hidden, self.w2) + self.b2
            out_feature = rearrange(out_feature, 'n (b c) o -> n b c o', b=batch_len)

        return out_feature

class DynamicModel(nn.Module):
    def __init__(
        self,
        config,
        decoder_dim_head = 64
    ):
        super().__init__()
        # construct encoder
        self.config = config
        self.n_agent = config.context_len # self.n_agent means the max input block
        self.action_dim = config.action_space
        self.obs_dim = config.obs_space
        self.sa2s = config.sa2s
        self.spectral = config.spectral
        self.n_ensemble = config.n_ensemble
        # self.action_max = config.action_max

        self.time_step = self.n_agent
        self.device = config.device
        self.addition_encoder = config.addition_encoder
        # self.mode = config.mode

        self.embed_dim = config.n_embed
        self.encoder_head = config.n_head
        self.encoder_dim = config.n_embed
        self.encoder_layer = config.n_layer
        #self.mode = config.mode #'oa' 'oar' 'oaro'
        self.encoder = Transformer(self.encoder_dim,
                                   self.encoder_layer,
                                   self.encoder_head,
                                   decoder_dim_head,
                                   self.encoder_dim * 4,
                                   dropout=0.1,
                                   att_type='Galerkin') # TODO:dropout, dim_head

        if self.addition_encoder:
            self.encoder_sa = Transformer(self.encoder_dim * 2,
                                          self.encoder_layer,
                                          self.encoder_head,
                                          decoder_dim_head * 2,
                                          self.encoder_dim * 4,
                                          dropout=0.1,
                                          att_type='Galerkin')

        # self.encoder = Transformer(self.encoder_dim,
        #                            self.encoder_layer,
        #                            self.encoder_head,
        #                            decoder_dim_head,
        #                            self.encoder_dim * 4) # TODO:dropout, dim_head

        # self.encoder = maskedTransformer(self.encoder_dim,
        #                            self.encoder_layer,
        #                            self.encoder_head,
        #                            decoder_dim_head,
        #                            self.encoder_dim * 4,
        #                            att_type= 'ScaleDot') # TODO:dropout, dim_head

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(self.obs_dim) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(self.obs_dim) * -10, requires_grad=True)
        )

        # token [TODO: all the same or different?]
        self.cls = config.cls
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        # self.sigma_param = nn.Parameter(torch.ones(self.action_dim)*-2)#.to(self.device)
        self.register_parameter(name='sigma_param', param=nn.Parameter(torch.ones(self.action_dim)*-2))
        # self.register_parameter(name='sigma_param', param=nn.Parameter(torch.tensor([-4.0, -4.0])))
        # self.sigma_param = nn.Parameter(torch.tensor([-1.0, -1.0]))  # .to(self.device)
        self.with_std = config.with_std
        self.modes = config.modes

        self.mseloss = nn.MSELoss()

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

        if self.n_ensemble > 1:
            if self.with_std:
                self.embed_to_obs_next = EnsembleLinearDecoder(self.n_ensemble, self.embed_dim * 2, self.embed_dim,
                                                               self.obs_dim * 2, self.device)
            else:
                self.embed_to_obs_next = EnsembleLinearDecoder(self.n_ensemble, self.embed_dim * 2, self.embed_dim,
                                                               self.obs_dim, self.device)

            self.embed_to_reward = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                Swish(),
                nn.Linear(self.embed_dim, 2)
            )
        else:

            if self.with_std:
                self.embed_to_obs_next = nn.Sequential(
                    nn.Linear(self.embed_dim * 2, self.embed_dim),
                    Swish(),
                    nn.Linear(self.embed_dim, self.obs_dim * 2)
                )
                self.embed_to_reward = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    Swish(),
                    nn.Linear(self.embed_dim, 2)
                )



            else:

                if self.spectral:
                    self.embed_to_obs_next = SpectralRegressor(self.embed_dim * 2,
                                                               self.embed_dim * 2,
                                                               self.embed_dim,
                                                               self.obs_dim,
                                                               self.modes,
                                                               dropout=0.05)
                else:
                    self.embed_to_obs_next = nn.Sequential(
                        nn.Linear(self.embed_dim * 2, self.embed_dim),
                        Swish(),
                        nn.Linear(self.embed_dim, self.obs_dim)
                    )
                    self.embed_to_reward = nn.Sequential(
                        nn.Linear(self.embed_dim, self.embed_dim),
                        Swish(),
                        nn.Linear(self.embed_dim, 1)
                    )

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
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
            action = rearrange(action, "b t c d-> (b t) c d")
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
        tokens_ls = [obs_token, act_token]
        tokens_ls = rearrange(tokens_ls, 'n b t d-> b (t n) d')  # change the seq [oooorrrrraaaa] to seq [oraoraoraora]
        if self.cls:
            tokens_ls = [cls_token] + tokens_ls
        # tokens = torch.cat(tokens_ls, dim=1)

        # get the patches to be masked for the final reconstruction loss
        # attend with vision transformer [TODO: in CV, mlp head is used to merge infomation]
        encoded_tokens = self.encoder(tokens_ls)

        if self.cls:
            obs_tokens, action_tokens = rearrange(encoded_tokens[:, 1:, :], 'b (t n) d -> n b t d', n=2)
        else:
            obs_tokens, action_tokens = rearrange(encoded_tokens, 'b (t n) d -> n b t d', n=2)

        # If train, provide full sequence; if inference, only need to predict the upcoming 'next state'
        if self.sa2s:
            token_mid = torch.cat((obs_tokens, action_tokens), dim=-1)
        else:
            token_mid = obs_tokens

        if self.addition_encoder:
            token_out = self.encoder_sa(token_mid)
        else:
            token_out = token_mid

        if train:
            obs_next = self.embed_to_obs_next(token_out)
        else:
            first_token = token_out[:, -1]
            obs_next = self.embed_to_obs_next(first_token)

        if re_flag == True:
            if self.n_ensemble > 1:
                obs_next = rearrange(obs_next, 'n (b t) c d -> n b t c d', b=b)
            else:
                obs_next = rearrange(obs_next, '(b t) c d -> b t c d', b=b)
            # a_log = rearrange(a_log, '(b t) d -> b t d', b=b)


        if self.with_std:
            mean, logvar = torch.chunk(obs_next, 2, dim=-1)
            logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        else:
            mean = obs_next
            logvar = None

        return mean, logvar


    def reset_optimizer(self):
        self.optimizer = self.configure_optimizers()
        return self.optimizer

    def configure_optimizers(self):
        config = self.config
        #TODO: update optimaizer setting, identify the actor and critic
        # learning rate schedular

        optimizer = torch.optim.AdamW(self.parameters(), lr=config.d_lr)

        return optimizer

    def set_elites(self, index):
        self.register_parameter('elites', nn.Parameter(torch.tensor(index), requires_grad=False))
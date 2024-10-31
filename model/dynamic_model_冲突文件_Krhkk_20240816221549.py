import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.distributions import Independent
from model.mae_deterministic import Transformer, maskedTransformer
from typing import Dict, List, Union, Tuple, Optional

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
        # self.action_max = config.action_max

        self.time_step = self.n_agent
        self.device = config.device
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
                                   self.encoder_dim * 4) # TODO:dropout, dim_head

        # self.encoder = maskedTransformer(self.encoder_dim,
        #                            self.encoder_layer,
        #                            self.encoder_head,
        #                            decoder_dim_head,
        #                            self.encoder_dim * 4,
        #                            att_type= 'Galerkin') # TODO:dropout, dim_head

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

        if self.sa2s:
            self.embed_to_obs_next = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                Swish(),
                nn.Linear(self.embed_dim, self.obs_dim * 2)
            )

            self.embed_to_reward = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                Swish(),
                nn.Linear(self.embed_dim, 2)
            )
        else:
            self.embed_to_obs_next = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                Swish(),
                nn.Linear(self.embed_dim, self.obs_dim * 2)
            )

            self.embed_to_reward = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                Swish(),
                nn.Linear(self.embed_dim, 2)
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
            token_out = torch.cat((obs_tokens, action_tokens), dim=-1)
        else:
            token_out = obs_tokens

        if train:
            obs_next = self.embed_to_obs_next(token_out)
        else:
            first_token = token_out[:, -1]
            obs_next = self.embed_to_obs_next(first_token)

        if re_flag == True:
            obs_next = rearrange(obs_next, '(b t) c d -> b t c d', b=b)
            # a_log = rearrange(a_log, '(b t) d -> b t d', b=b)

        mean, logvar = torch.chunk(obs_next, 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)

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
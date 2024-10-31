import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.distributions import Independent

import copy

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x






class MLPAgent(nn.Module):
    def __init__(
        self,
        config,
        decoder_dim_head = 64
    ):
        super().__init__()
        # construct encoder
        self.config = config
        self.n_agent = config.context_len # self.n_agent means the max input block
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
        self.sigma_scaler = torch.tensor([1.0] * self.action_dim).to(device=self.device)
        # self.action_max.requires_grad = True
        self.sigma_min = -6
        self.sigma_max = 0.5

        self.time_step = self.n_agent
        self.device = config.device
        self.mode = config.mode

        self.embed_dim = 256
        self.encoder_head = config.n_head
        self.encoder_dim = config.n_embed
        self.encoder_layer = config.n_layer
        #self.mode = config.mode #'oa' 'oar' 'oaro'

        # token [TODO: all the same or different?]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        # self.sigma_param = nn.Parameter(torch.ones(self.action_dim)*-2)#.to(self.device)
        self.register_parameter(name='sigma_param', param=nn.Parameter(torch.ones(self.action_dim)*-1))
        # self.register_parameter(name='sigma_param', param=nn.Parameter(torch.tensor([-4.0, -4.0])))
        # self.sigma_param = nn.Parameter(torch.tensor([-1.0, -1.0]))  # .to(self.device)
        self.cls = config.cls

        # position embeding
        self.obs_pos_embedding = nn.Parameter(torch.randn(1, self.time_step, self.embed_dim))
        self.timestep_embeding = nn.Embedding(self.time_step, self.embed_dim)

        # embed [TODO: is ReLu necessary?]

        self.obs_to_action = nn.Sequential(
            nn.Linear(self.obs_dim, self.embed_dim),
            Swish(),
            nn.Linear(self.embed_dim, self.embed_dim),
            Swish(),
            nn.Linear(self.embed_dim, self.action_dim)
        )

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

        # get the first token
        first_token = obs[:, -1]
        action = self.obs_to_action(first_token)

        return action

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

    def getValue(self, obs):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")

        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        value = self.forward(obs)
        if re_flag == True:
            value = rearrange(value, "(b t) 1 -> b t 1", b=b)
        return value

    def getAction(self, obs, train=False):
        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)
        # a = rearrange(a, 'b (s d)-> (b s) d', s=2)
        # mu, sigma = (a[:, 0], a[:, 1]
        mu, sigma = rearrange(a, 'b d-> (b d)'), repeat(self.sigma_param, 'd -> (b d)', b=a.shape[0])
        # sigma = torch.mul(sigma, self.sigma_scaler)
        # new_mu = torch.tanh(mu)
        # new_mu[0] *= self.action_max_0
        # new_mu[1] *= self.action_max_1
        # mu = new_mu
        mu = torch.mul(torch.tanh(mu), self.action_max)
        # mu = self.action_max * torch.tanh(mu)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()
        a_dis = torch.distributions.Normal(mu, sigma)
        a_ = a_dis.sample().detach().cpu().numpy()
        if train:
            a_log = a_dis.log_prob(a_).detach().cpu().numpy()
        else:
            a_log = None
        return a_, a_log

    def getActionLogProb(self, obs, action, train=False, entropy=False):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
            action = rearrange(action, "b t d-> (b t) d")

        # input dim [batch * context * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)
        # a = rearrange(a, 'b (s d)-> (b s) d', s=2)
        # mu, sigma = (a[:, 0], a[:, 1]
        # [trick] tanh action
        mu, sigma = a, repeat(self.sigma_param, 'd -> b d', b=a.shape[0])
        # sigma = torch.mul(sigma, self.sigma_scaler)
        # new_mu = torch.tanh(mu)
        # new_mu[0] *= self.action_max_0
        # new_mu[1] *= self.action_max_1
        # mu = new_mu
        mu = torch.mul(torch.tanh(mu), self.action_max)
        # mu = self.action_max * torch.tanh(mu)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()
        if train:
            a_dis = Independent(torch.distributions.Normal(mu, sigma), 1)
        else:
            a_dis = Independent(torch.distributions.Normal(mu.detach(), sigma.detach()), 1)
        a_log = a_dis.log_prob(action)

        if re_flag == True:
            a_log = rearrange(a_log, "(b t) -> b t", b=b)

        if entropy:
            return a_log, a_dis.entropy().mean()
        else:
            return a_log

    def getVecAction(self, obs, train=True):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)
        #a = rearrange(a, 'b (s d)-> (b s) d', s=2)
        #mu, sigma = (a[:, 0], a[:, 1])
        mu, sigma = a, repeat(self.sigma_param, 'd -> b d', b=a.shape[0])
        # sigma = torch.mul(sigma, self.sigma_scaler)
        # print(sigma)
        # new_mu = torch.tanh(mu)
        # new_mu[0] *= self.action_max_0
        # new_mu[1] *= self.action_max_1
        # mu = new_mu
        mu = torch.mul(torch.tanh(mu), self.action_max)
        # mu = self.action_max * torch.tanh(mu)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()
        a_dis = Independent(torch.distributions.Normal(mu, sigma), 1)
        a_ = a_dis.sample()
        self.entropy = a_dis.entropy().mean().item()
        if train:
            a_log = a_dis.log_prob(a_).detach().cpu().numpy()
        else:
            a_ = mu
            a_log = None

        if re_flag == True:
            a_ = rearrange(a_, '(b t) d -> b t d', b=b)

        return a_.detach().cpu().numpy(), a_log

    def get_mu_prob(self, obs, action, train=True):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
            action = rearrange(action, "b t d-> (b t) d")
        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)

        mu, sigma = a, repeat(self.sigma_param, 'd -> b d', b=a.shape[0])
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()

        mu = torch.mul(torch.tanh(mu), self.action_max)

        a_dis = Independent(torch.distributions.Normal(mu, sigma), 1)

        a_log = a_dis.log_prob(action)

        if re_flag == True:
            a_log = rearrange(a_log, "(b t) -> b t", b=b)
            mu = rearrange(mu, '(b t) d -> b t d', b=b)

        return mu, a_log

    def imitate_action_batch(self, obs, a_ref, mu_ratio, dist_ratio):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
            a_ref = rearrange(a_ref, "b t d-> (b t) d")
        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)
        # a = rearrange(a, 'b (s d)-> (b s) d', s=2)
        # mu, sigma = (a[:, 0], a[:, 1])
        mu, sigma = a, repeat(self.sigma_param, 'd -> b d', b=a.shape[0])
        sigma = torch.mul(sigma, self.sigma_scaler)

        mu = torch.mul(torch.tanh(mu), self.action_max)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()

        a_dis = Independent(torch.distributions.Normal(mu, sigma), 1)
        a_log = a_dis.log_prob(a_ref)

        self.entropy = a_dis.entropy().mean().item()

        loss = F.mse_loss(mu, a_ref).mean() * mu_ratio
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

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
        mu, sigma = a, repeat(self.sigma_param, 'd -> b d', b=a.shape[0])
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()
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


class MLPQAgent(nn.Module):
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


        # token [TODO: all the same or different?]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        # self.sigma_param = nn.Parameter(torch.ones(self.action_dim)*-2)#.to(self.device)
        self.register_parameter(name='sigma_param', param=nn.Parameter(torch.ones(self.action_dim)*-2))
        # self.register_parameter(name='sigma_param', param=nn.Parameter(torch.tensor([-4.0, -4.0])))
        # self.sigma_param = nn.Parameter(torch.tensor([-1.0, -1.0]))  # .to(self.device)

        # position embeding


        # embed [TODO: is ReLu necessary?]

        self.obs_to_action = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.embed_dim),
            Swish(),
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

        # get the first token
        first_o_token = obs[:, -1]
        first_a_token = action[:, -1]
        # -1, -2 affecting final performance?

        first_token = torch.cat((first_o_token, first_a_token), dim=-1)
        q = self.obs_to_action(first_token)

        return q

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

        value = self.forward(obs, action)
        if re_flag == True:
            value = rearrange(value, "(b t) 1 -> b t 1", b=b)
        return value

    def getActionDistribution(self, obs):
        pass

    def loss(self):
        pass
        # calculate reconstruction loss
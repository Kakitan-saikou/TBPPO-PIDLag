import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Categorical
import torch.nn as nn
import os
from einops import rearrange, repeat
from torch.optim import Adam
# from framework.buffer1 import Replay_buffer
from model.actor1 import GaussianActor, RActor
from model.critic1 import Critic, RCritic
import wandb
from model.mae import TransformerAgent
from model.mae_variable_sigma import TransformerAgentVS
from model.mlp import MLPAgent
from framework.buffer_beifen import Buffer as buffer
from copy import deepcopy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #  random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(7)


def trajectory_property():
    return ["action", "hidden", "next_hidden", "hidden_q",
            "next_hidden_q", "hidden_q_target",
            "next_hidden_q_target", "id"]
    # return ["action", "id"]


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for step_idx in reversed(range(td_delta.shape[-1])):
        delta = td_delta[:, step_idx]
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float).T


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def update_params(optim, loss, clip=False, param_list=False, retain_graph=False):
    optim.zero_grad()
    loss.backward()
    if clip is not False:
        for i in param_list:
            nn.utils.clip_grad_norm_(i, clip)
    optim.step()


class BC_ensemble(object):
    def __init__(self, args):
        # env parameters
        self.state_dim = args.obs_space
        self.action_dim = args.action_space
        self.env_num = args.env_num
        self.device = args.device

        # model parameters
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.n_embed = args.n_embed
        self.max_seq_len = args.context_len
        self.n_ensemble = args.n_ensemble

        # training parameters
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lamda = args.gae_lambda
        self.clip = args.ppo_clip
        self.epoch = args.ppo_epoch
        self.entropy = args.ppo_entropy
        self.grad_norm = args.grad_norm_clip
        self.actor_loss = 0
        self.critic_loss = 0


        self.bc_kl = args.bc_kl
        self.kl_alpha = args.kl_alpha
        self.mu_ratio = args.mu_ratio
        self.dist_ratio = args.dist_ratio
        self.sigma_type = args.sigma_type

        if args.sigma_type == 'share':
            self.mu_ratio = 1.0
            self.dist_ratio = 0.1

            # define actor and critic
            args.mode = "actor"
            ensemble = []
            if args.mlp == True:
                for i in range(self.n_ensemble):
                    bc = MLPAgent(args).to(self.device)
                    ensemble.append(bc)
            else:
                for i in range(self.n_ensemble):
                    bc = TransformerAgent(args).to(self.device)
                    ensemble.append(bc)
            self.ensemble = ensemble

        elif args.sigma_type == 'variable':
            self.mu_ratio = 0.05
            self.dist_ratio = 1.0

            # define actor and critic
            args.mode = "actor"
            ensemble = []
            for i in range(self.n_ensemble):
                bc = TransformerAgentVS(args).to(self.device)
                ensemble.append(bc)
            self.ensemble = ensemble

        self.reset_optimizer()

    def reset_optimizer(self):
        for bc in self.ensemble:
            bc.reset_optimizer()

    def get_ensemble(self, ) -> list:
        return self.ensemble

    def vs_train(self, s, a, alpha: float, shuffle: bool = True) -> list:
        # s, a, _, _, _, _, _, _ = replay_buffer.sample(self.batch_size)

        losses = []
        # separately train each polciy
        if alpha == 0. or self.n_ensemble == 1:
            for bc in self.ensemble:
                each_loss = bc.imitate_action_batch(s, a, self.mu_ratio, self.dist_ratio)
                losses.append(each_loss)
        # jointly train each behavior policy
        else:
            all_prob_a, means, dists, stds = [], [], [], []
            p_is = np.arange(0, self.n_ensemble)
            # shuffle pi's order
            if shuffle:
                np.random.shuffle(p_is)
                for i, p_i in enumerate(p_is):
                    bc = self.ensemble[p_i]
                    mu, prob = bc.get_mu_prob(s, a)
                    means.append(mu)
                    all_prob_a.append(prob.unsqueeze(0))


                for i, p_i in enumerate(p_is):
                    # print(i, p_i)
                    bc = self.ensemble[p_i]

                    others_prob = [all_prob_a[pb].detach() for pb in range(self.n_ensemble)]
                    # others_prob = deepcopy(all_prob_a.detach())
                    # del others_mu[i]
                    del others_prob[i]

                    probs = torch.cat(all_prob_a, dim=0)
                    max_prob, id = probs.max(0)

                    loss = (-all_prob_a[i].mean() * self.dist_ratio - self.kl_alpha * (all_prob_a[i] - max_prob.detach()).mean() * self.dist_ratio
                            + F.mse_loss(means[i], a) * self.mu_ratio)
                    bc.optimizer.zero_grad()
                    loss.backward()
                    bc.optimizer.step()

                    all_prob_a[i] = bc.getActionLogProb(s, a).unsqueeze(0).detach()

                    losses.append(loss.item())
            else:
                pass

        return losses

    def sh_train(self, s, a, ratio: float) -> list:
        losses = []
        # separately train each polciy
        if ratio == 1. or self.n_ensemble == 1:
            for bc in self.ensemble:
                each_loss = bc.imitate_action_batch(s, a, self.mu_ratio, 0.0)
                losses.append(each_loss)
        else:
            for i in range(self.n_ensemble):
                each_loss = self.ensemble[i].imitate_action_batch(s[i], a[i], self.mu_ratio, 0.0)
                losses.append(each_loss)

        return losses

    def ensemble_train(self, s, a, parameter: float, shuffle: bool = True) -> list:
        if self.sigma_type == 'share':
            losses = self.sh_train(s, a, ratio=parameter)
        elif self.sigma_type == 'variable':
            losses = self.vs_train(s, a, alpha=parameter, shuffle=shuffle)

        return losses


    def evaluation(self,
                   env_name: str,
                   seed: int,
                   mean: np.ndarray,
                   std: np.ndarray,
                   eval_episodes: int = 10) -> list:
        scores = []
        for i in range(self.n_ensemble):
            bc = self.ensemble[i]
            each_score = bc.offline_evaluate(env_name, seed, mean, std, eval_episodes=eval_episodes)
            scores.append(each_score)
        return np.array(scores)

    def ensemble_save(self, save_path: str, save_id: list) -> None:
        train_path = os.path.join(save_path, 'trained_model')
        base_path = os.path.join(train_path, 'BC')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if save_id:
            for i in save_id:
                bc = self.ensemble[i]

                model_actor_path = os.path.join(base_path, "actor_" + str(i) + ".pth")
                torch.save(bc.state_dict(), model_actor_path)


    def ensemble_load(self, save_path: str) -> None:
        train_path = os.path.join(save_path, 'trained_model')
        base_path = os.path.join(train_path, 'BC')
        # if not os.path.exists(base_path):
        #     os.makedirs(base_path)

        for i in range(self.n_ensemble):
            model_actor_path = os.path.join(base_path, "actor_" + str(i) + ".pth")
            self.ensemble[i].load_state_dict(torch.load(model_actor_path,  map_location=self.device), strict=False)

    def ope_dynamics_eval(self, args, dynamics_eval, q_eval, dynamics, eval_buffer, env, mean, std):
        best_mean_qs = []
        for bc in self.ensemble:
            best_mean_q, _ = dynamics_eval(args, bc, q_eval, dynamics, eval_buffer, env, mean, std)
            best_mean_qs.append(best_mean_q)
        return np.array(best_mean_qs)

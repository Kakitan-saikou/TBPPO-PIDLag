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

class TBPPO(object):
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

        # training parameters
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lamda = args.gae_lambda
        self.clip = args.ppo_clip
        self.epoch = args.offline_epoch
        self.entropy = args.ppo_entropy
        self.grad_norm = args.grad_norm_clip
        self.actor_loss = 0
        self.critic_loss = 0

        self.entropy_ratio = args.entropy_ratio

        # define actor and critic
        args.mode = "actor"
        if args.mlp == True:
            self.actor = MLPAgent(args).to(self.device)
        else:
            if args.sigma_type == 'share':
                self.actor = TransformerAgent(args).to(self.device)
            else:
                self.actor = TransformerAgentVS(args).to(self.device)
        self.old_actor = deepcopy(self.actor).to(self.device)

        # define for transformer
        self.state = torch.zeros(self.env_num, self.max_seq_len ,self.state_dim).to(self.device)

        # define buffer
        self.memory = buffer(args)


    def reset_optimizer(self):
        self.actor.reset_optimizer()

    def reset_state(self):
        self.state = torch.zeros(self.env_num, self.max_seq_len ,self.state_dim).to(self.device)

    def choose_action(self, state, train=True):
        # TODO: [0807]inference by context
        state = torch.FloatTensor(state).unsqueeze(1).to(self.device)
        self.state = torch.cat([self.state, state], dim=1)[:, 1:, :]
        if train:
            action, _ = self.actor.getVecAction(self.state)
        else:
            action, _ = self.actor.getVecAction(self.state, train=False)
        return action

    def initial_action(self, state, rolling_a, actor_old=False, train=True):
        state = torch.FloatTensor(state).to(self.device)
        self.state = state

        if actor_old:
            if train:
                action, _ = self.old_actor.getVecAction(self.state)
            else:
                action, _ = self.old_actor.getVecAction(self.state, train=False)
        else:
            if train:
                action, _ = self.actor.getVecAction(self.state)
            else:
                action, _ = self.actor.getVecAction(self.state, train=False)

        # Scaling
        act = action / 12.0

        if len(state.shape) == 4:
            new_rolling_a = np.concatenate((rolling_a[:, :, :-1, :], np.expand_dims(act, axis=2)), axis=2)
        else:
            new_rolling_a = np.concatenate((rolling_a[:, :-1, :], np.expand_dims(act, axis=1)), axis=1)

        return new_rolling_a

    def offline_action(self, state, rolling_a, train=True):
        state = torch.FloatTensor(state).to(self.device)
        self.state = state
        if train:
            action, _ = self.actor.getVecAction(self.state)
        else:
            action, _ = self.actor.getVecAction(self.state, train=False)

        act = action / 12.0
        # new_rolling_a = np.concatenate((rolling_a[:, 1:, :], np.expand_dims(act, axis=1)), axis=1)
        if len(state.shape) == 4:
            new_rolling_a = np.concatenate((rolling_a[:, :, 1:, :], np.expand_dims(act, axis=2)), axis=2)
        else:
            new_rolling_a = np.concatenate((rolling_a[:, 1:, :], np.expand_dims(act, axis=1)), axis=1)

        return new_rolling_a

    def offline_learn(self, state, action, advantage):
        # input dim [batch * context_len * dim], batch -> [num_envs * time_steps]
        # state, action, reward, done, next_state = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        advantage = torch.FloatTensor(advantage).to(self.device)

        ## ppo update
        # td_target = reward + self.gamma * self.critic.getValue(next_state).reshape(state.shape[0], state.shape[1]) * (1 - done) #20*1
        # td_error = td_target - self.critic.getValue(state).reshape(state.shape[0], state.shape[1])
        # advantage = compute_advantage(self.gamma, self.lamda, td_error.cpu()).to(self.device)
        # [trick] : advantage normalization
        # td_lamda_target = advantage + self.critic.getValue(state).reshape(state.shape[0], state.shape[1])
        advantage = ((advantage - advantage.mean()) / (advantage.std() +1e-5)).detach()
        # print('Advantages: ', advantage)
        old_action_log_prob = self.old_actor.getActionLogProb(state, action)


        for _ in range(self.epoch):
            # sample new action and new action log prob
            # print(_)
            new_action_log_prob, new_entropy = self.actor.getActionLogProb(state, action, train=True, entropy=True)
            # update actor
            ratio = (new_action_log_prob - old_action_log_prob).exp()
            surprise = ratio * advantage
            clipped_surprise = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
            actor_loss = -torch.min(surprise, clipped_surprise).mean() - new_entropy * self.entropy_ratio

            # update
            self.actor.optimizer.zero_grad()
            actor_loss.backward()

            # trick: clip gradient
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)

            self.actor.optimizer.step()

        # the fraction of the training data that triggered the clipped objective
        self.clipfrac = torch.mean(torch.greater(torch.abs(ratio - 1), self.clip).float()).item()
        self.approxkl = torch.mean(-new_action_log_prob + old_action_log_prob).item()

        self.actor_loss = actor_loss.item()

        return actor_loss

    def replace_old_policy(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

    def insert_data(self, data):
        for k, v in data.items():
            self.memory.insert(k, v)


    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)

    def load(self, file):
        # load data from file, and map to the correct device
        # self.actor.load_state_dict(torch.load(file, map_location='cuda:0'))
        self.actor.load_state_dict(torch.load(file,  map_location=self.device), strict=False)
        self.old_actor.load_state_dict(torch.load(file, map_location=self.device), strict=False)
        self.actor.reset_optimizer()


class ETBPPO(object):
    def __init__(self, args):
        self.n_ensemble = args.n_ensemble
        self.ensemble = []

        for _ in range(self.n_ensemble):
            agent = TBPPO(args)
            self.ensemble.append(agent)

    def reset_optimizer(self):
        for agent in self.ensemble:
            agent.reset_optimizer()

    def ensemble_train(self, state, action, advantage, selects=None):
        ensemble_loss = []

        for i in range(self.n_ensemble):
            agent = self.ensemble[i]
            if selects != None:
                if i in selects:
                    agent_loss = agent.offline_learn(state, action[i], advantage[i])
                else:
                    agent_loss = 0.0
            else:
                agent_loss = agent.offline_learn(state, action[i], advantage[i])
            ensemble_loss.append(agent_loss)

        return ensemble_loss

    def ensemble_get_action(self, obs, seq_action):
        ensemble_seq_a = []
        ensemble_a = []

        for agent in self.ensemble:
            new_seq_a = agent.initial_action(obs, seq_action, actor_old=True, train=True)
            if len(new_seq_a.shape) == 4:
                new_a = new_seq_a[:, :, -1, :] * 12.0
            else:
                new_a = new_seq_a[:, -1, :] * 12.0

            ensemble_seq_a.append(new_seq_a)
            ensemble_a.append(new_a)

        return ensemble_seq_a, ensemble_a

    def replace_old_policy(self, change_list):
        for index in change_list:
            self.ensemble[index].replace_old_policy()

    def ensemble_load(self, bc_path):
        # base_path = os.path.join(total_path, 'trained_model')
        # bc_path = os.path.join(base_path, 'BC')

        for i in range(self.n_ensemble):
            model_actor_path = os.path.join(bc_path, "actor_" + str(i) + ".pth")
            self.ensemble[i].load(model_actor_path)

    def ensemble_save(self, total_path, flag):
        base_path = os.path.join(total_path, 'trained_model')
        bppo_path = os.path.join(base_path, 'BPPO')
        if not os.path.exists(bppo_path):
            os.makedirs(bppo_path)

        if flag==None:
            for i in range(self.n_ensemble):
                model_actor_path = os.path.join(bppo_path, "actor_" + str(i) + ".pth")
                torch.save(self.ensemble[i].actor.state_dict(), model_actor_path)
        else:
            for i in range(self.n_ensemble):
                model_actor_path = os.path.join(bppo_path, "actor_" + str(i) + '_' + flag + ".pth")
                torch.save(self.ensemble[i].actor.state_dict(), model_actor_path)


    def indexed_save(self, total_path, flag, index):
        base_path = os.path.join(total_path, 'trained_model')
        bppo_path = os.path.join(base_path, 'BPPO')
        if not os.path.exists(bppo_path):
            os.makedirs(bppo_path)

        if len(index) > 0:
            if flag == None:
                for i in index:
                    model_actor_path = os.path.join(bppo_path, "actor_" + str(i) + ".pth")
                    torch.save(self.ensemble[i].actor.state_dict(), model_actor_path)
            else:
                for i in index:
                    model_actor_path = os.path.join(bppo_path, "actor_" + str(i) + '_' + flag + ".pth")
                    torch.save(self.ensemble[i].actor.state_dict(), model_actor_path)









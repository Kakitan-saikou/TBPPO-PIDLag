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
from model.mae_E_deterministic import TransformerAgent, TransformerQAgent
from framework.buffer_beifen import Buffer as buffer
import copy




class TD3_BC_ensemble(object):
    def __init__(
            self,
            config,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
            num_nets=1,
            device=None
    ):

        self.device = device
        self.num_nets = config.n_ensemble
        self.L_actor = TransformerAgent(config)
        self.L_actor_target = copy.deepcopy(self.L_actor)
        self.L_critic = TransformerQAgent(config)
        self.L_critic_target = copy.deepcopy(self.L_critic)

        self.reset_optimizer()
        self.memory = buffer(config)

        self.max_action = config.action_max
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.sets = config.n_iff / config.n_ensemble

        self.state_dim = config.obs_space
        self.action_dim = config.action_space
        self.env_num = config.env_num
        self.device = config.device

        self.state_dim = config.obs_space
        self.action_dim = config.action_space

        self.state = torch.zeros(self.env_num, self.max_seq_len, self.state_dim).to(self.device)
        self.action = torch.zeros(self.env_num, self.max_seq_len, self.action_dim).to(self.device)

        self.total_it = 0

    def reset_optimizer(self):
        self.L_actor.reset_optimizer()
        self.L_critic.reset_optimizer()

    def reset_state(self):
        self.state = torch.zeros(self.env_num, self.max_seq_len, self.state_dim).to(self.device)
        self.action = torch.zeros(self.env_num, self.max_seq_len, self.action_dim).to(self.device)


    def ensemble_eval_select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        a = None
        for en_index in range(self.num_nets):
            _a = self.L_actor[en_index](state).cpu().data.numpy().flatten()
            if en_index == 0:
                a = _a
            else:
                a += _a
        a = a / self.num_nets
        return a

    def ensemble_expl_select_action(self, state, trans_parameter):
        state = torch.FloatTensor(state).unsqueeze(1).to(self.device)
        self.state = torch.cat([self.state, state], dim=1)[:, 1:, :]

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # Definitely only used when inferencing
        real_actions = np.zeros((self.num_nets * self.sets, self.action_dim))
        with torch.no_grad():

            actions = self.L_actor.getEnsembleAction(self.state)
            q_list = self.L_critic.getExpiles(self.state, actions)

        for i in range(self.n_iff):
            current_Qs = torch.stack(q_list[i], dim=-1)
            logits = current_Qs
            logits = logits * trans_parameter
            w_dist = torch.distributions.Categorical(logits=logits)
            w = w_dist.sample()
            w = w.squeeze(-1).detach().cpu().numpy()
            real_actions[i, :] = actions[w][i, :]

        return real_actions

    def ensemble_select_action(self, state, trans_parameter):
        state = torch.FloatTensor(state).unsqueeze(1).to(self.device)
        self.state = torch.cat([self.state, state], dim=1)[:, 1:, :]

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # Definitely only used when inferencing
        real_actions = np.zeros((self.num_nets * self.sets, self.action_dim))
        with torch.no_grad():

            actions = self.L_actor.getEnsembleAction(self.state)

        for i in range(self.num_nets * self.sets):
            real_actions[i, :] = actions[i][i, :]

        return real_actions


        #     Q_values = self.L_critic.getEnsembleValue(self.state, self.actions)
        #
        #
        # current_Qs = torch.stack(Q_values, dim=-1)
        # #TODO: Check here (stack)
        # logits = current_Qs
        # logits = logits * trans_parameter
        # w_dist = torch.distributions.Categorical(logits=logits)
        # w = w_dist.sample()
        # w = w.squeeze(-1).detach().cpu().numpy()
        # action = actions[w]
        #
        # return action
    def train_offline(self, batch_size=256, t=None, Utd=None):
        self.total_it += 1

        # Sample replay buffer
        offline_batch_size = batch_size * Utd

        offline_state, offline_action, offline_next_state, offline_reward, offline_not_done = self.memory.sample_offline(
            offline_batch_size)


    def train_online(self, batch_size=256, t=None, Utd=None):
        self.total_it += 1

        # Sample replay buffer
        online_batch_size = batch_size * Utd
        offline_batch_size = batch_size * Utd

        online_state, online_action, online_next_state, online_reward, online_not_done = self.memory.sample(
            online_batch_size)
        offline_state, offline_action, offline_next_state, offline_reward, offline_not_done = self.memory.sample_offline(
            offline_batch_size)

        for i in range(Utd):
            state = torch.concat(
                [online_state[batch_size * i:batch_size * (i + 1)], offline_state[batch_size * i:batch_size * (i + 1)]])
            action = torch.concat([online_action[batch_size * i:batch_size * (i + 1)],
                                   offline_action[batch_size * i:batch_size * (i + 1)]])
            next_state = torch.concat([online_next_state[batch_size * i:batch_size * (i + 1)],
                                       offline_next_state[batch_size * i:batch_size * (i + 1)]])
            reward = torch.concat([online_reward[batch_size * i:batch_size * (i + 1)],
                                   offline_reward[batch_size * i:batch_size * (i + 1)]])
            not_done = torch.concat([online_not_done[batch_size * i:batch_size * (i + 1)],
                                     offline_not_done[batch_size * i:batch_size * (i + 1)]])

            with torch.no_grad():
                #TODO: Here correct for enabling ensembles
                # Select action according to policy and add clipped noise
                noise = (
                        torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action = (
                        self.L_actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.L_critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.L_critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.L_critic.optimizer.zero_grad()
            critic_loss.backward()
            self.L_critic.optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.L_critic.parameters(),
                                           self.L_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            actor_loss = 0.0
            a_loss_list = self.L_critic.Q1(state, self.L_actor(state))
            for en_index in range(self.num_nets):
                # Compute TD3 actor losse
                actor_loss -= a_loss_list[en_index].mean()

                # Optimize the actor
            actor_loss /= self.num_nets
            self.L_actor.optimizer.zero_grad()
            actor_loss.backward()
            self.L_actor.optimizer.step()

            for param, target_param in zip(self.L_actor[en_index].parameters(),
                                           self.L_actor_target[en_index].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



        return current_Q1

    def load(self, policy_file, file_name):
        for en_index in range(self.num_nets):
            self.L_critic[en_index].load_state_dict(
                torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_critic", map_location=self.device))
            self.L_critic_optimizer[en_index].load_state_dict(
                torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_critic_optimizer",
                           map_location=self.device))
            self.L_critic_target[en_index] = copy.deepcopy(self.L_critic[en_index])

            self.L_actor[en_index].load_state_dict(
                torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_actor", map_location=self.device))
            self.L_actor_optimizer[en_index].load_state_dict(
                torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_actor_optimizer",
                           map_location=self.device))
            self.L_actor_target[en_index] = copy.deepcopy(self.L_actor[en_index])
            print('model ', en_index, ' load done...')
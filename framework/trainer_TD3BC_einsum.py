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
from model.mae_E_deterministic_einsum import TransformerAgent, TransformerQAgent
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
        config.mode = 'actor'
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
        self.max_seq_len = config.context_len

        self.sets = int(config.n_iff / config.n_ensemble)
        self.n_iff = config.n_iff

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

        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # Definitely only used when inferencing
        real_actions = np.zeros((self.num_nets * self.sets, self.action_dim))
        with torch.no_grad():

            actions = self.L_actor.getEnsembleAction(self.state, Vector=True)
            q_list = self.L_critic.getExpiles(self.state, actions)

            logits = q_list * trans_parameter
            
            w_dist = torch.distributions.Categorical(logits=logits)
            w = w_dist.sample()
            # print('single Q shape', w.shape)
            w = w.detach().cpu().numpy()

            real_actions = actions[w, torch.arange(self.n_iff)].detach().cpu().numpy()

        return real_actions

    def ensemble_select_action(self, state, trans_parameter):
        state = torch.FloatTensor(state).unsqueeze(1).to(self.device)
        self.state = torch.cat([self.state, state], dim=1)[:, 1:, :]

        # Definitely only used when inferencing
        real_actions = np.zeros((self.num_nets * self.sets, self.action_dim))
        with torch.no_grad():

            actions = self.L_actor.getEnsembleAction(self.state)

        for i in range(self.num_nets):
            for j in range(self.sets):
                real_actions[i*self.sets + j, :] = actions[i][i*self.sets + j, :]

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


    
    def train_online(self, batch_size=256, t=None, Utd=1):
        self.total_it += 1

        # Sample replay buffer
        online_batch_size = batch_size * Utd
        offline_batch_size = batch_size * Utd

        online_state, online_action, online_reward, online_done, online_next_state = self.memory.sample(
            online_batch_size)
        # offline_state, offline_action, offline_next_state, offline_reward, offline_not_done = self.memory.sample_offline(
        #     offline_batch_size)

        for i in range(Utd):
            # state = torch.concat(
            #     [online_state[batch_size * i:batch_size * (i + 1)], offline_state[batch_size * i:batch_size * (i + 1)]])
            # action = torch.concat([online_action[batch_size * i:batch_size * (i + 1)],
            #                        offline_action[batch_size * i:batch_size * (i + 1)]])
            # next_state = torch.concat([online_next_state[batch_size * i:batch_size * (i + 1)],
            #                            offline_next_state[batch_size * i:batch_size * (i + 1)]])
            # reward = torch.concat([online_reward[batch_size * i:batch_size * (i + 1)],
            #                        offline_reward[batch_size * i:batch_size * (i + 1)]])
            # not_done = torch.concat([online_not_done[batch_size * i:batch_size * (i + 1)],
            #                          offline_not_done[batch_size * i:batch_size * (i + 1)]])
            state = torch.FloatTensor(online_state).to(self.device)
            print(state.shape)
            action = torch.FloatTensor(online_action).to(self.device)
            reward = torch.FloatTensor(online_reward).to(self.device)
            next_state = torch.FloatTensor(online_next_state).to(self.device)
            not_done = torch.FloatTensor(1 - online_done).to(self.device)


            with torch.no_grad():
                # TODO: Here correct for enabling ensembles
                # Select action according to policy and add clipped noise
                next_mu = self.L_actor_target.getEnsembleAction(next_state, Vector=True)
                next_action = []

                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_mu + noise).clamp(-self.max_action, self.max_action)


                # Compute the target Q value
                target_Q1, target_Q2 = self.L_critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = repeat(reward, 'b t -> n b t', n=self.num_nets) + not_done * self.discount * target_Q
                # for en in range(self.num_nets):
                #      target_Q = torch.min(target_Q1[en], target_Q2[en])
                #      target_Q = reward + not_done * self.discount * target_Q
                #      target_Q_list.append(target_Q)

            # Get current Q estimates
            current_Q1, current_Q2 = self.L_critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # for en in range(self.num_nets):
            #     critic_loss += F.mse_loss(current_Q1[en], target_Q[en]) + F.mse_loss(current_Q2[en], target_Q[en])
            # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.L_critic.optimizer.zero_grad()
            critic_loss.backward()
            self.L_critic.optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.L_critic.parameters(),
                                           self.L_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            
            a_loss_list = self.L_critic.get_q1(state, self.L_actor.getEnsembleAction(state, Vector=True))
            actor_loss = - a_loss_list.mean()
            # for en_index in range(self.num_nets):
            #     # Compute TD3 actor losse
            #     actor_loss -= a_loss_list[en_index].mean()

                # Optimize the actor
            # actor_loss /= self.num_nets
            self.L_actor.optimizer.zero_grad()
            actor_loss.backward()
            self.L_actor.optimizer.step()

            for param, target_param in zip(self.L_actor.parameters(),
                                           self.L_actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return current_Q1
    

    
    def insert_data(self, data):
        for k, v in data.items():
            self.memory.insert(k, v)

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_actor_t_path = os.path.join(base_path, "actor_target_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        model_critic_t_path = os.path.join(base_path, "critic_target_" + str(episode) + ".pth")
        torch.save(self.L_actor.state_dict(), model_actor_path)
        torch.save(self.L_actor_target.state_dict(), model_actor_t_path)
        torch.save(self.L_critic.state_dict(), model_critic_path)
        torch.save(self.L_critic_target.state_dict(), model_critic_t_path)


    
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
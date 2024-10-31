import torch
import numpy as np
import copy, glob
from torch.utils.data import Dataset
# from .utils import padding_obs
# from .feature_translation import translate_local_obs
from tqdm import tqdm
import time
from einops import rearrange, repeat
import pickle

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)): # good method to count in an anti-direction
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class Buffer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.max_ep_len = 80000
        self.state_dim = config.obs_space
        self.action_dim = config.action_space
        self.length = config.context_len
        self.buffer_size = config.buffer_capacity

        self.buffer_dict = {}
        self.property_list = ['states', 'states_next', 'rewards', 'dones', 'actions']
        self.buffer_dict_clear()

        self.offline_buffer_dict = {}
        self.offline_properties = ['states', 'states_next', 'states_actions', 'rewards', 'dones', 'actions', 'returns']
        for item in self.offline_properties:
            self.offline_buffer_dict[item] = list()

        self.size = 0


    def buffer_dict_clear(self):
        for item in self.property_list:
            self.buffer_dict[item] = list()
        self.size = 0

    def insert(self, item_name: str, data: np.ndarray):
        self.buffer_dict[item_name].append(data)
        # self.buffer_dict[item_name][1:self.buffer_size]
        self.size += 1

    def save_offline_data(self, path):
        pickle.dump(self.buffer_dict, open(path, 'wb'))

    def sample(self, batch_size=256, length=30):
        for item in self.property_list:
            if 'state' in item:
                self.buffer_dict[item] = np.array(self.buffer_dict[item]).transpose(1, 0, 2, 3)
            elif 'rewards' in item or 'dones' in item:
                self.buffer_dict[item] = np.array(self.buffer_dict[item]).transpose(1, 0)
            else:
                self.buffer_dict[item] = np.array(self.buffer_dict[item]).transpose(1, 0, 2)
        state = self.buffer_dict['states']
        action = self.buffer_dict['actions']
        reward = self.buffer_dict['rewards']
        done = self.buffer_dict['dones']
        state_next = self.buffer_dict['states_next']
        return state, action, reward, done, state_next

    def sample_offline(self, batch_size=256, length=30):
        ind = np.random.randint(0, self.size, size=batch_size)

        state = self.offline_buffer_dict['states'][:, ind, :, :]
        action = self.buffer_dict['actions'][:, ind, :]
        reward = self.buffer_dict['rewards'][:, ind]
        done = self.buffer_dict['dones'][:, ind]
        state_next = self.buffer_dict['states_next'][:, ind, :, :]
        sa_pair = self.offline_buffer_dict['states_actions'][:, ind, :, :]

        return state, sa_pair, reward, done, state_next

    def cal_return(self, gamma):
        gamma = gamma
        reward_np = np.array(self.offline_buffer_dict['rewards'])
        done_np = np.array(self.offline_buffer_dict['dones'])
        return_np = np.zeros([reward_np.shape[0], reward_np.shape[1]])

        pre_return = np.zeros(reward_np.shape[0])
        for i in reversed(range(reward_np.shape[1])):
            return_np[:, i] = reward_np[:, i] + gamma * pre_return * (1 - done_np[:, i])
            pre_return = return_np[:, i]

        self.offline_buffer_dict['returns'] = return_np

    def load_all_offlines(self, path, gamma):
        #S, S', SA, A, R, D, Rt
        data = pickle.load(open(path, "rb"))
        offline_properties = ['states', 'states_next', 'states_actions', 'rewards', 'dones', 'actions', 'returns']
        for item in offline_properties:
            self.offline_buffer_dict[item] = list()

        for item in self.offline_properties:
            if 'state' in item:
                self.offline_buffer_dict[item] = np.array(data[item]).transpose(1, 0, 2, 3)
            elif 'rewards' in item or 'dones' in item:
                self.offline_buffer_dict[item] = np.array(data[item]).transpose(1, 0)
            elif item == 'actions':
                self.offline_buffer_dict[item] = np.array(data[item]).transpose(1, 0, 2)

        self.cal_return(gamma)
        self.size = self.offline_buffer_dict['states'].shape[0]

        state = self.offline_buffer_dict['states']
        action = self.offline_buffer_dict['actions']
        reward = self.offline_buffer_dict['rewards']
        done = self.offline_buffer_dict['dones']
        state_next = self.offline_buffer_dict['states_next']
        returns = self.offline_buffer_dict['returns']
        sa_pair = self.offline_buffer_dict['states_actions']

        return state, state_next, sa_pair, action, reward, done, returns



    def preprocess_data(self):
        #   timestep * dim
        len_ls = [item['actions'].shape[0] for item in self.data]
        total_len = np.max(len_ls) + self.length - 1
        self.max_traj_len = total_len

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for idx in range(len(self.data)):
            traj = self.data[idx]
            traj_len = len_ls[idx]

            # s, a, r , done, rtg, timestep, mask
            # for the first padding
            # 0 ,0,0, false, final_rtg_padding, false

            # get sequences from dataset
            s.append(traj['observations'].reshape(1, -1, self.state_dim))
            a.append(traj['actions'].reshape(1, -1, self.action_dim))
            r.append(traj['rewards'].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'].reshape(1, -1))
            elif 'terminal' in traj:
                d.append(np.array(traj['terminal']).reshape(1, -1))
            else:
                d.append(traj['dones'].reshape(1, -1))
            timesteps.append(np.arange(traj_len).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            timesteps[-1][timesteps[-1] <= 0] = 0  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'], gamma=1.).reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([np.zeros((1, 1, 1)), rtg[-1]], axis=1)

            # TODO: padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, total_len - tlen, self.state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, total_len - tlen, self.action_dim)) * -0., a[-1]], axis=1) #TODO: why? *-10?
            r[-1] = np.concatenate([np.zeros((1, total_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, total_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, total_len - tlen, 1)), rtg[-1]], axis=1)  # / scale
            timesteps[-1] = np.concatenate([np.zeros((1, total_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, total_len - tlen)), np.ones((1, tlen))], axis=1))
        # print(f"{time.time()}==={batch_idx[0]}", time.time()-start_time)
        self.traj_len = len_ls
        self.state = np.concatenate(s,axis=0)
        self.action = np.concatenate(a,axis=0)
        self.reward = np.concatenate(r,axis=0)
        self.done = np.concatenate(d,axis=0)
        self.rtg = np.concatenate(rtg, axis=0)
        self.timestep = np.concatenate(timesteps, axis=0)
        self.mask = np.concatenate(mask,axis=0)
        ## for cuda mode
        self.state, self.action, self.reward, self.done, self.rtg, self.timesteps, self.mask = self.move_data_to_device(self.state, self.action, self.reward, self.done, self.rtg, self.timestep, self.mask)

    def sample_prepared_data(self, batch_size=256, length=30):
        start_time = time.time()
        # replace set
        replace = False if len(self.data) > batch_size else True
        batch_idx = np.random.choice(np.arange(len(self.data)), size=batch_size, p=self.p, replace=replace)

        mask = np.zeros((len(self.data), self.max_traj_len), dtype=bool)
        # TODO: fix bug if length is short like 30, how can we sample 60 from it?
        for idx in batch_idx:
            while True:
                start_idx = np.random.randint(0, self.traj_len[idx])
                end_idx = start_idx + length
                # since mask is boolean, we should make all span don't cover each other
                if sum(mask[idx, start_idx:end_idx]) ==0:
                    break
            mask[idx, start_idx:end_idx] = True
        #print(f"{time.time()}3==={batch_idx[0]}", time.time() - start_time) # 0.0173
        s = self.state[mask].reshape(batch_size, length, -1)
        #print(f"{time.time()}2==={batch_idx[0]}", time.time() - start_time) # 0.0231
        a = self.action[mask].reshape(batch_size, length, -1)
        r = self.reward[mask].reshape(batch_size, length, -1)
        d = self.done[mask].reshape(batch_size, length, -1)
        #print(f"{time.time()}1==={batch_idx[0]}", time.time() - start_time) # 0.0353
        rtg = np.array(0)#self.rtg[mask].reshape(batch_size, length, -1)
        timesteps = self.timestep[mask].reshape(batch_size, length, -1)
        mask = self.mask[mask].reshape(batch_size, length, -1)
        #print(f"{time.time()}==={batch_idx[0]}", time.time()-start_time) #0.0370
        return s, a, r, d, rtg, timesteps, mask

    def sample_prepared_data_cuda(self, batch_size=256, length=30):
        start_time = time.time()
        # replace set
        replace = False if len(self.data) > batch_size else True
        batch_idx = np.random.choice(np.arange(len(self.data)), size=batch_size, p=self.p, replace=replace)

        mask = np.zeros((len(self.data), self.max_traj_len), dtype=bool)
        # TODO: fix bug if length is short like 30, how can we sample 60 from it?
        for idx in batch_idx:
            while True:
                start_idx = np.random.randint(0, self.traj_len[idx])
                end_idx = start_idx + length
                # since mask is boolean, we should make all span don't cover each other
                if sum(mask[idx, start_idx:end_idx]) ==0:
                    break
            mask[idx, start_idx:end_idx] = True
        mask = torch.from_numpy(mask).to(device=self.device)
        #print(f"{time.time()}3==={batch_idx[0]}", time.time() - start_time) # 0.05
        s = self.state[mask].reshape(batch_size, length, -1)
        a = self.action[mask].reshape(batch_size, length, -1)
        #print(f"{time.time()}2==={batch_idx[0]}", time.time() - start_time) #0.20
        r = self.reward[mask].reshape(batch_size, length, -1)
        d = self.done[mask].reshape(batch_size, length, -1)
        mask = self.mask[mask].reshape(batch_size, length, -1)
        #TODO: fix the implementation
        rtg = np.array(0)#self.rtg[mask].reshape(batch_size, length, -1)
        timesteps = np.array(0)#self.timestep[mask].reshape(batch_size, length, -1)

        #print(f"{time.time()}1==={batch_idx[0]}", time.time()-start_time) #0.24
        return s, a, r, d, rtg, timesteps, mask

    def context_eval_sample(self, idx=256, length=30):
        length = self.traj_len[idx]
        mask = np.zeros((length , self.max_traj_len), dtype=bool)
        for i in range(length):
            mask[i,i:i+self.length] = True

        s = repeat(self.state[idx,:], 't d -> b t d', b = length)
        s = s[mask].reshape(-1, self.length, s.shape[2])

        a = repeat(self.action[idx, :], 't d -> b t d', b=length)
        a = a[mask].reshape(-1, self.length, a.shape[2])

        r = repeat(self.reward[idx, :], 't d -> b t d', b=length)
        r = r[mask].reshape(-1, self.length, r.shape[2])

        # TODO: fix the implementation
        d = np.array(0)  # self.done[mask].reshape(batch_size, length, -1)
        mask = np.array(0)  # self.mask[mask].reshape(batch_size, length, -1)
        rtg = np.array(0)  # self.rtg[mask].reshape(batch_size, length, -1)
        timesteps = np.array(0)  # self.timestep[mask].reshape(batch_size, length, -1)

        return  s, a, r, d, rtg, timesteps, mask

    def sample_data(self, batch_size=256, length=30):
        return self.sample_prepared_data_cuda(batch_size, length)#self.sample_start_data(batch_size, length)

    def sample_start_data(self, batch_size=256, length=30):
        start_time = time.time()
        np.random.seed()
        batch_idx = np.random.choice(np.arange(len(self.data)), size=batch_size, p=self.p)
        #print(f"{time.time()}==start:{batch_idx[0]}")
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for idx in batch_idx:
            traj = self.data[idx]
            end_idx = np.random.randint(0, len(traj['observations']) - 1)+1
            start_idx = end_idx-length if end_idx-length>0 else 0
            # get sequences from dataset
            s.append(traj['observations'][start_idx:end_idx].reshape(1, -1, self.state_dim))
            a.append(traj['actions'][start_idx: end_idx].reshape(1, -1, self.action_dim))
            r.append(traj['rewards'][start_idx: end_idx].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][start_idx: end_idx].reshape(1, -1))
            else:
                d.append(traj['dones'][start_idx: end_idx].reshape(1, -1))
            timesteps.append(np.arange(end_idx-s[-1].shape[1], end_idx ).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            timesteps[-1][timesteps[-1] <= 0] = 0  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][start_idx:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([np.zeros((1, 1, 1)), rtg[-1]], axis=1)

            # TODO: padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, length - tlen, self.state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, length - tlen, self.action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, length - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, length - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, length - tlen, 1)), rtg[-1]], axis=1)  # / scale
            timesteps[-1] = np.concatenate([np.zeros((1, length - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, length - tlen)), np.ones((1, tlen))], axis=1))
        #print(f"{time.time()}==={batch_idx[0]}", time.time()-start_time)
        return s, a, r, d, rtg, timesteps, mask

    def move_data_to_device(self, s, a, r, d, rtg, timesteps, mask):
        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)

        # s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        # a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        # r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        # d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        # rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        # timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        # mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, timesteps, mask

    def sample_end_data(self, batch_size=256, length=30):
        batch_idx = np.random.choice(np.arange(len(self.data)), size=batch_size, p=self.p)
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for idx in batch_idx:
            traj = self.data[idx]
            start_idx = np.random.randint(0, len(traj['observations']) - 1)

            # get sequences from dataset
            s.append(traj['observations'][start_idx:start_idx + length].reshape(1, -1, self.state_dim))
            a.append(traj['actions'][start_idx:start_idx + length].reshape(1, -1, self.action_dim))
            r.append(traj['rewards'][start_idx:start_idx + length].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][start_idx:start_idx + length].reshape(1, -1))
            else:
                d.append(traj['dones'][start_idx:start_idx + length].reshape(1, -1))
            timesteps.append(np.arange(start_idx, start_idx + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            timesteps[-1][timesteps[-1] <= 0] = 0  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][start_idx:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # TODO: padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, length - tlen, self.state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, length - tlen, self.action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, length - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, length - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, length - tlen, 1)), rtg[-1]], axis=1) #/ scale
            timesteps[-1] = np.concatenate([np.zeros((1, length - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, length - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, timesteps, mask


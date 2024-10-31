import random
import torch
import numpy as np
import os
import yaml
from types import SimpleNamespace as SN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ConfigDict(dict):
    __getattr__= dict.__getitem__
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

    def washDictChange(self):
        key_ls = list(self.keys())
        for name in key_ls:
            if '.' in name: # means this is a dict item
                sub_name_ls = name.split('.')
                value = self.get(name)
                father = self
                for i in range(len(sub_name_ls)-1):
                    sub_name = sub_name_ls[i]
                    if sub_name not in father:
                        father[sub_name] = {}
                    father = father.get(sub_name)

                father[sub_name_ls[-1]] = value


def load_config(dir):
    file = open(os.path.join(dir), "r")
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict


def get_paras_from_dict(config_dict):
    dummy_dict = config_reformat(config_dict)
    args = SN(**dummy_dict)
    return args

def config_reformat(my_dict):
    dummy_dict = {}
    for k, v in my_dict.items():
        if type(v) is dict:
            for k2, v2 in v.items():
                dummy_dict[k2] = v2
        else:
            dummy_dict[k] = v
    return dummy_dict


from pathlib import Path
import os

def make_logpath(game_name, algo):
    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / Path('./models') / game_name.replace('-', '_') / algo

    log_dir = base_dir / Path('./models/config_training')
    if not log_dir.exists():
        os.makedirs(log_dir)

    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run

    print(f"----------------------{run_dir}----------------------")

    return run_dir, log_dir

def fake_run(length, transition, iql, policy, buffer, batch=64, single_batch=1, starting_from_zero=False, adv_type='IQL', train=False):
    if starting_from_zero:
        if single_batch == 1:
            obs = np.zeros((batch, policy.max_seq_len, policy.state_dim))
            rolling_a = np.zeros((batch, policy.max_seq_len, policy.action_dim))
        else:
            obs = np.zeros((batch, single_batch, policy.max_seq_len, policy.state_dim))
            rolling_a = np.zeros((batch, single_batch, policy.max_seq_len, policy.action_dim))

    else:
        obs,_, rolling_a, _,_,_,_,_,_ = buffer.sample_offline(batch_size=batch, length=single_batch)
    q_list = []
    G_list = []

    print(obs.shape)
    transition.reset_with_obs(obs)
    transition.dynamics.eval()

    # N_batch * action_dim
    new_rolling_a = policy.initial_action(obs, rolling_a, train=train)

    next_obs, reward, done, _ = transition.step(obs, new_rolling_a)

    rolling_a = new_rolling_a

    if adv_type == 'IQL':
        ss = torch.FloatTensor(obs).to(iql.device)
        aa = torch.FloatTensor(rolling_a).to(iql.device)

        if iql._is_double_q:
            q = iql.minQ(ss, aa).detach().cpu().numpy()
        else:
            q = iql._Q.getValue(ss, aa).detach().cpu().numpy()

    elif adv_type == 'TD0':
        ss = torch.FloatTensor(next_obs).to(iql.device)
        next_V = iql._value.getValue(ss).squeeze(-1)
        q = reward + iql._gamma * next_V.detach().cpu().numpy()

    obs = next_obs

    q_list.append(q)
    G_list.append(reward)

    for _ in range(length-1):
        new_rolling_a = policy.offline_action(obs, rolling_a, train=train)
        next_obs, reward, done, _ = transition.step(obs, new_rolling_a)

        rolling_a = new_rolling_a

        if adv_type == 'IQL':
            ss = torch.FloatTensor(obs).to(iql.device)
            aa = torch.FloatTensor(rolling_a).to(iql.device)

            if iql._is_double_q:
                q = iql.minQ(ss, aa).detach().cpu().numpy()
            else:
                q = iql._Q.getValue(ss, aa).detach().cpu().numpy()

        elif adv_type == 'TD0':
            ss = torch.FloatTensor(next_obs).to(iql.device)
            next_V = iql._value.getValue(ss).squeeze(-1)
            q = reward + iql._gamma * next_V.detach().cpu().numpy()

        obs = next_obs

        q_list.append(q)
        G_list.append(reward)

    return np.mean(q_list), np.mean(G_list)




import time

from env.flow_field_env import fake_env, foil_env
import argparse
import json
from model.online_gpt_model import GPTConfig, GPT
from framework.utils import set_seed, ConfigDict, make_logpath
from framework.logger import LogServer, LogClient
# from framework.buffer import OnlineBuffer
from framework.IQL import IQL_Q_V
from framework.trainer_TD3BC_einsum import TD3_BC_ensemble
from framework import utils
from framework.normalization import RewardScaling, Normalization
from model.sin_policy import SinPolicy
# from model.parametric_policy import parametric_agent
from datetime import datetime, timedelta
import wandb
import numpy as np
from tqdm import tqdm
import gym
import pickle
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt

from env.IFF_env_2 import ServoControlEnv

def prepare_arguments():
    parser = argparse.ArgumentParser()
    # Required_parameter
    parser.add_argument("--config-file", "--cf", default="./config/config.json",
                        help="pointer to the configuration file of the experiment", type=str)
    args, unknown = parser.parse_known_args()
    args.config = json.load(open(args.config_file, 'r', encoding='utf-8'))
    print(args.config)

    ### set seed
    if args.config['seed'] == "none":
        args.config['seed'] = datetime.now().microsecond % 65536
        args.seed = args.config['seed']
    set_seed(args.seed)

    # reconfig some parameter
    args.name = f"[Debug]FishRL_PPO_transformer_{args.seed}"
    # v4 actionDevide_eta_actionRange

    # wandb remote logger, can mute when debug
    mute = True
    remote_logger = LogServer(args, mute=mute)  # open logging when finish debuging
    remote_logger = LogClient(remote_logger)

    # for the hyperparameter search
    if mute:
        new_args = args
    else:
        new_args = remote_logger.server.logger.config if not mute else args
        new_args = ConfigDict(new_args)
        new_args.washDictChange()
    new_args.remote_logger = remote_logger
    return new_args, remote_logger


### load config AND prepare logger
args, remote_logger = prepare_arguments()
config = args.config
dir = "config/ttd3.yaml"
config_dict = utils.load_config(dir)
paras = utils.get_paras_from_dict(config_dict)
print("local:", paras)
# wandb.init(project="Fish_0715", entity="krhkk")
wandb.init(project="Fish_0816", entity="krhkk", config=paras, name=args.name, mode="disabled")# ("disabled" or "online")
paras = utils.get_paras_from_dict(wandb.config)

# paras = utils.get_paras_from_dict(paras)

print("finetune", paras)
run_dir, log_dir = make_logpath('iff', paras.algo)
### start env
#env = foil_env(paras)
# num_envs = 10
num_envs = paras.n_iff
# env = gym.vector.make('foil-v0', num_envs=num_envs, config=paras)
# env = gym.vector.make('fish-v0', num_envs=num_envs, config=paras, local_port=8686)
# env = foil_env(paras, local_port=8686)
# obs = env.reset()
#paras.action_space, action_dim = env.envs[0].action_dim, env.envs[0].action_dim
#paras.obs_space, observation_dim = env.envs[0].observation_dim, env.envs[0].observation_dim
# paras.action_space, action_dim = env.single_action_space.shape[0], env.single_action_space.shape[0]
# paras.obs_space, observation_dim = env.single_observation_space.shape[0], env.single_observation_space.shape[0]
paras.action_space, action_dim = 3, 3
paras.obs_space, observation_dim = 9, 9
paras.device = "cuda:0" if torch.cuda.is_available() else "cpu"
paras.env_num = num_envs

# mid_values = [
#     187, 146, 185, 180, 185, 190, 190, 192, 180, 182, 180, 185,
#     184, 180, 178, 182, 176, 178, 182, 180, 57, 184, 182, 186
#     ]

# mid_values = [
#     187, 146, 180, 
#         180, 185, 190, 
#         190, 192, 180, 
#         182, 180, 185, 
#         184, 180, 178, 
#         182, 176, 185, 
#         182, 180, 65,  
#         184, 182, 186  
# ]
mid_values = [
187, 146, 190, 
186, 185, 196, 
190, 174, 173, 
182, 180, 184, 
184, 182, 173, 
182, 175, 183, 
182, 179, 78,  
184, 183, 195
]


env = ServoControlEnv(paras)
env.load_midvalue(mid_values)

state_norm_flag = False
state_norm = Normalization((num_envs, paras.obs_space))
reward_norm = RewardScaling((num_envs), 0.99)
ret = []
# obs = env.reset()
# obs = state_norm(obs) if state_norm_flag else obs
done = [False] * num_envs
epsoide_length = 0
epsoide_num = 0
buffer_new_data = 0
Gt_best = -100000

train_cnt = 0
train_target = 1

adjust_cnt = 0
adjust_target = 15

environment_steps = int((1/paras.motor_velocity - paras.steady_time) * paras.control_frequency)
# environment_steps = 10

agent = TD3_BC_ensemble(paras)
if paras.load_actor:
    agent.load(paras.actor_path)
if paras.load_critic:
    agent.load_v(paras.critic_path)

agent.reset_optimizer()  # change the mode from offline to online

print('paras batch', paras.batch_size)
for i in tqdm(range(config['epochs'])):
    # rollout in env  -  rollout()
    print('I:', i)
    trans_para = 1
    epsoide_length, Gt, time_cost = 0, 0.0, 0
    ct_ls, cp_ls, fx_ls, dt_ls = [], [], [], []
    agent.reset_state()
    obs = np.ones((8, 9))
    obs = state_norm(obs) if state_norm_flag else obs
    #Fake choosing
    action = agent.ensemble_expl_select_action(obs, trans_parameter=trans_para)
    info = [{}]
    info[0]['dt'] = 0

    obs = env.reset()
    #TODO: Change for debugging
    # obs = np.ones((8, 9))
    obs = state_norm(obs) if state_norm_flag else obs
    done = [False] * paras.n_iff

    #TODO: Add functions testing the sensors
    start_time = time.time()
    remaining_time = 6.0
    execution_time = time.time() - start_time

    while not any(done) and epsoide_length < environment_steps and execution_time < remaining_time:
        print('Ep step:',epsoide_length,'cost time:', (time.time()-env.collect_time))
        tc1 = time.time()
        with torch.no_grad():
            action = agent.ensemble_expl_select_action(obs, trans_parameter=trans_para)
            # action = agent.ensemble_select_action(obs, trans_parameter=trans_para)
        cost = time.time() - tc1
        print('cost', cost)
        # print(action.shape)

        b_t = 0.042 - cost

        next_obs, r, d, _ = env.step(action, break_t=b_t)
        # next_obs = np.ones((8, 9))
        # r = np.zeros(8)
        # d = [False] * paras.n_iff
        #TODO: Change for debugging


        next_state = next_obs.reshape(next_obs.shape[0], 1, next_obs.shape[1])
        next_state = np.concatenate([agent.state.cpu().detach().numpy(), next_state], axis=1)[:, 1:, :]
        agent.insert_data({'states': agent.state.cpu().detach().numpy(), 'actions': action, 'rewards': r,
                           'states_next': next_state, 'dones': d})

        Gt += r[0]
        # print('steps:', epsoide_length, 'rshape:', r)

        obs = next_obs
        epsoide_length += 1
        execution_time = time.time() - start_time
        env.rate.sleep()
        

    env.save(i, save_full_data=True)
    ct_avg = env.average_Ct
    cl_avg = env.average_Cl
    train_cnt += 1
    adjust_cnt += 1

    train_start_time = time.time()
    if train_cnt >= train_target:
        agent.train_online(8)
        agent.memory.buffer_dict_clear()
        # wandb.log({"Gt": Gt, "length": epsoide_length, "episode_num": i, "avg_reward": Gt / epsoide_length,
        #           "ct": ct_avg, 'cl': cl_avg, "policy_entropy": agent.actor.entropy, "clip_frac": agent.clipfrac,
        #            "approxkl": agent.approxkl, "critic_loss": agent.critic_loss, "actor_loss": agent.actor_loss})

        print("epoch:", i, "length:", epsoide_length, "G:", Gt)
        print("action:", action)

        if Gt > Gt_best and i > 10:
            Gt_best = Gt
            agent.save(run_dir, "best")
            print("save best model")
        if i % 200 == 0:
            agent.save(run_dir, i)

    train_time = time.time() - train_start_time
    env.refresh(train_time)
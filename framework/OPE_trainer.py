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
from model.dynamic_model import DynamicModel
from framework.buffer_beifen import Buffer as buffer
from typing import Dict, List, Union, Tuple, Optional
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR

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
        torch.nn.init.xavier_uniform_(m.weight, gain=1e-2)
        # torch.nn.init.constant_(m.bias, 1e-2)


def update_params(optim, loss, clip=False, param_list=False, retain_graph=False):
    optim.zero_grad()
    loss.backward()
    if clip is not False:
        for i in param_list:
            nn.utils.clip_grad_norm_(i, clip)
    optim.step()

class Transition(object):
    def __init__(self, args):
        # env parameters
        self.state_dim = args.obs_space
        self.action_dim = args.action_space
        self.env_num = args.env_num
        self.device = args.device
        self.pred_type = args.prediction_type
        self._uncertainty_mode = args.uncertainty_mode


        # model parameters
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.n_embed = args.n_embed
        self.max_seq_len = args.context_len
        self.n_iff = args.n_iff
        self.num_elites = args.num_elites

        self.pe_counts = np.zeros(self.n_iff)
        self.pe = np.zeros(self.n_iff)

        self.stepping_angles = np.zeros((self.n_iff, 3))
        self.lower_values, self.upper_values = -30.0, 30.0

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

        self.dynamics = DynamicModel(args).to(self.device)
        self.d_lr = args.d_lr
        # self.dynamics.apply(weights_init_)

        # define for transformer
        self.state = torch.zeros(self.env_num, self.max_seq_len, self.state_dim).to(self.device)
        self.n_ensemble = args.n_ensemble
        self.with_std = args.with_std

        # define buffer
        self.memory = buffer(args)

    def reset_optimizer(self):
        self.dynamics.reset_optimizer()

    def symlog_clip(self, x, scale):
        return scale * np.sign(x) * np.log((np.abs(x / scale)) + 1.0)

    def double_symlog_clip(self, x, scale, scale2):
        z = self.symlog_clip(x, scale)
        return z - (scale2 * 1.15) * np.log((np.abs((x - z) / scale2)) + 1.0)

    def compute_reward_done(self, state):
        if len(state.shape) == 3:
            Fx = state[:, :, 3]
            # Fy = state[:, 4]
            # Clipped_Fx = np.clip(Fx, -1.0, 2.0)
            # Clipped_Fy = np.clip(Fy, -1.2, 2.0)
            Clipped_Fx = self.double_symlog_clip(Fx, 8.0, 16.0)
            # Clipped_Fy = Fy
            # reward = np.zeros(self.n_iff)
            exps = np.exp(self.pe / 5.0) - 1.0  # -e^0
            count_positive = np.sum(self.pe > 1e-3, axis=-1)
            done_condition = np.array((count_positive > 1))
            penalty = np.sum(exps, axis=-1)

            self.pe_counts += done_condition
            self.pe_counts *= done_condition

            penalty *= np.array(self.pe_counts > 3)
            penalty += np.array(self.pe_counts > 12) * penalty
            penalty += np.array(self.pe_counts > 24) * penalty

            done = np.zeros((state.shape[0], state.shape[1]))
        else:
            Fx = state[:, 3]
            # Fy = state[:, 4]
            # Clipped_Fx = np.clip(Fx, -1.0, 2.0)
            # Clipped_Fy = np.clip(Fy, -1.2, 2.0)
            Clipped_Fx = self.symlog_clip(Fx, 1.0)
            # Clipped_Fy = Fy
            # reward = np.zeros(self.n_iff)
            exps = np.exp(self.pe / 5.0) - 1.0  # -e^0

            count_positive = np.sum(self.pe > 1e-3, axis=-1)
            done_condition = np.array((count_positive > 1))
            penalty = np.sum(exps, axis=-1)

            self.pe_counts += done_condition
            self.pe_counts *= done_condition

            penalty *= np.array(self.pe_counts > 3)
            penalty += np.array(self.pe_counts > 12) * penalty
            penalty += np.array(self.pe_counts > 24) * penalty

            done = np.zeros(state.shape[0])

        reward = 1.5 * Clipped_Fx - 0.05 * penalty
        return reward * 1.0, done

    def reset_with_obs(self, obs):
        if len(obs.shape) == 4:
            self.pe_counts = np.zeros((obs.shape[0], obs.shape[1]))
        else:
            self.pe_counts = np.zeros(obs.shape[0])


    def multiple_step(self, obs, action_list):
        s_next_list = []
        r_list = []
        for action in action_list:
            next_pred, next_r, _, _ = self.step(obs, action)
            s_next_list.append(next_pred)
            r_list.append(next_r)

        return s_next_list, r_list


    @torch.no_grad()
    def step(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            out_with_context=True,
            restrict_angles=True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        # if out_with_context, out context len, else single one s'
        Tobs = torch.FloatTensor(obs).to(self.device)
        Taction = torch.FloatTensor(action).to(self.device)
        mean, logvar = self.dynamics(Tobs, Taction, train=out_with_context)

        mean = mean.cpu().numpy()

        if self.pred_type == 'delta':
            mean += obs

        if self.with_std:
            logvar = logvar.cpu().numpy()
            std = np.sqrt(np.exp(logvar))
            e_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)
        else:
            e_samples = mean

        if self.n_ensemble > 1:
            if len(e_samples.shape) == 5:
                n_e, b, t, c, d = e_samples.shape
                model_idxes = np.random.choice(self.dynamics.elites.data.cpu().numpy(), size=b)
                samples = e_samples[model_idxes, np.arange(b)]
                re_flag = True
            else:
                n_e, b, c, d = e_samples.shape
                model_idxes = np.random.choice(self.dynamics.elites.data.cpu().numpy(), size=b)
                samples = e_samples[model_idxes, np.arange(b)]
                re_flag = False
        else:
            if len(e_samples.shape) == 4:
                # print(e_samples.shape)
                re_flag = True
            else:
                # print(e_samples.shape)
                re_flag = False
            samples = e_samples

        if re_flag==True:
            old_angles = obs[:, :, -1, 0:3]
            new_angles = old_angles + action[:, :, -1, :] * 12.0 * np.pi / 180.0

            stepping_angles = np.clip(new_angles, self.lower_values, self.upper_values)
            self.pe = np.abs(stepping_angles - new_angles)

            next_obs = samples[:, :, -1, :]
            if restrict_angles:
                next_obs[:, :, 0:3] = stepping_angles
            full_next = np.concatenate((obs[:, :, 1:, :], np.expand_dims(next_obs, axis=2)), axis=2)

            reward, done = self.compute_reward_done(next_obs)
            info = {}
            info["raw_reward"] = reward
        else:
            # print(re_flag, obs.shape, e_samples.shape, action.shape)
            old_angles = obs[:, -1, 0:3]
            new_angles = old_angles + action[:, -1, :] * 12.0 * np.pi / 180.0

            stepping_angles = np.clip(new_angles, self.lower_values, self.upper_values)
            self.pe = np.abs(stepping_angles - new_angles)

            next_obs = samples[:, -1, :]
            if restrict_angles:
                next_obs[:, 0:3] = stepping_angles
            full_next = np.concatenate((obs[:, 1:, :], np.expand_dims(next_obs, axis=1)), axis=1)

            reward, done = self.compute_reward_done(next_obs)
            info = {}
            info["raw_reward"] = reward

        if out_with_context:
            next_obs = full_next

        return next_obs, reward, done, info

    @torch.no_grad()
    def compute_model_uncertainty(self, obs: np.ndarray, action: np.ndarray,
                                  uncertainty_mode="aleatoric") -> np.ndarray:
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        if uncertainty_mode == "aleatoric":
            penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
        elif uncertainty_mode == "pairwise-diff":
            next_obses_mean = mean[:, :, :-1]
            next_obs_mean = np.mean(next_obses_mean, axis=0)
            diff = next_obses_mean - next_obs_mean
            penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
        else:
            raise ValueError

        penalty = np.expand_dims(penalty, 1).astype(np.float32)

        return self._penalty_coef * penalty


    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obss = np.array(data["states"]).transpose(1, 0, 2, 3)
        actions = np.array(data["states_actions"]).transpose(1, 0, 2, 3)
        actions = actions[..., -3:]
        next_obss = np.array(data["states_next"]).transpose(1, 0, 2, 3)
        # rewards = data["rewards"]

        if self.pred_type=='delta':
            targets = next_obss - obss
        else:
            targets = next_obss

        print('max in targets', np.max(targets), 'mean in targets', np.mean(targets))

        return obss, actions, targets

    def train(
            self,
            obss, acts, targets,
            save_path,
            # logger: Logger,
            max_epochs: Optional[float] = None,
            max_epochs_since_update: int = 25,
            batch_size: int = 128,
            holdout_ratio: float = 0.25,
            logvar_loss_coef: float = 0.01 #0.01
    ):
        # obss, acts, targets = self.format_samples_for_training(data)
        data_size = obss.shape[0]
        print(obss.shape)
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_obss, train_acts, train_targets = obss[train_splits.indices], acts[train_splits.indices], targets[train_splits.indices]
        holdout_obss, holdout_acts, holdout_targets = obss[holdout_splits.indices], acts[holdout_splits.indices], targets[holdout_splits.indices]

        # holdout_obss, holdout_acts, holdout_targets = holdout_obss[:, 0:512, :, :], holdout_acts[:, 0:512, :, :], holdout_targets[:, 0:512, :, :]

        # self.scaler.fit(train_inputs)
        # train_inputs = self.scaler.transform(train_inputs)
        # holdout_inputs = self.scaler.transform(holdout_inputs)
        # holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        #
        # data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        holdout_loss = 1e10
        holdout_losses = []
        train_losses = []

        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        def shuffle_second_dim(arr):
            idxes = np.arange(arr.shape[1])
            np.random.shuffle(idxes)
            idxes_expanded = np.expand_dims(idxes, tuple(range(arr.ndim)[1:]))

            result = np.take_along_axis(arr, idxes_expanded, axis=1)
            return result

        epoch = 0
        cnt = 0
        history_bst = []
        # logger.log("Training dynamics:")
        while True:
            epoch += 1

            train_loss = self.learn(train_obss, train_acts, train_targets, batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_obss, holdout_acts, holdout_targets)
            # holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            new_holdout_loss = new_holdout_losses
            holdout_losses.append(new_holdout_loss)
            train_losses.append(train_loss)
            # logger.logkv("loss/dynamics_train_loss", train_loss)
            # logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            # logger.set_timestep(epoch)
            # logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            # data_idxes = shuffle_rows(data_idxes)
            # train_obss = shuffle_second_dim(train_obss)
            # train_acts = shuffle_second_dim(train_acts)
            # train_targets = shuffle_second_dim(train_targets)

            print(train_loss)
            print(new_holdout_loss)

            # indexes = []
            # for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
            #     improvement = (old_loss - new_loss) / old_loss
            #     if improvement > 0.01:
            #         indexes.append(i)
            #         holdout_losses[i] = new_loss
            #
            # if len(indexes) > 0:
            #     self.model.update_save(indexes)
            #     cnt = 0
            # else:
            #     cnt += 1
            if np.min(new_holdout_loss) < holdout_loss:
                holdout_loss = np.min(new_holdout_loss)
                history_bst = new_holdout_loss
                cnt = 0
                self.save(save_path)
            else:
                cnt += 1

            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                if self.n_ensemble > 1:
                    self.select_elites(history_bst)

                break


        # indexes = self.select_elites(holdout_losses)
        # self.model.set_elites(indexes)
        # self.model.load_save()
        # self.save(logger.model_dir)
        # self.model.eval()
        # logger.log(
        #     "elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
        return train_losses, holdout_losses

    def sheduled_train(
            self,
            obss, acts, targets,
            save_path,
            # logger: Logger,
            max_epochs: Optional[float] = None,
            max_epochs_since_update: int = 25,
            batch_size: int = 128,
            holdout_ratio: float = 0.25,
            logvar_loss_coef: float = 0.01 #0.01
    ):
        # obss, acts, targets = self.format_samples_for_training(data)
        data_size = obss.shape[0]
        print(obss.shape)
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_obss, train_acts, train_targets = obss[train_splits.indices], acts[train_splits.indices], targets[train_splits.indices]
        holdout_obss, holdout_acts, holdout_targets = obss[holdout_splits.indices], acts[holdout_splits.indices], targets[holdout_splits.indices]

        # holdout_obss, holdout_acts, holdout_targets = holdout_obss[:, 0:512, :, :], holdout_acts[:, 0:512, :, :], holdout_targets[:, 0:512, :, :]

        # self.scaler.fit(train_inputs)
        # train_inputs = self.scaler.transform(train_inputs)
        # holdout_inputs = self.scaler.transform(holdout_inputs)
        # holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        #
        # data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        holdout_loss = 1e10
        holdout_losses = []
        train_losses = []

        self.scheduler = ExponentialLR(self.dynamics.optimizer, gamma=0.95)
        # self.scheduler = CosineAnnealingWarmRestarts(self.dynamics.optimizer, T_0=20, T_mult=2,
        #                                              eta_min=0.01 * self.d_lr)

        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        def shuffle_second_dim(arr):
            idxes = np.arange(arr.shape[1])
            np.random.shuffle(idxes)
            idxes_expanded = np.expand_dims(idxes, tuple(range(arr.ndim)[1:]))

            result = np.take_along_axis(arr, idxes_expanded, axis=1)
            return result

        epoch = 0
        cnt = 0
        history_bst = []
        # logger.log("Training dynamics:")
        while True:
            epoch += 1

            train_loss = self.learn(train_obss, train_acts, train_targets, batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_obss, holdout_acts, holdout_targets)
            # holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            new_holdout_loss = new_holdout_losses
            holdout_losses.append(new_holdout_loss)
            train_losses.append(train_loss)

            print(train_loss)
            print(new_holdout_loss)

            if np.min(new_holdout_loss) < holdout_loss:
                holdout_loss = np.min(new_holdout_loss)
                history_bst = new_holdout_loss
                cnt = 0
                self.save(save_path)
            else:
                cnt += 1

            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                if self.n_ensemble > 1:
                    self.select_elites(history_bst)

                break

            # self.dynamics.scheduler.step()
            if epoch >=15 and epoch<=115:
                self.scheduler.step()


        return train_losses, holdout_losses

    def learn(
            self,
            input_obss: np.ndarray,
            input_acts: np.ndarray,
            targets: np.ndarray,
            batch_size: int = 256,
            logvar_loss_coef: float = 0.01,
            loss_type='logvar'
    ) -> float:
        self.dynamics.train()
        train_size = input_obss.shape[0]
        losses = []

        # for batch_num in range(int(np.ceil(train_size / batch_size))):
        #
        #     inputs_o_batch = input_obss[batch_num * batch_size:(batch_num + 1) * batch_size]
        #     inputs_a_batch = input_acts[batch_num * batch_size:(batch_num + 1) * batch_size]
        #     targets_batch = targets[batch_num * batch_size:(batch_num + 1) * batch_size]

        for index in BatchSampler(RandomSampler(range(train_size)), batch_size, False):
            inputs_o_batch = input_obss[index]
            inputs_a_batch = input_acts[index]
            targets_batch = targets[index]


            # print('oshape', inputs_o_batch.shape, 'ashape', inputs_a_batch.shape)

            inputs_o_batch = torch.FloatTensor(inputs_o_batch).to(self.dynamics.device)
            inputs_a_batch = torch.FloatTensor(inputs_a_batch).to(self.dynamics.device)
            targets_batch = torch.FloatTensor(targets_batch).to(self.dynamics.device)
            if self.n_ensemble > 1:
                targets_batch = repeat(targets_batch, 'b t c d -> b_repeat b t c d', b_repeat=self.n_ensemble)

            mean, logvar = self.dynamics(inputs_o_batch, inputs_a_batch, train=True)

            if self.with_std:
                inv_var = torch.exp(-logvar)

                if self.n_ensemble > 1:
                    mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2, 3, 4))
                    var_loss = logvar.mean(dim=(1, 2, 3, 4))

                    # Only sum over ensembles
                    loss = mse_loss_inv.sum() + var_loss.sum()
                    loss = loss + logvar_loss_coef * self.dynamics.max_logvar.sum() - logvar_loss_coef * self.dynamics.min_logvar.sum()
                else:
                    mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean()
                    var_loss = logvar.mean()
                    loss = mse_loss_inv + var_loss
                    loss = loss + logvar_loss_coef * self.dynamics.max_logvar.sum() - logvar_loss_coef * self.dynamics.min_logvar.sum()

            else:
                if self.n_ensemble > 1:
                    # Only sum over ensembles
                    loss_mse = self.dynamics.mseloss(mean, targets_batch).mean(dim=(1, 2, 3, 4))
                    loss = loss_mse.sum()

                else:
                    loss = self.dynamics.mseloss(mean, targets_batch).mean()

            self.dynamics.optimizer.zero_grad()
            loss.backward()
            self.dynamics.optimizer.step()

            # self.scheduler.step()

            losses.append(loss.item())
        return np.mean(losses)

    @torch.no_grad()
    def validate(self, input_obss: np.ndarray, input_acts: np.ndarray, targets: np.ndarray, batch_size: int = 256) -> float:
        self.dynamics.eval()

        train_size = input_obss.shape[0]
        bts = int(np.ceil(train_size / batch_size))
        if self.n_ensemble > 1:
            val_loss = [0.0] * self.n_ensemble
        else:
            val_loss = 0.0

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_o_batch = input_obss[batch_num * batch_size:(batch_num + 1) * batch_size]
            inputs_a_batch = input_acts[batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[batch_num * batch_size:(batch_num + 1) * batch_size]

            # print('oshape', inputs_o_batch.shape, 'ashape', inputs_a_batch.shape)

            inputs_o_batch = torch.FloatTensor(inputs_o_batch).to(self.dynamics.device)
            inputs_a_batch = torch.FloatTensor(inputs_a_batch).to(self.dynamics.device)
            targets_batch = torch.FloatTensor(targets_batch).to(self.dynamics.device)
            if self.n_ensemble > 1:
                targets_batch = repeat(targets_batch, 'b t c d -> b_repeat b t c d', b_repeat=self.n_ensemble)

            mean, _ = self.dynamics(inputs_o_batch, inputs_a_batch, train=True)

            if self.n_ensemble > 1:
                # diff = mean - targets_batch
                diff1 = ((mean - targets_batch) ** 2).mean()
                loss = ((mean - targets_batch) ** 2).mean(dim=(1,2,3,4))
                print('loss shape: ', loss.shape, diff1)
            # val_loss = list(loss.cpu().numpy())
                l_loss = list(loss.cpu().numpy())
                for i in range(self.n_ensemble):
                    val_loss[i] += l_loss[i] / bts
            else:
                loss = ((mean - targets_batch) ** 2).mean()
                # val_loss = list(loss.cpu().numpy())
                val_loss += loss.item() / bts

        return val_loss

    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.num_elites)]
        self.dynamics.set_elites(elites)
        return elites




    def save(self, save_path: str) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.dynamics.state_dict(), os.path.join(save_path, "dynamics.pth"))
        # self.scaler.save_scaler(save_path)
        print('dynamics model saved in {}'.format(str(save_path)))

    def load(self, load_path: str) -> None:
        self.dynamics.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.dynamics.device))
        # self.scaler.load_scaler(load_path)
        print('dynamics model loaded from {}'.format(str(load_path)))


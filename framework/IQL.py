import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mae import TransformerAgent, TransformerQAgent
from model.mlp import MLPAgent, MLPQAgent
import os

class doubleQ(nn.Module):
    def __init__(
            self, args
    ) -> None:
        super().__init__()
        args.mode = 'Q'
        if args.mlp == True:
            self._net1 = MLPQAgent(args)
            self._net2 = MLPQAgent(args)
        else:
            self._net1 = TransformerQAgent(args)
            self._net2 = TransformerQAgent(args)

    def forward(self, s, a):
        return self._net1.getValue(s, a), self._net2.getValue(s, a)


class IQL_Q_V(object):
    def __init__(
            self, args
    ) -> None:

        super().__init__()
        self.device = args.device
        self._omega = args.omega
        self._is_double_q = args.is_double_q
        self.sa_type = args.sa_type
        Q_lr = args.q_lr
        v_lr = args.c_lr
        # for q
        if self._is_double_q:
            print('using double q learning')
            self._Q = doubleQ(args).to(self.device)
            self._target_Q = doubleQ(args).to(self.device)
        else:
            args.mode = 'Q'
            if args.mlp == True:
                self._Q = MLPQAgent(args)
                self._target_Q = MLPQAgent(args)
            else:
                self._Q = TransformerQAgent(args).to(self.device)
                self._target_Q = TransformerQAgent(args).to(self.device)


        self._q_optimizer = torch.optim.AdamW(
            self._Q.parameters(),
            lr=Q_lr,
        )

        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = args.target_update_freq
        self._tau = args.tau
        self._gamma = args.gamma
        self._batch_size = args.offline_batch_size
        self.grad_norm = args.grad_norm_clip
        # for v
        args.mode = 'critic'
        if args.mlp == True:
            self._value = MLPQAgent(args).to(self.device)
        else:
            self._value = TransformerAgent(args).to(self.device)
        self._v_optimizer = torch.optim.AdamW(
            self._value.parameters(),
            lr=v_lr,
        )

    def minQ(self, s: torch.Tensor, a: torch.Tensor):
        Q1, Q2 = self._Q(s, a)
        return torch.min(Q1, Q2)

    def target_minQ(self, s: torch.Tensor, a: torch.Tensor):
        Q1, Q2 = self._target_Q(s, a)
        return torch.min(Q1, Q2)

    def expectile_loss(self, loss: torch.Tensor) -> torch.Tensor:
        weight = torch.where(loss > 0, self._omega, (1 - self._omega))
        return weight * (loss ** 2)

    def update(self, s, a, r, s_p, done):
        # s, a, r, s_p, _, done, _, _ = replay_buffer.sample(self._batch_size)
        # s, a, r, s_p, sa, done, _, _ = replay_buffer.sample_offline(self._batch_size)
        # Compute value loss
        with torch.no_grad():
            self._target_Q.eval()
            if self._is_double_q:
                target_q = self.target_minQ(s, a)
            else:
                target_q = self._target_Q.getValue(s, a)
        value = self._value.getValue(s)
        value_loss = self.expectile_loss(target_q - value).mean()

        # update v
        self._v_optimizer.zero_grad()
        value_loss.backward()
        self._v_optimizer.step()

        # Compute critic loss
        with torch.no_grad():
            self._value.eval()
            next_v = self._value.getValue(s_p)
            # print(next_v.shape)

        target_q = r + (1 - done) * self._gamma * next_v
        if self._is_double_q:
            current_q1, current_q2 = self._Q(s, a)
            Q = torch.min(current_q1, current_q2)
            q_loss = ((current_q1 - target_q) ** 2 + (current_q2 - target_q) ** 2).mean()
        else:
            Q = self._Q.getValue(s, a)
            q_loss = F.mse_loss(Q, target_q)

        # update q and target q
        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        print('Q Value:', Q.mean(), Q.max(), Q.min(), ' V: ', value.mean(), value.max(), value.min(), 'R', r.mean())

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return q_loss, value_loss

    def get_advantage(self, s, a):
        #Used for offline train, so in & out by ndarray
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        if self._is_double_q:
            adv = self.minQ(s, a) - self._value.getValue(s)
        else:
            adv = self._Q.getValue(s, a) - self._value.getValue(s)

        adv = adv.squeeze(-1)

        return adv.detach().cpu().numpy()

    def get_multiple_advantage(self, s, a_list, r_list, type='IQL'):
        ensembles = len(a_list)
        adv_list = []

        if type == 'IQL':
            for case in range(ensembles):
                adv = self.get_advantage(s, a_list[case])
                adv_list.append(adv)
        elif type == 'TD0':
            s = torch.FloatTensor(s).to(self.device)
            V_s = self._value.getValue(s).reshape(s.shape[0], s.shape[1])
            for case in range(ensembles):
                s_p = torch.FloatTensor(a_list[case]).to(self.device)
                r = torch.FloatTensor(r_list[case]).to(self.device)
                td_error = r + self._gamma * self._value.getValue(s_p).reshape(s.shape[0], s.shape[1]) - V_s
                adv_list.append(td_error.detach().cpu().numpy())

        return adv_list

    def gae_advantage(self, s, s_p, r, gamma, lamda, numpy_in=True, tensor=False):
        #Used for offline train, so in & out by ndarray
        #All of them not done
        if numpy_in:
            s = torch.FloatTensor(s).to(self.device)
            s_p = torch.FloatTensor(s_p).to(self.device)
            r = torch.FloatTensor(r).to(self.device)

        td_target = r + gamma * self._value.getValue(s_p).reshape(s.shape[0], s.shape[1])
        td_error = td_target - self._value.getValue(s).reshape(s.shape[0], s.shape[1])
        adv = compute_advantage(gamma, lamda, td_error.cpu()).to(self.device)

        if tensor:
            return adv
        else:
            return adv.detach().cpu().numpy()

    def pre_estimate_value(self, s, s_p, r, gamma, lamda, with_q=False):
        # input dim [batch * context_len * dim], batch -> [num_envs * time_steps]

        advantage = self.gae_advantage(s, s_p, r, gamma, lamda, tensor=True)
        state = torch.FloatTensor(s).to(self.device)
        ## like ppo update
        # [trick] : advantage normalization
        advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5)).detach()
        td_lamda_target = advantage + self._value.getValue(state).reshape(state.shape[0], state.shape[1])

        for _ in range(2):
            # sample new action and new action log prob
            # update critic
            critic_loss = F.mse_loss(td_lamda_target.detach(), self._value.getValue(state).reshape(state.shape[0], state.shape[1]))
            # update

            self._value.optimizer.zero_grad()

            critic_loss.backward()
            # trick: clip gradient

            nn.utils.clip_grad_norm_(self._value.parameters(), self.grad_norm)

            self._value.optimizer.step()

        # the fraction of the training data that triggered the clipped objective

        self.critic_loss = critic_loss.item()

        return self.critic_loss

    def pre_estimate_critic(self, s, s_p, a, r, gamma, lamda, scale_factor=1.0):
        # input dim [batch * context_len * dim], batch -> [num_envs * time_steps]

        advantage = self.gae_advantage(s, s_p, r, gamma, lamda, tensor=True)
        state = torch.FloatTensor(s).to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        ## like ppo update
        # [trick] : advantage normalization
        advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5)).detach()
        td_lamda_target = advantage + self._value.getValue(state).reshape(state.shape[0], state.shape[1])

        for _ in range(2):
            # sample new action and new action log prob
            # update critic
            if self._is_double_q:
                current_q1, current_q2 = self._Q(state, action)
                Q = torch.min(current_q1, current_q2)
                q_loss = ((current_q1.squeeze(-1) - td_lamda_target.detach()) ** 2 +
                          (current_q2.squeeze(-1) - td_lamda_target.detach()) ** 2).mean()
            else:
                Q = self._Q.getValue(state, action)
                q_loss = F.mse_loss(Q.squeeze(-1), td_lamda_target.detach())

                # update q and target q
            self._q_optimizer.zero_grad()
            q_loss.backward()
            self._q_optimizer.step()

            print('Q Value:', Q.mean(), Q.max(), Q.min(), 'R', r.mean())

            self._total_update_step += 1

        if self._total_update_step % self._target_update_freq == 0:
            scaled_tau = scale_factor * self._tau
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(scaled_tau * param.data + (1 - scaled_tau) * target_param.data)

        return q_loss.item()

    def save(
            self, save_path: str, save_id: str
    ) -> None:
        train_path = os.path.join(save_path, 'trained_model')
        base_path = os.path.join(train_path, 'IQL')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        q_path = os.path.join(base_path, "Q_" + str(save_id) + ".pth")
        v_path = os.path.join(base_path, "V_" + str(save_id) + ".pth")
        torch.save(self._Q.state_dict(), q_path)
        torch.save(self._value.state_dict(), v_path)
        print('Q and V saved in {}'.format(base_path))

    def load(
            self, q_path: str, v_path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(q_path, map_location=self.device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')
        self._value.load_state_dict(torch.load(v_path, map_location=self.device))
        print('Value parameters loaded')

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
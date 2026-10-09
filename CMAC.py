from CMARL import *
from torch.distributions import Categorical
import torch.nn as nn
import torch
from copy import deepcopy
from params import configs
import numpy as np


class Memory:
    def __init__(self, gamma, gae_lambda):
        """
            the memory used for collect trajectories for PPO training
        :param gamma: discount factor
        :param gae_lambda: GAE parameter for PPO algorithm
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # input variables of DANIEL
        self.fea_j_seq = []  # [T, tensor[B, N, 10]]
        self.op_mask_seq = []  # [N, tensor[sz_b, N, N]]
        self.fea_m_seq = []  # [T, tensor[B, M, 8]]
        self.mch_mask_seq = []  # [N, tensor[sz_b, N, M]]
        self.dynamic_pair_mask_seq = []  # [N, tensor[sz_b, J, M]]
        self.comp_idx_seq = []  # [N, tensor[sz_b, M, M, J]]
        self.candidate_seq = []  # [N, tensor[sz_b, J]]
        self.fea_pairs_seq = []  # [N, tensor[sz_b, J]]

        # other variables
        self.action_seq = []  # action index with shape [N, tensor[sz_b]]
        self.reward_seq = []  # reward value with shape [N, tensor[sz_b]]
        self.val_seq = []  # state value with shape [N, tensor[sz_b]]
        self.done_seq = []  # done flag with shape [N, tensor[sz_b]]
        self.log_probs = []  # legacy joint log probability
        self.role_log_probs = []  # [T, batch, 2]: operation and conditional machine
        self.nested_baselines = []  # [T, batch, 2], from the frozen old policy

    def clear_memory(self):
        self.clear_state()
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]
        del self.role_log_probs[:]
        del self.nested_baselines[:]

    def clear_state(self):
        del self.fea_j_seq[:]
        del self.op_mask_seq[:]
        del self.fea_m_seq[:]
        del self.mch_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.comp_idx_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]

    def push(self, state):
        """
            push a state into the memory
        :param state: the MDP state
        :return:
        """
        self.fea_j_seq.append(state.fea_j_tensor)
        self.op_mask_seq.append(state.op_mask_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.mch_mask_seq.append(state.mch_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)
        self.comp_idx_seq.append(state.comp_idx_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)

    def transpose_data(self):
        """
            transpose the first and second dimension of collected variables
        """
        # 14
        t_Fea_j_seq = torch.stack(self.fea_j_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_op_mask_seq = torch.stack(self.op_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Fea_m_seq = torch.stack(self.fea_m_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_mch_mask_seq = torch.stack(self.mch_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_dynamicMask_seq = torch.stack(self.dynamic_pair_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Compete_m_seq = torch.stack(self.comp_idx_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_seq = torch.stack(self.candidate_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_pairMessage_seq = torch.stack(self.fea_pairs_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)
        t_done_seq = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)

        t_role_logprobs = torch.stack(self.role_log_probs, 0).transpose(0, 1).flatten(0, 1)
        t_baselines = torch.stack(self.nested_baselines, 0).transpose(0, 1).flatten(0, 1)
        return t_Fea_j_seq, t_op_mask_seq, t_Fea_m_seq, t_mch_mask_seq, t_dynamicMask_seq, \
               t_Compete_m_seq, t_candidate_seq, t_pairMessage_seq, \
               t_action_seq, t_reward_seq, t_val_seq, t_done_seq, t_logprobs_seq, \
               t_role_logprobs, t_baselines

    def get_gae_advantages(self):
        """
            Compute terminal-aware GAE and lambda-return value targets.
        :return: flattened, unnormalized GAE and lambda returns [B*T]
        """
        values = self.t_old_val_seq.transpose(0, 1).detach()  # [T, B]
        rewards = torch.stack(self.reward_seq, dim=0).to(values.dtype)
        done = torch.stack(self.done_seq, dim=0).bool()
        advantage = torch.zeros_like(values[0])
        advantage_seq = []
        for i in reversed(range(len(self.reward_seq))):
            next_value = values[i + 1] if i + 1 < len(self.reward_seq) else torch.zeros_like(advantage)
            not_done = (~done[i]).to(values.dtype)
            delta = rewards[i] + self.gamma * next_value * not_done - values[i]
            advantage = delta + self.gamma * self.gae_lambda * not_done * advantage
            advantage_seq.append(advantage)

        gae = torch.stack(advantage_seq[::-1], dim=0)
        targets = gae + values
        return (gae.transpose(0, 1).flatten(0, 1).detach(),
                targets.transpose(0, 1).flatten(0, 1).detach())


class CMAC:
    def __init__(self, config):
        """
            The implementation of PPO algorithm
        :param config: a package of parameters
        """
        self.lr = config.lr
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.tau = config.tau

        self.ploss_coef = config.ploss_coef
        self.vloss_coef = config.vloss_coef
        self.entloss_coef = config.entloss_coef
        self.minibatch_size = config.minibatch_size

        self.policy = CMARL(config)
        self.policy_old = deepcopy(self.policy)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)

    def update(self, memory):
        '''
        :param memory: data used for PPO training
        :return: total_loss and critic_loss
        '''

        t_data = memory.transpose_data()

        # Bootstrap with the frozen state critic as in DANIEL's GAE. The
        # lambda return is then centered on each role's own old-policy,
        # nested conditional baseline rather than replacing those baselines.
        gae, returns = memory.get_gae_advantages()
        role_advantage = gae[:, None] + t_data[10].detach()[:, None] - t_data[14].detach()
        role_advantage = role_advantage.reshape(len(memory.reward_seq[0]), len(memory.reward_seq), 2)
        role_advantage = (role_advantage - role_advantage.mean(1, keepdim=True)) / \
                         role_advantage.std(1, keepdim=True, unbiased=False).clamp_min(1e-3)
        role_advantage = role_advantage.flatten(0, 1)
        value_scale = returns.std(unbiased=False).clamp_min(1.0)
        full_batch_size = len(t_data[8])
        loss_epochs = 0.0
        v_loss_epochs = 0.0
        sample_count = 0

        for _ in range(self.k_epochs):
            # Mix the trajectories before each PPO epoch to avoid batches
            # dominated by consecutive decisions from the same instance.
            order = torch.randperm(full_batch_size, device=t_data[8].device)
            for start in range(0, full_batch_size, self.minibatch_size):
                batch = order[start:start + self.minibatch_size]

                pis, vals, pj, pm, q, _, _ = self.policy(
                                        fea_j=t_data[0][batch],
                                        op_mask=t_data[1][batch],
                                        candidate=t_data[6][batch],
                                        fea_m=t_data[2][batch],
                                        mch_mask=t_data[3][batch],
                                        comp_idx=t_data[5][batch],
                                        dynamic_pair_mask=t_data[4][batch],
                                        fea_pairs=t_data[7][batch],
                                        return_roles=True)
                actions = t_data[8][batch].long()
                jobs = actions // pm.shape[-1]
                machines = actions % pm.shape[-1]
                rows = torch.arange(len(actions), device=actions.device)
                dist_j = Categorical(probs=pj)
                dist_m = Categorical(probs=pm[rows, jobs])
                logs = torch.stack((dist_j.log_prob(jobs), dist_m.log_prob(machines)), -1)
                ratios = torch.exp(logs - t_data[13][batch].detach())
                advantages = role_advantage[batch]
                clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                p_loss = -torch.minimum(ratios * advantages, clipped * advantages).mean()
                targets = returns[batch]
                v_loss = ((vals.squeeze(-1) - targets) / value_scale).square().mean()
                q_loss = ((q[rows, jobs, machines] - targets) / value_scale).square().mean()
                entropy = (dist_j.entropy() + (pj * Categorical(probs=pm).entropy()).sum(-1)).mean()
                loss = (self.vloss_coef * (v_loss + q_loss) + self.ploss_coef * p_loss
                        - self.entloss_coef * entropy)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite multi-agent PPO loss')

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError('Nonfinite multi-agent PPO gradient')
                self.optimizer.step()
                loss_epochs += loss.item() * len(batch)
                v_loss_epochs += v_loss.item() * len(batch)
                sample_count += len(batch)
        # soft update
        for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
            policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)

        return loss_epochs / sample_count, v_loss_epochs / sample_count


def CMAC_initialize():
    cmac = CMAC(config=configs)
    return cmac

from TRAE import *
from sub_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttentionNetwork(nn.Module):
    def __init__(self, config):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim
        self.fea_m_input_dim = config.fea_m_input_dim
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        # mid_dim = [self.embedding_output_dim] * (self.num_dan_layers - 1)
        mid_dim = self.output_dim_per_layer[:-1]

        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i] * m_input_dim_per_layer[i],
                    edge_input_dim=(self.num_heads_OAB[i] if i < self.num_dan_layers - 1 else 1)
                                   * self.output_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: operation features [B, N, 10]
        :param op_mask: false at all real same-job operation pairs [B, N, N]
        :param fea_m: machine features [B, M, 8]
        :param mch_mask: unscheduled operation-machine compatibility [B, N, M]
        :param comp_idx: original DANIEL competition tensor, retained for API compatibility
        :return:
            fea_j.shape = [sz_b, N, output_dim]
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
        """
        for layer in range(self.num_dan_layers):
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask)
            # mch_mask now carries [batch, operation, machine] compatibility
            # for every unscheduled operation, not only current job candidates.
            fea_m = self.mch_attention_blocks[layer](fea_m, fea_j, mch_mask)

        # Job attention includes earlier operations; global context pools only
        # operations that have not yet been assigned to any machine.
        remaining = mch_mask.bool().any(-1).to(fea_j.dtype)
        fea_j_global = (fea_j * remaining[..., None]).sum(1) / remaining.sum(1, keepdim=True).clamp_min(1)
        fea_m_global = fea_m.mean(1)

        return fea_j, fea_m, fea_j_global, fea_m_global


class CMARL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(CMARL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(
            device)
        action_pair_dim = 2 * self.embedding_output_dim + self.pair_input_dim
        critic_pair_dim = 4 * self.embedding_output_dim + self.pair_input_dim
        # The action heads see only candidate operation, machine and pair features.
        # The Q critic retains the pooled context used by the original baselines.
        self.machine_actor = Actor(config.num_mlp_layers_actor, action_pair_dim,
                                   config.hidden_dim_actor, 1).to(device)
        self.operation_actor = Actor(config.num_mlp_layers_actor,
                                     2 * self.embedding_output_dim + self.pair_input_dim,
                                     config.hidden_dim_actor, 1).to(device)
        self.pair_actor = Actor(config.num_mlp_layers_actor, action_pair_dim + 2,
                                config.hidden_dim_actor, 1).to(device)
        self.q_critic = Critic(config.num_mlp_layers_critic, critic_pair_dim,
                               config.hidden_dim_critic, 1).to(device)
        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                dynamic_pair_mask, fea_pairs, return_roles=False):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation features [B, N, 10]
        :param op_mask: false for all real same-job operation pairs [B, N, N]
        :param fea_m: input machine features [B, M, 8]
        :param mch_mask: unscheduled compatible operation-machine pairs [B, N, M]
        :param comp_idx: original competition tensor, retained for the API
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """

        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # Only the Q critic receives globally pooled features. Neither the
        # machine role nor the joint pair head uses them for action scoring.
        action_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized,
                                    fea_pairs), dim=-1)
        critic_pair_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized,
                                         Fea_Gj_input, Fea_Gm_input, fea_pairs), dim=-1)

        valid = ~dynamic_pair_mask.bool()
        if not bool(valid.flatten(1).any(-1).all()):
            raise ValueError("An active FJSP instance has no feasible operation-machine action")
        pair_input = action_feature.reshape(sz_b, J, M, -1)
        machine_logits = self.machine_actor(pair_input).squeeze(-1)
        proposed = F.softmax(machine_logits.masked_fill(~valid, -1e9), dim=-1)
        proposed = proposed * valid.to(proposed.dtype)
        proposed = proposed / proposed.sum(-1, keepdim=True).clamp_min(1e-12)
        machine_message = torch.cat((fea_m.unsqueeze(1).expand(-1, J, -1, -1),
                                     fea_pairs.reshape(sz_b, J, M, -1)), dim=-1)
        # Deliberately stop the message gradient as in CMAC: the operation
        # agent receives the machine agent's feasibility-conditioned proposal.
        message = (proposed.detach()[..., None] * machine_message).sum(2)
        op_logits = self.operation_actor(torch.cat((Fea_j_JC, message), dim=-1)).squeeze(-1)
        joint_input = torch.cat((pair_input, op_logits[:, :, None, None].expand(-1, -1, M, 1),
                                 machine_logits[..., None]), dim=-1)
        joint_logits = op_logits[:, :, None] + machine_logits + self.pair_actor(joint_input).squeeze(-1)
        # Softmax across legal PAIRS, so pair features directly determine the action.
        pi = F.softmax(joint_logits.flatten(1).masked_fill(~valid.flatten(1), -1e9), dim=1)
        pi = pi * valid.flatten(1).to(pi.dtype)
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-12)
        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        if not return_roles:
            return pi, v
        joint = pi.reshape(sz_b, J, M)
        job_prob = joint.sum(-1)
        conditional_mch = joint / job_prob[:, :, None].clamp_min(1e-12)
        conditional_mch = torch.where(job_prob[:, :, None] > 0, conditional_mch,
                                       torch.full_like(conditional_mch, 1.0 / M))
        q = self.q_critic(critic_pair_feature.reshape(sz_b, J, M, -1)).squeeze(-1)
        machine_baseline = (conditional_mch * q).sum(-1)
        operation_baseline = (job_prob * machine_baseline).sum(-1)
        return pi, v, job_prob, conditional_mch, q, operation_baseline, machine_baseline

"""Multi-head operation and machine attention over the remaining FJSP graph."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleOpAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        super().__init__()
        self.W = nn.Parameter(torch.empty(input_dim, output_dim))
        self.a = nn.Parameter(torch.empty(2 * output_dim, 1))
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, h, op_mask):
        # op_mask[b,i,k] is true if operation k is unavailable to query i.
        wh = h @ self.W
        d = wh.shape[-1]
        source = (wh @ self.a[:d]).squeeze(-1)
        target = (wh @ self.a[d:]).squeeze(-1)
        scores = F.leaky_relu(source[:, :, None] + target[:, None, :], 0.2)
        valid = ~op_mask.bool()
        weights = F.softmax(scores.masked_fill(~valid, -1e9), dim=-1)
        weights = self.dropout(weights * valid.to(weights.dtype))
        return (weights @ wh) * valid.any(-1, keepdim=True).to(wh.dtype)


class MultiHeadOpAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.concat = concat
        self.activation = activation
        self.attentions = nn.ModuleList(
            [SingleOpAttnBlock(input_dim, output_dim, dropout_prob) for _ in range(num_heads)])

    def forward(self, h, op_mask):
        results = [head(self.dropout(h), op_mask) for head in self.attentions]
        result = torch.cat(results, dim=-1) if self.concat else torch.stack(results, -1).mean(-1)
        if self.activation is not None:
            result = self.activation(result)
        return result * (~op_mask.bool()).any(-1, keepdim=True).to(result.dtype)


class SingleMchAttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob):
        super().__init__()
        self.W = nn.Parameter(torch.empty(node_input_dim, output_dim))
        self.W_edge = nn.Parameter(torch.empty(edge_input_dim, output_dim))
        self.a = nn.Parameter(torch.empty(2 * output_dim, 1))
        for parameter in (self.W, self.W_edge, self.a):
            nn.init.xavier_uniform_(parameter, gain=1.414)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, h, operation_features, operation_machine_adj):
        # Every unscheduled compatible operation is a machine's neighbor.
        machine = h @ self.W
        operation = operation_features @ self.W_edge
        d = machine.shape[-1]
        scores = F.leaky_relu((machine @ self.a[:d]).squeeze(-1)[:, :, None] +
                               (operation @ self.a[d:]).squeeze(-1)[:, None, :], 0.2)
        valid = operation_machine_adj.transpose(1, 2).bool()
        weights = F.softmax(scores.masked_fill(~valid, -1e9), dim=-1)
        weights = self.dropout(weights * valid.to(weights.dtype))
        # Retain the machine's own representation even if it has no neighbors.
        return machine + weights @ operation


class MultiHeadMchAttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob, num_heads,
                 activation, concat=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.concat = concat
        self.activation = activation
        self.attentions = nn.ModuleList([SingleMchAttnBlock(node_input_dim, edge_input_dim,
                                                             output_dim, dropout_prob)
                                          for _ in range(num_heads)])

    def forward(self, h, operation_features, operation_machine_adj):
        h = self.dropout(h)
        results = [head(h, operation_features, operation_machine_adj) for head in self.attentions]
        result = torch.cat(results, -1) if self.concat else torch.stack(results, -1).mean(-1)
        return result if self.activation is None else self.activation(result)

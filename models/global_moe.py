
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ExpertNetwork(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GlobalMoEModule(nn.Module):

    def __init__(
        self,
        num_views: int,
        view_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dims: List[int],
        expert_output_dim: int,
        router_tau: float = 1.0,
        router_hidden_dim: int = 256,
        attn_dropout: float = 0.0,
        expert_dropout: float = 0.0,
        use_residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_views = num_views
        self.view_dim = view_dim
        self.input_dim = num_views * view_dim
        self.num_experts = int(num_experts)
        self.top_k = max(1, int(top_k))
        self.tau = float(router_tau)
        self.d_r = int(router_hidden_dim)
        self.d_e = int(expert_output_dim)
        self.use_residual = use_residual

        self.z_ln = nn.LayerNorm(self.input_dim)
        self.q_proj = nn.Linear(self.input_dim, self.num_experts * self.d_r)
        self.k_proj = nn.Linear(self.input_dim, self.num_experts * self.d_r)
        self.v_gate_proj = nn.Linear(self.input_dim, self.num_experts)  # B×E×1

        self.router_scale = self.d_r ** -0.5
        self.attn_drop = nn.Dropout(attn_dropout)

        self.experts = nn.ModuleList([
            ExpertNetwork(self.input_dim, expert_hidden_dims, self.d_e, dropout=expert_dropout)
            for _ in range(self.num_experts)
        ])

        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Linear(self.input_dim, self.d_e),
                nn.LayerNorm(self.d_e)
            )
            self.res_scale = nn.Parameter(torch.tensor(0.3))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight); nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight); nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_gate_proj.weight); nn.init.zeros_(self.v_gate_proj.bias)

    @staticmethod
    def _topk_prune(scores: torch.Tensor, k: int) -> torch.Tensor:
        B, E = scores.size()
        if k >= E:
            denom = scores.sum(dim=1, keepdim=True) + 1e-8
            return scores / denom

        topk_vals, topk_idx = torch.topk(scores, k, dim=1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk_idx, 1.0)
        pruned = scores * mask
        pruned = pruned / (pruned.sum(dim=1, keepdim=True) + 1e-8)
        return pruned

    def forward(self, z_views: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        z_concat = torch.cat(z_views, dim=1)  # (B, V * d_feat)
        z_norm = self.z_ln(z_concat)

        B = z_concat.size(0); E = self.num_experts; d_r = self.d_r

        Q = self.q_proj(z_norm).view(B, E, d_r)
        K = self.k_proj(z_norm).view(B, E, d_r)
        V_gate = self.v_gate_proj(z_norm).view(B, E, 1)

        attn_logits = torch.matmul(Q, K.transpose(1, 2)) * self.router_scale  # / sqrt(d_r)
        if self.tau > 0:
            attn_logits = attn_logits / self.tau
        A = F.softmax(attn_logits, dim=-1)
        A = self.attn_drop(A)

        p = torch.matmul(A, V_gate).squeeze(-1)
        gate_logits = p
        if self.tau > 0:
            gate_logits = gate_logits / self.tau
        scores = F.softmax(gate_logits, dim=-1)
        scores_sparse = self._topk_prune(scores, self.top_k)

        expert_outs = [exp(z_norm) for exp in self.experts]
        stacked = torch.stack(expert_outs, dim=1)

        H_glo = torch.sum(stacked * scores_sparse.unsqueeze(-1), dim=1)

        if self.use_residual:
            H_res = self.residual(z_concat)
            H_glo = H_glo + self.res_scale.tanh() * H_res

        return H_glo, scores_sparse, expert_outs

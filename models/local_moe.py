# models/local_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TopKRouter(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int, top_k: int, tau: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts)
        )
        self.top_k = int(top_k)
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_in)
        logits = self.net(x)  # (B, E)
        scores = F.softmax(logits / max(self.tau, 1e-6), dim=1)  # (B, E)
        if self.top_k < scores.size(1):
            topk_vals, topk_idx = scores.topk(self.top_k, dim=1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, topk_idx, 1.0)
            scores = scores * mask
            scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
        return scores  # (B, E)


class ExpertNetwork(nn.Module):
    """Shared expert MLP, reused by all views."""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, d_out)


class LocalMoEModule(nn.Module):
    """
    Local MoE with a SHARED expert bank across views.
    - Shared experts: one ModuleList of experts reused for all views.
    - Per-view routers: each view owns an independent router to obtain view-specific Top-k scores.
    API is kept identical to the previous version:
        forward(...) -> (H_vs, scores_list, expert_outs_list)
            H_vs: List[Tensor] length V, each (B, d_out)
            scores_list: List[Tensor] length V, each (B, E)
            expert_outs_list: List[List[Tensor]] length V, each inner list length E with tensors (B, d_out)
    """
    def __init__(
        self,
        num_views: int,
        input_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dims: List[int],
        expert_output_dim: int,
        router_hidden_dim: int,
        router_tau: float = 1.0
    ):
        super().__init__()
        assert top_k >= 1 and top_k <= num_experts, "top_k must be in [1, num_experts]"
        self.num_views = int(num_views)
        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_output_dim = int(expert_output_dim)

        # per-view routers (view-specific routing)
        self.routers = nn.ModuleList([
            TopKRouter(self.input_dim, self.num_experts, int(router_hidden_dim), self.top_k, float(router_tau))
            for _ in range(self.num_views)
        ])

        # shared expert bank
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(self.input_dim, expert_hidden_dims, self.expert_output_dim)
            for _ in range(self.num_experts)
        ])

    def forward(self, z_views: List[torch.Tensor]):
        """
        z_views: list of length V, each tensor shape (B, d_in=input_dim)
        returns:
            H_vs:            list length V, each (B, d_out)
            scores_list:     list length V, each (B, E)
            expert_outs_list:list length V, each is a list of length E with tensors (B, d_out)
        """
        H_vs: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        expert_outs_list: List[List[torch.Tensor]] = []

        for v, z in enumerate(z_views):
            # (1) per-view routing over the SHARED expert bank
            scores = self.routers[v](z)  # (B, E)

            # (2) dense compute for all shared experts on this view's inputs
            #     (keeps behavior close to the original implementation; only the parameters are now shared)
            expert_outs = [exp(z) for exp in self.shared_experts]  # E × (B, d_out)
            stacked = torch.stack(expert_outs, dim=1)              # (B, E, d_out)

            # (3) sparse aggregation with Top-k-masked scores
            H_v = torch.sum(stacked * scores.unsqueeze(-1), dim=1)  # (B, d_out)

            H_vs.append(H_v)
            scores_list.append(scores)
            expert_outs_list.append(expert_outs)

        return H_vs, scores_list, expert_outs_list

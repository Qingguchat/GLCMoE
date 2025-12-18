import torch
import torch.nn.functional as F
from typing import List


def balance_loss(scores: torch.Tensor) -> torch.Tensor:
    mean_scores = torch.mean(scores, dim=0)  # (E,)
    E = scores.size(1)
    uniform = torch.full_like(mean_scores, 1.0 / E)
    loss = torch.sum((mean_scores - uniform) ** 2)
    return loss


def local_balance_loss(local_scores: List[torch.Tensor]) -> torch.Tensor:
    losses = [balance_loss(scores) for scores in local_scores]
    return torch.stack(losses).mean()


def global_balance_loss(global_scores: torch.Tensor) -> torch.Tensor:
    return balance_loss(global_scores)

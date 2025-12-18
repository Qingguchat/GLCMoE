
import torch
import torch.nn.functional as F
from typing import List


def reconstruction_loss(
    x_views: List[torch.Tensor],
    x_rec_views: List[torch.Tensor]
) -> torch.Tensor:

    num_views = len(x_views)
    if num_views == 0:
        return torch.tensor(0.0)

    loss = 0.0
    for x, x_rec in zip(x_views, x_rec_views):
        loss += F.mse_loss(x_rec, x, reduction='mean')
    return loss / num_views

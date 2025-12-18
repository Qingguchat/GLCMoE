
import torch
import torch.nn.functional as F


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


@torch.no_grad()
def _labels(batch_size: int, device) -> torch.Tensor:
    return torch.arange(batch_size, device=device)


def global_local_infonce_loss(
    global_feat: torch.Tensor,
    local_feat: torch.Tensor,
    temperature: float = 0.40,
    symmetric: bool = True,
    stop_grad: str = "none",
) -> torch.Tensor:

    assert global_feat.shape == local_feat.shape, "global/local dimensions must be consistent"
    g = _normalize(global_feat)
    l = _normalize(local_feat)

    if stop_grad == "global":
        g = g.detach()
    elif stop_grad == "local":
        l = l.detach()

    logits = (g @ l.t()) / max(temperature, 1e-6)  # (B, B)
    y = _labels(g.size(0), g.device)

    loss_g2l = F.cross_entropy(logits, y)
    if symmetric:
        loss_l2g = F.cross_entropy(logits.t(), y)
        return 0.5 * (loss_g2l + loss_l2g)
    else:
        return loss_g2l

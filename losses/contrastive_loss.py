import torch
import torch.nn.functional as F

def infonce_loss(features, temperature=0.5):

    batch_size, num_views, feature_dim = features.shape

    features_flat = features.view(-1, feature_dim)
    features_flat = F.normalize(features_flat, p=2, dim=1)

    sim_matrix = torch.matmul(features_flat, features_flat.T) / temperature

    positive_mask = torch.zeros_like(sim_matrix)
    for i in range(batch_size):
        start_idx = i * num_views
        end_idx = (i + 1) * num_views
        positive_mask[start_idx:end_idx, start_idx:end_idx] = 1.0

    eye_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
    positive_mask = positive_mask - eye_mask

    exp_sim = torch.exp(sim_matrix)
    exp_sim_no_diag = exp_sim * (1.0 - eye_mask)

    positive_sim = exp_sim_no_diag * positive_mask

    pos_sum = positive_sim.sum(dim=1, keepdim=True)
    all_sum = exp_sim_no_diag.sum(dim=1, keepdim=True)

    loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8)).mean()
    return loss

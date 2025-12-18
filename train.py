# train.py
import sys
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

from utils.config import load_config
from data.dataloader import get_dataloader
from models.model import FinalEnhancedModel
from losses.reconstruction_loss import reconstruction_loss
from losses.contrastive_loss import infonce_loss
from losses.expert_balance_loss import (
    local_balance_loss, global_balance_loss,
)
from losses.custom_losses import global_local_infonce_loss
from utils.metrics import compute_metrics


def train(cfg):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')

    train_loader, dataset, num_views, view_dims = get_dataloader(
        cfg.data.mat_file,
        cfg.train.batch_size,
        cfg.train.num_workers,
        shuffle=True,
        seed=getattr(cfg, "seed", 42)
    )

    eval_subset = Subset(dataset, list(range(dataset.num_samples)))
    eval_loader = DataLoader(
        eval_subset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.train.num_workers > 0,
        drop_last=False,
    )

    with torch.no_grad():
        unique_k = int(torch.unique(dataset.labels).numel())
    assert unique_k == cfg.model.num_clusters, \
        f"num_clusters={cfg.model.num_clusters} does not match ground-truth classes={unique_k}. Please fix config."

    model = FinalEnhancedModel(cfg, num_views, view_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg.train.lr),
                                  weight_decay=float(cfg.train.weight_decay))

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    dataset_name = Path(cfg.data.mat_file).stem
    checkpoint_root = "model"
    os.makedirs(checkpoint_root, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_root, f"{dataset_name}.pth")
    best_acc = -1.0

    gl_temp = getattr(cfg.train, "gl_temp", getattr(cfg.model, "temp", 0.40))

    tau0 = float(getattr(cfg.global_moe, "router_tau", 1.0))
    tau_final = float(getattr(cfg.global_moe, "router_tau_final", tau0))
    tau_anneal_epochs = int(getattr(cfg.global_moe, "router_tau_anneal_epochs", 0))
    model.global_moe.tau = tau0  # init

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for views, _ in train_loader:
            views = [v.to(device) for v in views]

            (recon_views, H_loc_list, H_glo, _,
             local_scores, _, global_scores, _) = model(views)

            H_loc_stacked = torch.stack(H_loc_list, dim=1)
            loss_mv_infonce = infonce_loss(H_loc_stacked, temperature=cfg.model.temp)

            H_loc_agg = model.aggregator(H_loc_list)
            z_g = model.proj_glo(H_glo)
            z_l = model.proj_loc(H_loc_agg)
            loss_gl_infonce = global_local_infonce_loss(
                z_g, z_l, temperature=gl_temp, symmetric=True, stop_grad="none"
            )

            loss_bal_loc = local_balance_loss(local_scores)
            loss_bal_glo = global_balance_loss(global_scores)

            loss_rec = reconstruction_loss(views, recon_views)

            total_loss = (cfg.train.rec_weight * loss_rec +
                          cfg.train.ctr_weight * loss_mv_infonce +
                          cfg.train.global_local_weight * loss_gl_infonce +
                          cfg.train.bal_loc_weight * loss_bal_loc +
                          cfg.train.bal_glo_weight * loss_bal_glo)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        model.eval()
        all_H, all_labels = [], []
        with torch.no_grad():
            for views, labels in eval_loader:
                views = [v.to(device) for v in views]
                _, H_loc_list, H_glo, *_ = model(views)
                H_loc_agg = model.aggregator(H_loc_list)
                if H_loc_agg.size(1) != H_glo.size(1):
                    raise ValueError(
                        f"Dim mismatch: H_loc_agg={H_loc_agg.size(1)} vs H_glo={H_glo.size(1)}. "
                        f"Please set model.local_output_dim == global_moe.expert_output_dim."
                    )
                H_comb = 0.5 * (H_loc_agg + H_glo)
                all_H.append(H_comb.cpu())
                all_labels.append(labels)
        all_H = torch.cat(all_H, dim=0).numpy()
        all_H = all_H / (np.linalg.norm(all_H, axis=1, keepdims=True) + 1e-12)
        all_labels = torch.cat(all_labels, dim=0).numpy()

        kmeans = KMeans(n_clusters=cfg.model.num_clusters,
                        n_init=cfg.train.kmeans_n_init,
                        random_state=cfg.seed)
        y_pred = kmeans.fit_predict(all_H)
        metrics = compute_metrics(all_labels, y_pred)
        acc = float(metrics['ACC'])

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)

        scheduler.step(acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{cfg.train.epochs} - AvgLoss: {epoch_loss/len(train_loader):.4f} ")

        if tau_anneal_epochs > 0 and epoch <= tau_anneal_epochs:
            t = epoch / float(tau_anneal_epochs)
            cur_tau = tau0 + (tau_final - tau0) * t
            model.global_moe.tau = float(cur_tau)

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    parser.add_argument('--checkpoint_dir', type=str, help='(ignored) Path to the checkpoint directory.')
    parser.add_argument('--lr', type=float, help='Learning rate override.')
    parser.add_argument('--weight_decay', type=float, help='Weight decay override.')
    parser.add_argument('--global_local_weight', type=float, help='Global-local loss weight override.')
    parser.add_argument('--feature_dim', type=int, help='Feature dimension override for encoders and MoEs.')
    parser.add_argument('--gl_temp', type=float, help='Temperature for GL-InfoNCE.')

    args = parser.parse_args()
    cfg = load_config(args.config)

    config_dir = os.path.dirname(os.path.abspath(args.config))
    if not os.path.isabs(cfg.data.mat_file):
        cfg.data.mat_file = os.path.join(config_dir, cfg.data.mat_file)

    if args.lr:
        cfg.train.lr = args.lr
    if args.weight_decay:
        cfg.train.weight_decay = args.weight_decay
    if args.global_local_weight:
        cfg.train.global_local_weight = args.global_local_weight
    if args.feature_dim:
        cfg.model.feature_dim = args.feature_dim
        cfg.model.local_output_dim = args.feature_dim
        cfg.global_moe.expert_output_dim = args.feature_dim
    if args.gl_temp:
        cfg.train.gl_temp = args.gl_temp

    os.makedirs("model", exist_ok=True)

    train(cfg)

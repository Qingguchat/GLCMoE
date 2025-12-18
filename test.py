
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")
import os
import sys
import csv
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import load_config
from data.dataloader import get_dataloader
from models.model import FinalEnhancedModel
from utils.metrics import compute_metrics


def _abs_mat_path(orig_cfg_path: Path, data_dict: dict):
    mat_file = data_dict.get('mat_file', None)
    if not mat_file:
        return
    if not os.path.isabs(mat_file):
        base_dir = orig_cfg_path.parent
        data_dict['mat_file'] = str((base_dir / mat_file).resolve())


@torch.no_grad()
def _extract_embeddings(model, loader, device):

    model.eval()
    all_H, all_labels = [], []
    for views, labels in loader:
        views = [v.to(device) for v in views]
        _, H_loc_list, H_glo, *_ = model(views)
        H_loc_agg = model.aggregator(H_loc_list)
        if H_loc_agg.size(1) != H_glo.size(1):
            raise ValueError(
                f"Dim mismatch: H_loc_agg={H_loc_agg.size(1)} vs H_glo={H_glo.size(1)}. "
                f"please set model.local_output_dim == global_moe.expert_output_dim。"
            )
        H_comb = 0.5 * (H_loc_agg + H_glo)
        all_H.append(H_comb.cpu())
        all_labels.append(labels)
    H = torch.cat(all_H, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    return H.astype(np.float32, copy=False), y.astype(np.int64, copy=False)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint multiple times by rerunning KMeans (full dataset).")
    ap.add_argument('--config', type=str, required=True, help='Path to config YAML used for training.')
    ap.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth')
    ap.add_argument('--runs', type=int, default=10, help='How many evaluation repeats (default: 10)')
    ap.add_argument('--outdir', type=str, default='eval_runs', help='Directory to save CSV summary')
    ap.add_argument('--kmeans_n_init', type=int, default=None, help='Override KMeans n_init (else use cfg.inference.kmeans_n_init or cfg.train.kmeans_n_init)')
    ap.add_argument('--seed', type=int, default=None, help='Base seed for KMeans random_state (default: cfg.seed or 42)')
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dict = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    _abs_mat_path(cfg_path, cfg_dict.get('data', {}))
    open(cfg_path, 'w', encoding='utf-8').write(yaml.safe_dump(cfg_dict, allow_unicode=True, sort_keys=False))
    cfg = load_config(str(cfg_path))

    device = torch.device(cfg.inference.device if torch.cuda.is_available() else 'cpu')

    train_loader, dataset, num_views, view_dims = get_dataloader(
        cfg.data.mat_file,
        cfg.inference.batch_size,
        cfg.inference.num_workers,
        shuffle=False,
        seed=getattr(cfg, "seed", 42)
    )
    eval_subset = Subset(dataset, list(range(dataset.num_samples)))
    eval_loader = DataLoader(
        eval_subset,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        num_workers=cfg.inference.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.inference.num_workers > 0,
        drop_last=False,
    )

    model = FinalEnhancedModel(cfg, num_views, view_dims).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    H, y_true = _extract_embeddings(model, eval_loader, device)

    n_clusters = int(cfg.model.num_clusters)
    if args.kmeans_n_init is not None:
        n_init = int(args.kmeans_n_init)
    else:
        n_init = int(getattr(cfg.inference, "kmeans_n_init", getattr(cfg.train, "kmeans_n_init", 20)))

    base_seed = args.seed if args.seed is not None else int(getattr(cfg, "seed", 42))

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.outdir, f"eval_{Path(cfg.data.mat_file).stem}_full_{stamp}.csv")

    results = []
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "n_init", "ACC", "NMI", "ARI"])
        for i in range(1, args.runs + 1):
            rs = base_seed + (i - 1)
            km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=rs)
            y_pred = km.fit_predict(H)
            metrics = compute_metrics(y_true, y_pred)
            acc, nmi, ari = metrics['ACC'], metrics['NMI'], metrics['ARI']
            results.append((acc, nmi, ari))
            w.writerow([i, rs, n_init, f"{acc:.6f}", f"{nmi:.6f}", f"{ari:.6f}"])
            print(f"[run {i:02d}] seed={rs} | ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")

        arr = np.array(results, dtype=np.float64)
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0, ddof=0)

        w.writerow([])
        w.writerow(["mean", "", "", f"{means[0]:.6f}", f"{means[1]:.6f}", f"{means[2]:.6f}"])
        w.writerow(["std",  "", "", f"{stds[0]:.6f}", f"{stds[1]:.6f}", f"{stds[2]:.6f}"])

    print("\n========== Summary ==========")
    print(f"ACC  mean={means[0]:.4f}, std={stds[0]:.4f}")
    print(f"NMI  mean={means[1]:.4f}, std={stds[1]:.4f}")
    print(f"ARI  mean={means[2]:.4f}, std={stds[2]:.4f}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()

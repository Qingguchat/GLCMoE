# data/dataloader.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .dataset import MultiViewDataset

def get_dataloader(
    mat_file: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 42,
):

    dataset = MultiViewDataset(mat_file)

    dataset.standardize(train_idx=None)

    n_samples = getattr(dataset, 'num_samples', len(dataset))
    full_indices = list(range(n_samples))
    full_subset = Subset(dataset, full_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin = torch.cuda.is_available()
    persistent = num_workers > 0

    loader = DataLoader(
        full_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        generator=generator if shuffle else None,  # 仅训练用
        drop_last=False,
    )

    num_views = dataset.num_views
    view_dims = [Xv.shape[1] for Xv in dataset.X_list]
    print(f"[get_dataloader] full-dataset loader built | batch_size={batch_size}, num_workers={num_workers}")
    print(f"  Sample size: {n_samples} | View count: {num_views} | Per view dimension: {view_dims}")

    return loader, dataset, num_views, view_dims

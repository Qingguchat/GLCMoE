import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):

    def __init__(self, mat_file, transform=None):
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"Mat file not found: {mat_file}")

        mat = sio.loadmat(mat_file)

        if 'X' not in mat:
            raise KeyError(f"{mat_file} must contain variable 'X' (object array of views).")

        label_key = 'Y' if 'Y' in mat else ('y' if 'y' in mat else None)
        if label_key is None:
            raise KeyError(f"{mat_file} must contain variable 'Y' or 'y' for labels.")

        X_cell = mat['X']
        if not isinstance(X_cell, np.ndarray) or X_cell.dtype != 'object':
            raise ValueError("X must be an object array of views (Matlab cell).")

        Y = np.asarray(mat[label_key]).squeeze()
        if Y.ndim == 0:
            Y = np.array([Y])
        Y = Y.astype(np.int64)
        uniq = np.unique(Y)
        if uniq.min() == 1 and (uniq.max() - uniq.min() + 1) == uniq.size:
            Y = Y - 1
        self.labels = torch.from_numpy(Y)

        if X_cell.ndim != 2 or (X_cell.shape[0] != 1 and X_cell.shape[1] != 1):
            raise ValueError(f"X must be a 2D object array with shape (1,V) or (V,1), got {X_cell.shape}")

        is_row = (X_cell.shape[0] == 1)
        num_views = X_cell.shape[1] if is_row else X_cell.shape[0]

        self.X_list = []
        n = self.labels.shape[0]
        for v in range(num_views):
            Xv = X_cell[0, v] if is_row else X_cell[v, 0]
            Xv = np.asarray(Xv)
            if Xv.ndim != 2:
                raise ValueError(f"Each view must be 2D, got shape={Xv.shape} at view {v}")
            if Xv.shape[0] != n and Xv.shape[1] == n:
                Xv = Xv.T
            if Xv.shape[0] != n:
                raise ValueError(f"View {v} has {Xv.shape[0]} samples but labels have {n}. "
                                 f"Please check the orientation of X[{v}].")
            self.X_list.append(torch.from_numpy(Xv.astype(np.float32, copy=False)))

        self.num_views = num_views
        self.num_samples = n
        self.transform = transform

    def standardize(self, train_idx=None):
        if train_idx is None:
            for i, Xv in enumerate(self.X_list):
                mean = Xv.mean(dim=0, keepdim=True)
                std = Xv.std(dim=0, keepdim=True)
                std[std == 0] = 1.0
                self.X_list[i] = (Xv - mean) / std
        else:
            for i, Xv in enumerate(self.X_list):
                train_data = Xv[train_idx]
                mean = train_data.mean(dim=0, keepdim=True)
                std = train_data.std(dim=0, keepdim=True)
                std[std == 0] = 1.0
                self.X_list[i] = (Xv - mean) / std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            views = []
            for Xv in self.X_list:
                xv = Xv[idx]
                if self.transform:
                    xv = self.transform(xv)
                views.append(xv)
            label = self.labels[idx]
            return views, label
        except Exception as e:
            print(f"__getitem__ error at idx={idx}: {e}")
            raise

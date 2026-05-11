"""
utils.py — Training utilities for the AARN crop mapping project.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix


class CropDataset(Dataset):
    """Spectral-only dataset (Phase 1 & 2 base models)."""
    def __init__(self, X, mask, y, indices):
        self.X    = torch.tensor(X[indices],    dtype=torch.float32)
        self.mask = torch.tensor(mask[indices], dtype=torch.float32)
        self.y    = torch.tensor(y[indices],    dtype=torch.long)

    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.mask[i], self.y[i]


class CropDatasetSoil(Dataset):
    """Spectral + soil dataset (Phase 3 MCTNetCASoilV2)."""
    def __init__(self, X, mask, soil, y, indices):
        self.X    = torch.tensor(X[indices],    dtype=torch.float32)
        self.mask = torch.tensor(mask[indices], dtype=torch.float32)
        self.soil = torch.tensor(soil[indices], dtype=torch.float32)
        self.y    = torch.tensor(y[indices],    dtype=torch.long)

    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.mask[i], self.soil[i], self.y[i]


def compute_class_weights(y_train, n_classes, device):
    counts  = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    return torch.tensor(weights / weights.sum() * n_classes,
                        dtype=torch.float32).to(device)


def evaluate(model, loader, device, has_soil=False):
    """Return (OA, Kappa, macro-F1, per-class-F1) on a DataLoader."""
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in loader:
            if has_soil:
                Xb, mb, sb, yb = batch
                out = model(Xb.to(device), mb.to(device), sb.to(device))
            else:
                Xb, mb, yb = batch
                out = model(Xb.to(device), mb.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            labs.extend(yb.numpy())
    preds, labs = np.array(preds), np.array(labs)
    OA    = (preds == labs).mean()
    Kappa = cohen_kappa_score(labs, preds)
    F1    = f1_score(labs, preds, average='macro')
    F1_pc = f1_score(labs, preds, average=None)
    return OA, Kappa, F1, F1_pc

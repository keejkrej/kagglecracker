"""
CAFA 6 Protein Function Prediction: Gold-Tier Multi-Modal Hybrid Architecture
Author: Antigravity & User (KaggleCracker)
Hardware: Trained on NVIDIA A100-SXM4 (40GB)
"""

import os
import time
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Model Architecture
# ==========================================
class AspectExpertMLP(nn.Module):
    """Specialist Deep Residual Network for individual Gene Ontology Aspects."""
    def __init__(self, in_dim=6144, hidden_dim=1024, num_classes=500, dropout=0.35):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.act = nn.GELU()
        self.head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h = self.encoder(x)
        h = self.act(h + self.res1(h))
        h = self.act(h + self.res2(h))
        return self.head(h)

# ==========================================
# 2. Evaluation Metric: Vectorized GPU F-max
# ==========================================
def compute_fmax_gpu(y_pred_probs, y_true):
    best_f1 = 0.0
    best_th = 0.5
    for th in torch.linspace(0.1, 0.7, 13, device=y_pred_probs.device):
        pred_pos = (y_pred_probs >= th)
        true_pos = (y_true == 1.0)
        
        tp = (pred_pos & true_pos).sum(dim=1).float()
        fp = (pred_pos & ~true_pos).sum(dim=1).float()
        fn = (~pred_pos & true_pos).sum(dim=1).float()
        
        precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp)).mean()
        recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp)).mean()
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            if f1.item() > best_f1:
                best_f1 = f1.item()
                best_th = th.item()
    return best_f1, best_th

# ==========================================
# 3. GPU-Accelerated Homology Transfer (k-NN)
# ==========================================
def run_gpu_knn(X_test_norm, X_train_norm, Y_train, k=15, temp=0.05, batch_size=1000):
    """
    Sub-second proteome-wide sequence homology label transfer 
    via tensor matrix multiplication on the GPU.
    """
    knn_preds = []
    for i in range(0, len(X_test_norm), batch_size):
        batch = X_test_norm[i:i+batch_size]
        sims = torch.mm(batch, X_train_norm.t())
        top_sims, top_idx = torch.topk(sims, k=k, dim=1)
        w = F.softmax(top_sims / temp, dim=1)
        batch_pred = torch.sum(Y_train[top_idx] * w.unsqueeze(-1), dim=1)
        knn_preds.append(batch_pred)
    return torch.cat(knn_preds, dim=0)

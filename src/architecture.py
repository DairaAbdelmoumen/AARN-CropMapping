"""
architecture.py — All model classes for the AARN crop mapping project.

Models:
  - MCTNet          : Wang et al. 2024 reproduction (Phase 1 baseline)
  - MCTNetCASoilV2  : Proposed dual-stage soil injection model (Phase 3)

Usage:
    from src.architecture import MCTNet, MCTNetCASoilV2
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# BASE COMPONENTS  (shared by MCTNet and MCTNetCASoilV2)
# ══════════════════════════════════════════════════════════════════════════════

class ECA(nn.Module):
    """Efficient Channel Attention — Wang et al. 2024 Eq. (1)."""
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=kernel,
                                   padding=kernel // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x.transpose(1, 2))
        y = self.conv(y.transpose(1, 2))
        return x * self.sigmoid(y)


class ALPE(nn.Module):
    """Attention-based Learnable Positional Encoding — Wang et al. 2024 Eq. (3).
    Applied only at Stage 1 to handle cloud-gap masked timesteps."""
    def __init__(self, seq_len=36, d_model=10, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel,
                               padding=kernel // 2, bias=False)
        self.eca  = ECA(d_model, kernel=kernel)
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, mask):
        pe = self.pe.expand(x.size(0), -1, -1) * mask.unsqueeze(-1)
        pe = self.conv(pe.transpose(1, 2)).transpose(1, 2)
        pe = self.eca(pe)
        return x + pe


class CNNSubModule(nn.Module):
    """Residual CNN sub-module — Wang et al. 2024."""
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1, bias=False) \
                     if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        x   = F.relu(self.bn1(self.conv1(x)))
        x   = self.bn2(self.conv2(x))
        return F.relu(x + res)


class TransformerSubModule(nn.Module):
    """Single Transformer encoder layer."""
    def __init__(self, d_model, n_heads, dropout=0.3):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class CTFusionStage(nn.Module):
    """CNN-Transformer Fusion stage with optional ALPE and max-pooling."""
    def __init__(self, in_ch, out_ch, n_heads, kernel=3, dropout=0.3,
                 use_alpe=False, seq_len=36):
        super().__init__()
        self.use_alpe = use_alpe
        if use_alpe:
            self.alpe = ALPE(seq_len=seq_len, d_model=in_ch, kernel=kernel)
        self.cnn           = CNNSubModule(in_ch, out_ch, kernel)
        self.trans         = TransformerSubModule(in_ch, n_heads, dropout)
        self.fusion_linear = nn.Linear(out_ch + in_ch, out_ch)
        self.pool          = nn.MaxPool1d(2)

    def forward(self, x, mask=None):
        x_t   = self.alpe(x, mask) if (self.use_alpe and mask is not None) else x
        x_cnn = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x_t   = self.trans(x_t)
        fused = self.fusion_linear(torch.cat([x_cnn, x_t], dim=-1))
        return self.pool(fused.transpose(1, 2)).transpose(1, 2)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — MCTNet (Wang et al. 2024)
# ══════════════════════════════════════════════════════════════════════════════

class MCTNet(nn.Module):
    """
    MCTNet — Wang et al. 2024.
    Lightweight CNN-Transformer for pixel-based crop mapping.
    Tuned hyperparameters: Arkansas dropout=0.5 lr=0.0005 / California dropout=0.6 lr=0.0005
    Parameters: ~54,428
    """
    def __init__(self, n_classes=5, seq_len=36, n_bands=10,
                 n_heads=5, kernel=3, dropout=0.3):
        super().__init__()
        self.stage1     = CTFusionStage(n_bands, 20, n_heads, kernel, dropout,
                                         use_alpe=True, seq_len=seq_len)
        self.stage2     = CTFusionStage(20, 40, n_heads, kernel, dropout)
        self.stage3     = CTFusionStage(40, 80, n_heads, kernel, dropout)
        self.classifier = nn.Linear(80, n_classes)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.stage1(x, mask)   # (B, 18, 20)
        x = self.stage2(x)         # (B,  9, 40)
        x = self.stage3(x)         # (B,  4, 80)
        x = x.max(dim=1).values    # (B, 80)
        x = self.dropout(x)
        return self.classifier(x)  # (B, n_classes)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — MCTNetCASoilV2 (proposed)
# ══════════════════════════════════════════════════════════════════════════════

class SoilFiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on soil.
    Applied at Stage 1 output (B, 18, 20).

    soil (B,3) → MLP → γ(B,20) + β(B,20)
    output = x · (1+γ) + β  →  LayerNorm

    Provides early per-channel soil conditioning before temporal pooling.
    """
    def __init__(self, soil_dim=3, d_model=20):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(soil_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, soil):
        gamma, beta = self.film(soil).chunk(2, dim=-1)
        return self.norm(x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1))


class SoilTemporalGating(nn.Module):
    """
    Fixed per-timestep soil gating via reversed cross-attention.
    Applied at Stage 3 output (B, 4, 80).

    v1 bug:  soil→Q, temporal→K,V  → output (B,1,80) broadcast to all T  [WRONG]
    v2 fix: temporal→Q, soil→K,V  → output (B,T,80) unique per timestep [CORRECT]

    Also blends with FiLM pathway via learned α (helps when T=4 is small).
    """
    def __init__(self, soil_dim=3, d_model=80, n_heads=4, dropout=0.1):
        super().__init__()
        self.soil_proj = nn.Sequential(
            nn.Linear(soil_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.film = nn.Sequential(
            nn.Linear(soil_dim, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, temporal_feats, soil):
        soil_kv = self.soil_proj(soil).unsqueeze(1)        # (B, 1, d)
        attended, _ = self.cross_attn(
            query=temporal_feats, key=soil_kv, value=soil_kv)  # (B, T, d)
        gamma, beta = self.film(soil).chunk(2, dim=-1)
        film_out = temporal_feats * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        alpha   = torch.sigmoid(self.alpha)
        blended = alpha * attended + (1 - alpha) * film_out
        return self.norm(temporal_feats + blended)


class MCTNetCASoilV2(nn.Module):
    """
    MCTNetCASoilV2 — Phase 3 proposed model.
    Dual-stage soil injection: SoilFiLM at Stage 1 + SoilTemporalGating at Stage 3.

    Parameters: ~57,200
    Best results: California OA=0.9459 (+2.65 pp over Phase 1 baseline)
    """
    def __init__(self, n_classes=5, seq_len=36, n_bands=10,
                 n_heads=5, kernel=3, dropout=0.3,
                 soil_dim=3, ca_heads=4):
        super().__init__()
        self.stage1     = CTFusionStage(n_bands, 20, n_heads, kernel, dropout,
                                         use_alpe=True, seq_len=seq_len)
        self.soil_film1 = SoilFiLM(soil_dim=soil_dim, d_model=20)
        self.stage2     = CTFusionStage(20, 40, n_heads, kernel, dropout)
        self.stage3     = CTFusionStage(40, 80, n_heads, kernel, dropout)
        self.soil_ca    = SoilTemporalGating(soil_dim=soil_dim, d_model=80,
                                              n_heads=ca_heads, dropout=0.1)
        self.classifier = nn.Linear(80, n_classes)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask, soil):
        x = self.stage1(x, mask)        # (B, 18, 20)
        x = self.soil_film1(x, soil)    # (B, 18, 20) — early soil modulation
        x = self.stage2(x)              # (B,  9, 40)
        x = self.stage3(x)              # (B,  4, 80)
        x = self.soil_ca(x, soil)       # (B,  4, 80) — per-timestep gating
        x = x.max(dim=1).values         # (B, 80)
        x = self.dropout(x)
        return self.classifier(x)       # (B, n_classes)

# AARN — Crop Mapping with Sentinel-2 & Soil Covariates

> **Module:** Neural Networks (Réseaux de Neurones) — M1 SII 2025/2026  
> **University:** USTHB, Faculty of Computer Science, Algeria  
> **Base paper:** Wang et al. 2024 — *A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery*

---

## Project Overview

Three-phase deep learning project for pixel-level crop classification from Sentinel-2 time-series imagery (36 timesteps × 10 spectral bands), covering two US agricultural regions.

| Region | Crops | Train | Val | Test |
|--------|-------|-------|-----|------|
| Arkansas | Soybeans, Rice, Corn, Cotton, Others (5 classes) | 1,200 | 300 | ~8,500 |
| California Sacramento | Grapes, Rice, Alfalfa, Almonds, Pistachios, Others (6 classes) | 1,440 | 360 | ~8,200 |

---

## Results Summary

### Phase 1 — MCTNet Reproduction (test set)
| Region | OA | Kappa | Macro-F1 |
|--------|----|-------|----------|
| Arkansas | **0.9633** | 0.9542 | 0.9632 |
| California | **0.9194** | 0.9033 | 0.9209 |

### Phase 2 — Ablation with Auxiliary Covariates (val set ⚠)
Best config: **MCTNet + Soil** (pH, organic carbon, texture from OpenLandMap 250m)

| Region | OA | Kappa | Macro-F1 |
|--------|----|-------|----------|
| Arkansas | 0.9767 | 0.9708 | 0.9767 |
| California | 0.9500 | 0.9400 | 0.9501 |

> ⚠ Val-set only (300/360 pixels, same spatial pool as train). Inflated by spatial autocorrelation.

### Phase 3 — MCTNetCASoilV2 (test set)
Proposed model: dual-stage soil injection via **SoilFiLM** (Stage 1) + **SoilTemporalGating** (Stage 3).

| Region | OA | Kappa | Macro-F1 | Δ vs Phase 1 |
|--------|----|-------|----------|-------------|
| Arkansas | 0.9498 | 0.9244 | 0.9188 | −1.35 pp |
| **California** | **0.9459** | **0.9278** | **0.9199** | **+2.65 pp ✅ best overall** |

---

## Repository Structure

```
AARN-CropMapping/
├── notebooks/
│   ├── Preprocessing.ipynb           # Data extraction from GeoTIFF tiles
│   ├── Phase01_MCTNet.ipynb          # MCTNet reproduction + tuning
│   ├── Phase02_Ablation.ipynb        # Covariate ablation study
│   ├── Phase03_CASoilV2.ipynb        # MCTNetCASoilV2 training + evaluation
│   └── Phase03_ComparisonCell.py     # Standalone Cell 8: all comparison charts
│
├── src/
│   ├── architecture.py               # All model classes (MCTNet + MCTNetCASoilV2)
│   └── utils.py                      # Training loop, evaluation, dataset
│
├── results/
│   ├── phase1/new_mctnet_final_results.csv
│   ├── phase2/new_ablation_final_results.csv
│   └── phase3/
│       ├── p3v2_final_results.csv
│       └── p3_casoil_v1_results.csv
│
├── report/Phase3_Report.pdf
├── .gitignore
└── requirements.txt
```

> **Not tracked:** raw GeoTIFFs, `.npy` arrays, `.pth` weights — stored on Google Drive.

---

## Architecture: MCTNetCASoilV2

```
Input: X (B,36,10) + mask (B,36) + soil (B,3)
         ↓
Stage 1 CTFusion + ALPE  →  (B, 18, 20)
SoilFiLM  [early: soil→(γ,β)→scale+shift]  →  (B, 18, 20)
         ↓
Stage 2 CTFusion  →  (B, 9, 40)
Stage 3 CTFusion  →  (B, 4, 80)
SoilTemporalGating  [temporal→Q, soil→K,V → per-timestep]  →  (B, 4, 80)
         ↓
GlobalMaxPool → Dropout → Linear → (B, n_classes)
```

**Key fix over CASoil v1:** reversed Q/K/V so each timestep gets its own soil response instead of a broadcast channel vector.

---

## Open Notebooks in Colab

| Notebook | Colab Link |
|----------|-----------|
| Preprocessing | [▶ Open](https://colab.research.google.com/drive/10h0yHGNxbmewR7kriNNNbbH6QiVwWh0G) |
| Phase 1 — MCTNet | [▶ Open](https://colab.research.google.com/drive/1X-XGOz9Ato0Vlt9vzdGsg2N-hRPe6fSE) |
| Phase 2 — Ablation | [▶ Open](https://colab.research.google.com/drive/1K15Dg8YP9Uzxzi34qU6btmQEjnq6Sina) |
| Phase 3 — MCTNetCASoilV2 | [▶ Open](https://colab.research.google.com/drive/1MjbA9KknHhIhcf6Gx-To2G28Olhuu7IE) |

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/AARN-CropMapping.git
cd AARN-CropMapping
pip install -r requirements.txt
# Then open the Colab links above (Drive mount required)
```

---

## Data Sources

| Source | Use | Resolution |
|--------|-----|-----------|
| Sentinel-2 (GEE) | 10 bands × 36 timesteps, 2021 | 10 m |
| USDA CDL 2021 | Crop type labels | 30 m |
| OpenLandMap / SoilGrids | pH, OC, texture | 250 m |
| ERA5-Land | Climate ablation | 9 km |
| SRTM DEM | Topography ablation | 30 m |

---

## References

1. Wang et al. (2024). Lightweight CNN-Transformer for crop mapping. *Computers & Electronics in Agriculture*, 226, 109370.
2. Wang et al. (2021). Attention-based CNN for crop mapping. *Computers & Electronics in Agriculture*, 184, 106090.
3. Perez et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.
4. Woo et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
5. Poggio et al. (2021). SoilGrids 2.0. *SOIL*, 7, 217–240.
6. USDA NASS (2021). Cropland Data Layer.
7. Copernicus/ECMWF (2021). ERA5-Land hourly data.

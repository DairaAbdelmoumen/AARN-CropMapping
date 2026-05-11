# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Phase 3 v2 Visual Comparison  (paste into Colab after Cell 7)
#
# Produces 3 figures:
#   Fig 1 – OA / Kappa / F1 grouped bar chart (all 4 models, both regions)
#   Fig 2 – Per-class F1 bar chart  (MCTNetCASoilV2)
#   Fig 3 – OA progression bar + delta annotations
#
# Requires:  matplotlib, numpy (already imported in Cell 1)
# Data:      uses variables defined in Cell 1  (PHASE1, PHASE2_SOIL_VAL,
#            PHASE3_V1)  +  test_results_v2  from Cell 6
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── colour palette ───────────────────────────────────────────────────────────
C = {
    "p1"   : "#5C6BC0",   # indigo  – Phase 1  (test)
    "p2"   : "#FFA726",   # amber   – Phase 2  (val, ⚠)
    "v1"   : "#90A4AE",   # steel   – CASoil v1 (test)
    "v2"   : "#26A69A",   # teal    – CASoilV2  (test)
    "bg"   : "#FAFAFA",
    "grid" : "#EEEEEE",
    "text" : "#212121",
    "warn" : "#E65100",
}

METRICS = ["OA", "Kappa", "F1"]
REGIONS = ["arkansas", "california"]
REGION_LABELS = {"arkansas": "Arkansas", "california": "California"}

# ─── data (hard-coded so the cell works stand-alone after a Colab disconnect) ─
DATA = {
    "arkansas": {
        "p1": {"OA": 0.9633, "Kappa": 0.9542, "F1": 0.9632},
        "p2": {"OA": 0.9767, "Kappa": 0.9708, "F1": 0.9767},
        "v1": {"OA": 0.9376, "Kappa": 0.9058, "F1": 0.8963},
        "v2": {"OA": 0.9498, "Kappa": 0.9244, "F1": 0.9188},
    },
    "california": {
        "p1": {"OA": 0.9194, "Kappa": 0.9033, "F1": 0.9209},
        "p2": {"OA": 0.9500, "Kappa": 0.9400, "F1": 0.9501},
        "v1": {"OA": 0.9441, "Kappa": 0.9250, "F1": 0.9227},
        "v2": {"OA": 0.9459, "Kappa": 0.9278, "F1": 0.9199},
    },
}

PER_CLASS = {
    "arkansas": {
        "classes": ["Soybeans", "Rice", "Corn", "Cotton", "Others"],
        "v2":      [0.985,       0.952,  0.948,  0.870,   0.839],
    },
    "california": {
        "classes": ["Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios", "Others"],
        "v2":      [0.997,    0.964,  0.930,     0.851,     0.839,        0.939],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 – Grouped bar chart: OA / Kappa / F1  for all 4 models
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C["bg"])
fig1.suptitle(
    "Phase 1 (MCTNet) · Phase 2 best (MCTNet+Soil, val ⚠) · "
    "Phase 3 v1 (CASoil) · Phase 3 v2 (CASoilV2) — Test metrics",
    fontsize=11, color=C["text"], y=1.02
)

MODEL_KEYS   = ["p1", "p2", "v1", "v2"]
MODEL_LABELS = ["Phase 1 (test)", "Phase 2+Soil (val ⚠)", "CASoil v1 (test)", "CASoilV2 (test)"]
COLORS       = [C["p1"], C["p2"], C["v1"], C["v2"]]

BAR_W  = 0.18
OFFSET = np.array([-1.5, -0.5, 0.5, 1.5]) * BAR_W
X      = np.arange(len(METRICS))

for ax, region in zip(axes, REGIONS):
    ax.set_facecolor(C["bg"])
    ax.yaxis.grid(True, color=C["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for i, (key, label, col) in enumerate(zip(MODEL_KEYS, MODEL_LABELS, COLORS)):
        vals = [DATA[region][key][m] for m in METRICS]
        bars = ax.bar(X + OFFSET[i], vals, BAR_W,
                      label=label, color=col, alpha=0.88,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=C["text"], rotation=90)

    # horizontal reference line at Phase 1 OA
    ax.axhline(DATA[region]["p1"]["OA"], color=C["p1"],
               linestyle="--", linewidth=0.9, alpha=0.55, zorder=2)

    ax.set_xticks(X)
    ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_ylim(0.84, 1.035)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(REGION_LABELS[region], fontsize=12, color=C["text"], fontweight="bold")
    ax.tick_params(colors=C["text"])
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(C["grid"])
    ax.spines["bottom"].set_color(C["grid"])

axes[0].legend(fontsize=7.5, loc="lower right", framealpha=0.8)

# warn annotation
fig1.text(0.5, -0.04,
    "⚠  Phase 2 val set (300/360 pixels) — not directly comparable to test set",
    ha="center", fontsize=8, color=C["warn"])

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/AARN_project/results_new/p3v2_comparison_fig1.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Fig 1 saved → p3v2_comparison_fig1.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 – Per-class F1  (MCTNetCASoilV2 on test set)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4.5), facecolor=C["bg"])
fig2.suptitle("MCTNetCASoilV2 — Per-Class F1 (test set)", fontsize=12, color=C["text"])

for ax, region in zip(axes2, REGIONS):
    ax.set_facecolor(C["bg"])
    classes = PER_CLASS[region]["classes"]
    f1s     = PER_CLASS[region]["v2"]
    x       = np.arange(len(classes))

    bar_cols = [C["v2"] if f >= 0.90 else C["v1"] if f >= 0.85 else C["warn"] for f in f1s]
    bars = ax.bar(x, f1s, color=bar_cols, alpha=0.85,
                  edgecolor="white", linewidth=0.6, zorder=3)
    for bar, f in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, f + 0.006,
                f"{f:.3f}", ha="center", va="bottom", fontsize=8.5, color=C["text"])

    ax.axhline(0.90, color=C["p1"], linestyle="--", linewidth=0.8, alpha=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=22, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score", fontsize=9)
    ax.set_title(REGION_LABELS[region], fontsize=11, fontweight="bold", color=C["text"])
    ax.yaxis.grid(True, color=C["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/AARN_project/results_new/p3v2_comparison_fig2.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Fig 2 saved → p3v2_comparison_fig2.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 – OA progression + delta annotations
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=C["bg"])
fig3.suptitle("Overall Accuracy — progression across models", fontsize=12, color=C["text"])

MODELS_OA = ["Phase 1\n(test)", "Phase 2\n(val ⚠)", "CASoil v1\n(test)", "CASoilV2\n(test)"]
OA_COLS   = [C["p1"], C["p2"], C["v1"], C["v2"]]

for ax, region in zip(axes3, REGIONS):
    ax.set_facecolor(C["bg"])
    oa_vals = [DATA[region][k]["OA"] for k in MODEL_KEYS]
    x = np.arange(len(MODELS_OA))
    bars = ax.bar(x, oa_vals, color=OA_COLS, alpha=0.85,
                  edgecolor="white", linewidth=0.5, width=0.55, zorder=3)

    for bar, v in zip(bars, oa_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, color=C["text"])

    # Draw delta arrows v2 vs p1
    d_p1 = DATA[region]["v2"]["OA"] - DATA[region]["p1"]["OA"]
    sign = "+" if d_p1 >= 0 else ""
    fc   = C["v2"] if d_p1 >= 0 else C["warn"]
    ax.annotate(
        f'Δ vs Phase 1\n{sign}{d_p1*100:.2f} pp',
        xy=(3, DATA[region]["v2"]["OA"]),
        xytext=(2.6, DATA[region]["v2"]["OA"] + (0.012 if d_p1 >= 0 else -0.012)),
        fontsize=8, color=fc, ha="center",
        arrowprops=dict(arrowstyle="->", color=fc, lw=1.2),
    )

    ax.axhline(DATA[region]["p1"]["OA"], color=C["p1"],
               linestyle="--", linewidth=0.8, alpha=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS_OA, fontsize=8.5)
    lo = min(oa_vals) - 0.015
    hi = max(oa_vals) + 0.025
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Overall Accuracy", fontsize=9)
    ax.set_title(REGION_LABELS[region], fontsize=11, fontweight="bold", color=C["text"])
    ax.yaxis.grid(True, color=C["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

fig3.text(0.5, -0.04,
    "⚠  Phase 2 val set — spatially correlated with training, not comparable to test-set OA",
    ha="center", fontsize=7.5, color=C["warn"])

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/AARN_project/results_new/p3v2_comparison_fig3.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Fig 3 saved → p3v2_comparison_fig3.png")

print("\n✅ All 3 comparison figures saved to results_new/")

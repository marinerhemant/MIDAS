# %% [markdown]
# # Per-grain σ + sigma-aware trust tier
#
# v4 can compute proper data-driven (σ_X, σ_Y, σ_Z) per grain via
# Hessian inversion of the spot-residual NLL (delegated to
# `midas_propagate.per_grain_hessian_blocks`). This notebook shows:
#
# 1. How to enable Stage 7 (per-grain σ) and Stage 8 (strain)
# 2. How to read the new leaf columns
# 3. How to interpret the `sigma_aware` trust tier (gold requires σ + n_spots)

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from midas_process_grains.v4_pipeline import run_v4_pipeline

LAYER_DIR = Path("/path/to/your/layer/LayerNr_1")  # ← edit me
OUT_DIR   = LAYER_DIR / "v4_out_sigma"

# %% [markdown]
# ## 1. Run with σ + strain enabled
#
# Stage 7 (per-grain σ) takes ~0.5 s/grain on CPU. For large datasets
# (50k+ grains) use `position_sigma_max_grains` to sample a subset.

# %%
paths = run_v4_pipeline(
    layer_dir=LAYER_DIR,
    out_dir=OUT_DIR,
    merge_primitive="forward_predict",
    compute_position_sigma=True,
    position_sigma_max_grains=5000,
    compute_strain=True,
)

# %%
leaf = pd.read_csv(paths["leaf"], sep="\t")
new_cols = ["sigma_X_um", "sigma_Y_um", "sigma_Z_um",
            "sigma_residual_rms_px", "sigma_R_NNLS_um",
            "eps_11", "eps_22", "eps_33",
            "trust_tier_sigma_aware"]
print("New leaf columns:")
for c in new_cols:
    if c in leaf.columns:
        nn = leaf[c].notna().sum()
        print(f"  {c:<28s}  filled: {nn:>6,}/{len(leaf):,}")

# %% [markdown]
# ## 2. σ distributions

# %%
sig = leaf.dropna(subset=["sigma_X_um", "sigma_Y_um", "sigma_Z_um"])
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.logspace(0.5, 4, 80)
for col, lab in [("sigma_X_um", "σ_X"),
                  ("sigma_Y_um", "σ_Y"),
                  ("sigma_Z_um", "σ_Z")]:
    ax.hist(sig[col], bins=bins, alpha=0.5,
            label=f"{lab} (med {sig[col].median():.0f} µm)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("σ (µm)"); ax.set_ylabel("count")
ax.legend(); ax.grid(alpha=0.3, which="both")
ax.set_title(f"Per-grain position uncertainty (Hessian inversion, n={len(sig):,})")
plt.show()

# %% [markdown]
# ## 3. σ-aware trust filter
#
# The `sigma_aware` tier requires:
# - hkl_coverage ≥ 0.8 (= strict)
# - Clean cluster (no variant duplicates, no Stage-3 split)
# - σ ≤ 100 µm in all three axes
# - ≥ 20 matched spots

# %%
for tier_col in ["trust_tier_strict", "trust_tier_sigma_aware"]:
    if tier_col not in leaf.columns:
        continue
    counts = leaf[tier_col].value_counts().sort_index()
    print(f"{tier_col}:")
    for v, name in [(2, "gold"), (1, "silver"), (0, "bronze")]:
        n = int(counts.get(v, 0))
        print(f"  {name}: {n:,} ({100*n/len(leaf):.1f}%)")
    print()

# %% [markdown]
# ## 4. Out-of-box analysis using σ
#
# If you know the physical sample dimensions, you can ask: how many grains
# lie outside the sample, accounting for their own σ?

# %%
# Example: 1200 × 1000 × 200 µm sample, centred at origin
sample_um = (1200, 1000, 200)
hx, hy, hz = (s / 2 for s in sample_um)
out_z = np.maximum(np.abs(leaf["Z"]) - hz, 0)
sigma_devs = out_z / np.maximum(leaf["sigma_Z_um"].fillna(1.0), 1.0)
n_inside  = int((sigma_devs == 0).sum())
n_within2 = int((sigma_devs < 2).sum())
n_bad     = int((sigma_devs >= 2).sum())
print(f"Sample: {sample_um} µm  (Z half-extent {hz})")
print(f"  In box exactly:        {n_inside:,} ({100*n_inside/len(leaf):.1f}%)")
print(f"  Within 2σ_Z of box:    {n_within2:,} ({100*n_within2/len(leaf):.1f}%)")
print(f"  > 2σ outside (BAD):    {n_bad:,} ({100*n_bad/len(leaf):.1f}%)")

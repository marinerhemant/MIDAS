# %% [markdown]
# # Multi-phase workflows and auto-phase detection
#
# v4 supports multi-phase samples via **per-phase invocation**: call
# `run_v4_pipeline` once per phase, each with the right space group +
# lattice. Each invocation produces its own leaf table; the user unions
# the results downstream.
#
# `auto_phase.detect_phase` provides a starting guess for an unlabelled
# dataset based on observed ring d-spacings.

# %%
import pandas as pd
import numpy as np
from pathlib import Path

from midas_process_grains.compute.auto_phase import (
    detect_phase, detect_phase_from_inputall, COMMON_PHASES,
)

# %% [markdown]
# ## 1. Auto-phase from observed ring d-spacings

# %%
# Example: feed Au's d-spacings (should pick Au)
au_ds = np.array([2.355, 2.039, 1.442, 1.230, 1.178])
res = detect_phase(au_ds)
print(f"Best match: {res.best.name}  (SG {res.best.space_group})  score={res.score:.4f}")
print("Top 3:")
for name, score in sorted(res.all_scores.items(), key=lambda kv: kv[1])[:3]:
    print(f"  {name:>20s}: {score:.4f}")

# %% [markdown]
# ## 2. Convenience: auto-phase from an InputAll CSV
#
# Given a layer's InputAllExtraInfoFittingAll.csv, computes ring-median
# d-spacings from YLab/ZLab via 2θ = arctan(√(Y²+Z²)/Lsd).

# %%
LAYER_DIR = Path("/path/to/your/layer/LayerNr_1")  # ← edit me
if LAYER_DIR.exists():
    ia = pd.read_csv(LAYER_DIR / "InputAllExtraInfoFittingAll.csv",
                     sep=r"\s+", engine="c")
    res = detect_phase_from_inputall(ia, lsd_um=1_000_000.0, wavelength_A=0.172979)
    print(f"Auto-detected phase: {res.best.name} (SG {res.best.space_group})")

# %% [markdown]
# ## 3. Multi-phase invocation pattern
#
# For a duplex sample with two phases, run `run_v4_pipeline` once per
# phase by setting `space_group` explicitly and pointing each invocation
# at its own output directory. The CLI also accepts `--space-group N`.

# %%
# from midas_process_grains.v4_pipeline import run_v4_pipeline
# paths_phase1 = run_v4_pipeline(
#     layer_dir=LAYER_DIR,
#     out_dir=LAYER_DIR / "v4_phase1_FCC",
#     space_group=225,
# )
# paths_phase2 = run_v4_pipeline(
#     layer_dir=LAYER_DIR,
#     out_dir=LAYER_DIR / "v4_phase2_HCP",
#     space_group=194,
# )
#
# # Union the two leaf tables for downstream analysis
# leaf1 = pd.read_csv(paths_phase1["leaf"], sep="\t").assign(phase="FCC")
# leaf2 = pd.read_csv(paths_phase2["leaf"], sep="\t").assign(phase="HCP")
# both = pd.concat([leaf1, leaf2], ignore_index=True)
# print(both["phase"].value_counts())

# %% [markdown]
# ## 4. Library of known phases
#
# The default candidate set covers common FCC/BCC/HCP/oxide phases. Extend
# with your own PhaseCandidate entries when working on a non-standard
# material.

# %%
print(f"Built-in phases: {len(COMMON_PHASES)}")
for c in COMMON_PHASES:
    print(f"  {c.name:<20s}  SG={c.space_group:>3}  a={c.lattice[0]:.3f}Å  "
          f"(top d = {c.d_spacings_A[0]:.3f}Å)")

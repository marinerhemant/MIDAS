# %% [markdown]
# # v4 quick-start: from `paramstest.txt` to `GrainsV4.csv`
#
# This notebook walks through the minimum-effort path: take a MIDAS FF-HEDM
# layer directory and produce a v4 leaf grain table with per-grain
# orientations, positions, NNLS volumes, trust tiers, and (optionally)
# per-grain position uncertainty + strain.
#
# Prerequisites:
#
# * An existing layer directory with `paramstest.txt`,
#   `Results/OrientPosFit.bin`, `Results/ProcessKey.bin`,
#   `InputAllExtraInfoFittingAll.csv`, `hkls.csv`, and `SpotMatrix.csv`.
# * `midas_env` activated.
# * Set `LAYER_DIR` below to your layer.

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from midas_process_grains.v4_pipeline import run_v4_pipeline

LAYER_DIR = Path("/path/to/your/layer/LayerNr_1")  # ← edit me
OUT_DIR   = LAYER_DIR / "v4_out_tutorial"

# %% [markdown]
# ## 1. Run the pipeline with defaults
#
# Defaults: forward-predict merge primitive with auto-K, OM-spread split
# at 1.0°, matched-spot trust, FCC Σ3+Σ9+Σ27 twin labelling (or HCP/etc
# auto-dispatched from the space group + lattice).

# %%
paths = run_v4_pipeline(
    layer_dir=LAYER_DIR,
    out_dir=OUT_DIR,
    merge_primitive="forward_predict",
    trust_scheme="strict",
)
print("Wrote:")
for k, v in paths.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 2. Inspect the leaf grain table
#
# `GrainsV4.csv` is the LEAF table — one row per primitive grain. Twin
# variants are listed separately.

# %%
leaf = pd.read_csv(paths["leaf"], sep="\t")
print(f"n_grains = {len(leaf):,}")
print(f"columns:\n  {list(leaf.columns)}")
print()
print(leaf[["GrainID", "X", "Y", "Z", "Confidence", "hkl_coverage",
            "GrainRadius_NNLS", "trust_tier_strict", "twin_partner_id"]].head(8))

# %% [markdown]
# ## 3. Trust-tier breakdown

# %%
for tier_col in ["trust_tier_strict", "trust_tier_loose"]:
    if tier_col not in leaf.columns:
        continue
    counts = leaf[tier_col].value_counts().sort_index()
    print(f"{tier_col:>22s}: " + "  ".join(
        f"{name}={int(counts.get(v, 0)):,}"
        for v, name in [(2, "gold"), (1, "silver"), (0, "bronze")]
    ))

# %% [markdown]
# ## 4. Spatial map of grain centers (XY projection)

# %%
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(leaf["X"], leaf["Y"], c=leaf["Confidence"], s=4, cmap="viridis",
                vmin=0.5, vmax=1.0, alpha=0.7)
plt.colorbar(sc, ax=ax, label="Confidence")
ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
ax.set_aspect("equal")
ax.set_title(f"Grain centres ({len(leaf):,} leaf grains)")
plt.show()

# %% [markdown]
# ## 5. Parent-grain rollup (twin families collapsed)
#
# For workflows that want the coarse-grained view, `GrainsV4_families.csv`
# gives one row per **physical parent grain** — either a twin family or a
# singleton leaf. Iterating it yields `n_parent_grains` entries, with
# rotation-mean OM and volume-weighted mean position.

# %%
fam = pd.read_csv(paths["families"], sep="\t")
print(f"n_parent_grains = {len(fam):,}")
print(f"  twin families: {(fam.ParentType == 'twin').sum():,}")
print(f"  singletons:    {(fam.ParentType == 'singleton').sum():,}")
print()
print(fam[["ParentID", "ParentType", "MemberCount", "X_um", "Y_um", "Z_um",
          "EquivalentRadius_um", "trust_tier_strict"]].head(8))

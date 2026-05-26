# %% [markdown]
# # Twin labelling and family rollup
#
# v4 supports twin labelling for all common crystal systems with a single
# dispatcher: `default_twin_relations_for(space_group, c_over_a=...)`.
# For dataset workflows it auto-dispatches from the SG in paramstest.
#
# Coverage:
#
# | System | SG range | Operators | c/a needed |
# |---|---|---|---|
# | Cubic FCC/BCC | 195-230 | Σ3 (4 variants) + Σ9 + Σ11 + Σ27a + Σ27b | no |
# | Hexagonal HCP | 168-194 | 5 systems × 6 K₁ variants = 30 ops | yes |
# | Trigonal | 143-167 | 60° about c | no |
# | Tetragonal | 75-142 | 5 systems (FePt, ZrO₂, MnAl) | yes |
# | Ortho/mono/triclinic | 1-74 | user-supplied | n/a |

# %%
import numpy as np
import torch

from midas_process_grains.compute.twin_label import label_twins
from midas_process_grains.compute.twins import (
    default_twin_relations_for, TwinRelation,
    default_hcp_twin_relations, default_cubic_twin_relations,
)
from midas_stress.orientation import (
    orient_mat_to_quat, misorientation_quat_batch,
)


def _quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


def _random_quats(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q * np.sign(q[:, 0:1])

# %% [markdown]
# ## 1. Default dispatcher per crystal system

# %%
for sg, c_a in [(225, None), (229, None), (194, 1.587), (123, 0.967), (167, None), (22, None)]:
    try:
        tw = default_twin_relations_for(sg, c_over_a=c_a)
    except ValueError as e:
        tw = f"<ValueError: {e}>"
    print(f"SG {sg:>3}  c/a={c_a}:  {len(tw) if not isinstance(tw, str) else tw} operators")

# %% [markdown]
# ## 2. HCP variant-level labelling (Ti c/a = 1.587)
#
# Each twin SYSTEM has 6 symmetry-equivalent K₁ variants. The labeller
# returns the SPECIFIC K₁ variant that matched, so users can analyse twin
# variant selection in deformed Ti/Mg/Zr.

# %%
N = 50
parents = _random_quats(N, seed=42)
tw_ops = default_hcp_twin_relations(c_over_a=1.587, systems=("tension_1012",))
# Apply ONE variant to plant twin partners
T = tw_ops[0].quaternion
twins = _quat_mul(parents, T); twins /= np.linalg.norm(twins, axis=1, keepdims=True)
quats = np.concatenate([parents, twins], axis=0)

partner, family, twin_type, n_pairs = label_twins(
    grain_quats=quats, space_group=194, c_over_a=1.587, tol_deg=0.5,
)
labels = [t for t in twin_type if t]
print(f"planted {N} HCP {{10-12}} pairs → labeller found {n_pairs} unique pairs")
print(f"distinct labels: {set(labels)}")

# %% [markdown]
# ## 3. Cubic Σ3 + Σ9 + Σ27 + transitive closure
#
# A Σ3-Σ3 chain (= Σ9) is correctly merged into a single twin family by
# union-find over all detected pairs.

# %%
# Plant grain A → Σ3 → B → Σ9 → C
fcc_tw = default_cubic_twin_relations(include=("Sigma3", "Sigma9"))
T_sig3 = next(t for t in fcc_tw if "Sigma3" in t.name and "<111>" in t.name).quaternion
T_sig9 = next(t for t in fcc_tw if "Sigma9" in t.name).quaternion
A = _random_quats(1, seed=99).flatten()
B = _quat_mul(A, T_sig3); B /= np.linalg.norm(B)
C = _quat_mul(A, T_sig9); C /= np.linalg.norm(C)
quats = np.stack([A, B, C])

partner, family, twin_type, n_pairs = label_twins(
    grain_quats=quats, space_group=225, tol_deg=0.5,
)
print(f"A↔B via Σ3, A↔C via Σ9 — n_pairs={n_pairs}")
print(f"family ids: {family.tolist()}")
print(f"  → all three should share the same family id via transitive closure")
assert family[0] == family[1] == family[2], "transitive closure broken"

# %% [markdown]
# ## 4. User-supplied operator for an orthorhombic sample
#
# For lattices the dispatcher doesn't cover, pass `twin_relations=[my_op]`.

# %%
albite = TwinRelation(
    name="Ortho_Albite_180b",
    quaternion=np.array([0.0, 0.0, 1.0, 0.0]),
    angle_deg=180.0, axis=(0.0, 1.0, 0.0),
)
N = 25
parents = _random_quats(N, seed=77)
twins = _quat_mul(parents, albite.quaternion); twins /= np.linalg.norm(twins, axis=1, keepdims=True)
quats = np.concatenate([parents, twins], axis=0)
partner, _, twin_type, n_pairs = label_twins(
    grain_quats=quats, space_group=22, twin_relations=[albite], tol_deg=0.5,
)
print(f"planted {N} albite pairs → labelled {n_pairs}; labels = {set(t for t in twin_type if t)}")

# %% [markdown]
# ## 5. Family rollup
#
# The pipeline emits `GrainsV4_families.csv` with one row per **physical
# parent grain** (twin family OR singleton). Useful when you want the
# coarse-grained grain count rather than the leaf (variant-level) count.

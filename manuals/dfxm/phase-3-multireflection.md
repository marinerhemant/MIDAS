# Phase 3 — The multi-reflection deformation-gradient tensor

> Part of the **DFXM doc set**; spine is [`README.md`](README.md). **Do not enter this phase
> without passing the registration gate.** The binding systematic here is inter-reflection
> registration, not photon statistics (rule 4, Notebook §2) — this is the campaign's
> strongest real-data finding, and a **halt condition**.

---

## 3a. The registration gate — run it first

To recover a nine-component deformation-gradient tensor $\mathbf F(x)$ you need ≥3
non-coplanar reflections co-registered **voxel-for-voxel**. Different reflections diffract at
different 2θ → different magnification and field of view, so the per-reflection maps from
Phase 2 are **not on a common grid**.

Check, before anything else:

1. Is there **co-registration metadata** (fiducials, a shared sample frame) in the deposit?
   If **no** → **halt** (README STOP table). Report the wall; do not fabricate a tensor.
2. If you must try anyway, search the best intensity cross-correlation over scale + shift
   between two reflections and read where the maximum sits. On the real ID06 111↔002 set the
   best NCC was **0.43 at the search edge** (−0.30 native) — the signature of a wall, not a
   registration you can trust (`make_real_multibragg.py`, Notebook §2).

**A per-reflection map can be excellent and the tensor still unrecoverable.** ID06 111 had
p95 ≈ 45 mdeg and 002 ≈ 6 mdeg — both clean — yet could not be fused. Report the
per-reflection maps and the wall; that is the honest result.

## 3b. If the gate passes: the exact finite-strain inverse

With co-registered reflections, the observable is linear in the distortion,
$\Delta\mathbf Q^{(g)} = \mathbf H\,\mathbf G_0^{(g)}$, and the inverse is exact (no
small-strain assumption). Use the package, not a hand-rolled solver (rule 7):

```python
from midas_dfxm.field_inverse import (
    deformation_design_matrix, deformation_identifiability,
    recover_deformation_direct, recover_deformation_regularised, deformation_covariance,
)
rank_ok = deformation_identifiability(hkls, geometry)   # rank-9? which components recover?
F = recover_deformation_direct(delta_Q, hkls, geometry) # exact full-F, F = (I+H)^{-T}
```

Check `deformation_identifiability` first: a single reflection recovers only the projected
$\varepsilon_{gg}$; the full rank-6 strain + rotation needs a non-coplanar set. Report the
condition number and which components are supported — a low-rank set with a plausible tensor
is a trap.

**The rank ceiling is geometric, and no rocking strategy beats it.** Each θ-rocking sensitivity
row is an outer product $\hat{\mathbf Q}\otimes\mathbf v$, so with $\hat{\mathbf Q}$ confined to
a plane the row space lies in (2-D)×(3-D) and the **rank cannot exceed 6 for any rotation axis**
— verified against 500 random axes. Adding a third axis does not help; rank 9 requires
$\hat{\mathbf Q}$ **out** of the plane. So do not propose a cleverer rock as the fix, and read
the prior art before proposing anything here (Notebook §5f):

- **Detlefs et al., J. Appl. Cryst. (2025)** — oblique diffraction geometry, ≥ 3 non-coplanar
  **symmetry-equivalent** reflections under identical illumination. This is the published
  solution: it reaches rank 9, and symmetry-equivalence keeps |F| identical, which removes the
  intensity problem that sinks most reflection-set recommendations.
- **Kanesalingam et al. (2025)** — the full deformation-gradient inverse formalism plus a κ(ν)
  sensitivity metric for choosing the set.

Both are cited in `midas_dfxm`'s own docstrings (`deformation_identifiability`,
`fisher_information`). We independently re-derived a weaker version of this and were about to
recommend it as new; that retraction is Notebook §5f.

For noisy real data, `recover_deformation_regularised` adds curvature regularisation (helps
smooth fields, **over-smooths structured ones** — check against the direct inverse), and
`deformation_covariance` gives the per-voxel uncertainty (Cramér–Rao) to quote.

## 3c. Registration as a modelling aid (not a substitute for metadata)

Where the field is structured, the differentiable substrate can carry the per-reflection
shifts $\boldsymbol\delta_g$ as parameters and co-optimise them with the tensor (minimise
multi-reflection consistency). On the synthetic study this removed about half the
misregistration strain; the rest is set by a drift×gradient prefactor (~45 µε per pixel of
mis-registration). **This is a modelling aid, not a replacement for co-registration
metadata** — a self-registered tensor built from metadata-poor archived scans is not a
measurement (Notebook §2).

**Output of Phase 3:** either a per-voxel $\mathbf F(x)$ with its identifiability and
covariance, **or** the registration wall reported honestly with the per-reflection maps.

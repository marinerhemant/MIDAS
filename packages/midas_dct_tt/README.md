# midas-dct-tt

Differentiable forward + inverse for **diffraction contrast tomography (DCT)** and
**topotomography (TT)**, built on `midas-dfxm` and `midas-diffract`.

> **PRIVATE / UNRELEASED.** `packages/midas_dct_tt/` is excluded in
> `.git/info/exclude` (local-only, so the ignore rule itself cannot leak the
> name), and `release.sh` refuses to run while it is. Nothing about this package — the name
> included — should reach `origin`, PyPI, a talk, or a proposal until
> [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) has been worked through
> deliberately.

## What it is

DCT and TT are not separate physics. They are the same forward operator as DFXM
with the objective removed and the scan convention changed:

|          | acceptance                          | projection      | rotation axis  | projections/grain |
| -------- | ----------------------------------- | --------------- | -------------- | ----------------- |
| **DFXM** | divergence + bandwidth + objective NA | magnified, inclined | goniometer | n/a (real-space image) |
| **TT**   | divergence + bandwidth              | parallel (M = 1) | **parallel to G** | 10²–10³, freely chosen |
| **DCT**  | divergence + bandwidth              | parallel (M = 1) | lab vertical   | 10–60, crystallography-fixed |

So this package is a scan-convention layer plus a projection/reconstruction layer
over existing MIDAS primitives — not a new physics stack. The reuse map is in
`implementation_plan.md` §5, and the two reuse claims that matter (`na = 0` for
the acceptance, `M = 1` for the projection) are pinned by
[`tests/test_dfxm_contract.py`](tests/test_dfxm_contract.py) so an upstream
release cannot break them quietly.

## The one result to know first

With the tomographic rotation axis parallel to **G** — the TT alignment — the
Bragg condition gives, exactly and for every θ:

```
angle(k_h, a_hat) = 90° − θ
```

Parallel-beam tomography needs the projection direction perpendicular to the
rotation axis. TT is off by exactly θ, so **TT is laminography with a missing
cone of half-angle θ**, not tomography. Consequences: low-θ (high-energy,
low-index) reflections reconstruct better; Friedel-paired settings flip the cone
and recover part of it; a reconstructed sphere elongates along the axis.

`midas_dct_tt.missing_cone_half_angle_deg` carries the full statement; the
identity is verified from vectors, independently of the θ that produced them, in
`tests/test_geometry.py`.

## Install (editable, from the repo)

```bash
pip install -e packages/midas_dct_tt --no-deps    # siblings already editable
```

## Quickstart

```python
import torch
from midas_dct_tt import sphere_grain, attach_uniform_field, tt_alignment

# A planted grain: soft voxel occupancy chi(r) + a perfect-crystal F(r).
grain = attach_uniform_field(sphere_grain(radius_um=5.0, spacing_um=0.5))

# Align its (111) so G lies on the tomographic rotation axis.
G0 = grain.field.reference_G((1, 1, 1))          # sample frame, |G| = 2*pi/d
al = tt_alignment(G0, wavelength_A=0.172979)      # ~71.7 keV, 1-ID/HEXM

print(f"theta        = {float(al.theta_deg):.3f} deg")
print(f"missing cone = {float(al.missing_cone_deg()):.3f} deg half-angle")
print(f"k_h to axis  = {float(al.axis_beam_angle_deg()):.3f} deg  (= 90 - theta)")

# The TT scan: G is invariant, so k_h never moves.
for psi in (0.0, 90.0, 180.0, 270.0):
    lab_positions = grain.positions_lab(al.sample_rotation(psi))
```

Phase 1 — a topograph, and a full sinogram:

```python
from midas_dct_tt import PlaneDetector, psi_scan, topograph_image, topograph_stack, tt_resolution

det = PlaneDetector(pixel_um=0.5, shape=(96, 96), distance_um=5000.0)

# Ideal grain: the pixel reads path length in micrometers (centre = 2R).
img = topograph_image(grain, al, psi_deg=0.0, detector=det)

# With acceptance and deformation: intensity AND position respond to F(r).
res = tt_resolution(al)                       # DFXM's resolution at na = 0
sino = topograph_stack(grain, al, psi_scan(180), detector=det,
                       hkl=(1, 1, 1), resolution=res)     # (180, 96, 96)
```

DCT instead — solve the Bragg flashes, then render frames:

```python
from midas_dct_tt import bragg_flashes, dct_frames, dct_omega_scan

flashes = bragg_flashes(G0, wavelength_A=0.172979)   # 2 per 360 deg, or [] if blind
frames = dct_frames([grain], [flashes], wavelength_A=0.172979,
                    detector=PlaneDetector(pixel_um=1.0, shape=(192, 192), distance_um=500.0),
                    omega_centres=dct_omega_scan(720))
```

Phase 2 — reconstruct the grain shape from the sinogram:

```python
from midas_dct_tt import forward_operator, sirt, reconstruct_differentiable, dice

A = forward_operator(grain.positions, al, psi_scan(36), det,
                     voxel_volume_um3=grain.voxel_volume_um3)
sino = A(grain.occupancy)                       # linear in chi: no sigmoid, no acceptance

chi_sirt = sirt(sino, A, grain.n_voxels, n_iter=60)          # classical baseline
chi_grad, info = reconstruct_differentiable(sino, A, grain.n_voxels, steps=300)
print(dice(chi_sirt, grain.occupancy), dice(chi_grad, grain.occupancy))
```

## Conventions

Inherited unchanged from `midas_dfxm` so a grain, a field, and a reflection mean
the same thing in both packages:

- lab **x** along the incident beam, **z** vertical up, `y = z × x`;
- `k_in = (2π/λ) x̂`, and reciprocal vectors carry the 2π convention (`|G| = 2π/d`);
- `v_lab = R @ v_sample`;
- **micrometers, degrees, Ångström** (wavelength and lattice only).

Two conventions are hazards rather than choices, and both are named constants
with loud docstrings rather than literals:

- **DCT ω sign.** The 1-ID aero stage is clockwise, so recorded ω must be
  negated (`DCT_OMEGA_SIGN_AERO`). A wrong sign here does not degrade any
  residual — it *mirrors* the reconstruction, undetectably.
- **Soft-boundary volume bias.** Occupancy is `sigmoid(−sdf/w)`, which
  over-counts volume on a convex surface by `A·(2H)·w²·π²/6`. Measured, ∝ w²,
  and it *inverts* below `w ≈ 0.25 × spacing` where the grid stops resolving the
  ramp. Do not quote a grain volume from a soft occupancy without correcting or
  stating `w`. See `logits_from_signed_distance`.

## Status

**Phases 0-3 complete** — conventions/geometry/containers; the forward model (TT
topographs, DCT frames, acceptance, extinction, projector); shape reconstruction
(SIRT + differentiable, Friedel pairing, missing-cone study); and the deformation
inverse with a pre-registered identifiability study. 343 tests, all three device backends verified: CPU + MPS on macOS,
**CUDA on an H200** (sentosa, torch 2.11/cu128, PYTHONPATH overlay on the shared
MIDAS env).

Validated against two independent implementations:

- **`skimage.transform.radon`** for the projector — 4e-12 relative at 0°/90°,
  ~0.2% RMS elsewhere (its interpolation blur, which we don't incur).
- **`pymicro`** for the DCT scan geometry — our Bragg-flash angles match
  `Orientation.dct_omega_angles` to **1.1e-13 degrees** across 3 orientations ×
  3 reflections, with no fitted constant.

plus an internal NumPy oracle to 1e-13. A sphere projects to its chord, `2R`, and
projected mass is conserved exactly.

> **Phase 3 refuted this package's original novelty claim.** Exact finite-strain
> inversion is *correct* — its difference from the linearised model is exactly the
> omitted `O(|H|²)` term, verified to better than 1% — but the difference reaches
> only **2.2% at |H| = 5%** and needs **|H| ≈ 24%** to reach 10%, below the
> measurement floor throughout. Separately, a fixed TT setting goes dark at
> |H| ≈ 2e-3 (rotation), 13× before the linearisation matters. Pre-registered in
> `dev/paper/PREREGISTER.md`; full result in `dev/paper/RESULTS_phase3.md`.
>
> What survives: a validated differentiable forward model, a clean identifiability
> result (**one reflection constrains 3 of 9 components of F; three non-coplanar
> give all 9; collinear add nothing**), and the finding that TT's spatial
> resolution resolves a strain sign that intensity alone cannot.

Three results that change how you should use this package:

- **A Friedel partner is a lab-frame *mirror*, not an inversion.** The two
  lab-frame `G` vectors share `(x, y)` and have opposite `z` (cos = +0.79).
  Pair in the **sample** frame — a lab-frame antiparallel test returns zero pairs
  and fails silently. `BraggFlash` carries `G_sample` for exactly this.
- **Do not measure the missing cone by reconstructing a sphere.** It detects
  nothing (Dice 1.0000, zero elongation, at any θ) because same-grid noiseless
  data is an inverse crime. Measure the operator's Fourier response instead —
  `tests/test_missing_cone.py` shows ~10× suppression inside the cone with the
  width tracking θ.

And two from Phase 1 about reflection selection:

- **Low θ trades tomographic coverage against strain contrast.** Both scale as
  cot θ: fcc (111) at 71.7 keV gives a 2.4° missing cone but σ_par/|Q| = 6.4e-3;
  at 17 keV, 10.0° and 1.5e-3. Neither choice dominates, so reflection selection
  wants Fisher-information design rather than a rule of thumb.
- **TT strain contrast is divergence-limited, not bandwidth-limited** at HEXM
  energies (the cot²θ·div² term beats 4ε² by ~2000×). A narrower monochromator
  will not sharpen it; collimation will.

## Tests

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q
```

Markers: `unit`, `contract` (sibling API surfaces), `device` (CPU/CUDA/MPS),
`autograd` (gradcheck), `slow` (deselected by default; run with `-m slow`).

`tests/test_radon_crosscheck.py` needs `scikit-image` and skips without it.
`tests/test_pymicro_crosscheck.py` uses a frozen reference table rather than
importing pymicro, which pins `numpy<2`; regenerate with
`examples/pymicro_reference.py`.

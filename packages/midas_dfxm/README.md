# midas-dfxm

Differentiable **forward + inverse** for **Dark-Field X-ray Microscopy (DFXM)**.

DFXM images a bulk grain through a magnifying objective (CRL / MLL) placed on one
diffracted beam. The objective is a **reciprocal-space bandpass**: a sample voxel
contributes intensity to its magnified detector pixel only if its *local*
scattering vector falls inside the instrument's resolution function for the
current goniometer setting. Rocking the sample (a *mosaicity scan*) maps local
**orientation**; scanning 2θ / energy (a *strain scan*) maps local **d-spacing**;
weak-beam settings image **individual dislocations**.

This package models that signal from a voxelised **deformation-gradient field**
`F(r)`, and (later phases) inverts a DFXM image stack back to `F(r)` and to
**per-dislocation defect types** (edge/screw, Burgers vector, slip system).

**Reuse-first, no re-porting.** Built on the MIDAS differentiable stack:
`midas-stress` (orientation/strain), `midas-hkls` (structure factors, form
factors, DWF), `midas-2d` (continuous-q structure factor, resolution
convolution), `midas-defect` (Stroh anisotropic-elasticity contrast solver, slip
systems, GND, planar-defect rods), `midas-invert` (fit / UQ / experiment design),
`midas-distortion` (detector model). Everything is torch-differentiable and
device-portable (CPU / CUDA / MPS).

See [`implementation_plan.md`](implementation_plan.md) for the full roadmap,
physics reuse map, phase gates, and honest scope/novelty gates.

## Status

**Pre-alpha (v0.0.1a0).** Phases 0,1,3,4,5 + a simulation-anchored roadmap
implemented and tested — **64 tests pass (CPU + MPS; 3 CUDA-skipped), ~2400 LOC.**

Highlights:
- **Phase 4 (defect typing):** anisotropic-elasticity (Stroh) per-dislocation forward;
  `g·b` invisibility; edge/screw character; **Burgers-vector recovery**.
- **Phase 5 (inverse typing):** `identify_dislocation` recovers slip system, character,
  core position, and the **Burgers-vector sign** (the gap Borgi 2025 leaves open) from
  multi-reflection weak-beam images.
- **Phase 3 (field inverse):** full strain-tensor recovery + identifiability + UQ;
  honest finding — regularisation helps at low SNR, ~neutral at high SNR.
- **#1 physics coupling:** recover a **GND density** end-to-end through the forward
  (Nye `κ = ρb`) — the DFXM↔DDD/CP interface, differentiably.
- **#2 credibility anchor:** independent numpy oracle agrees **bit-for-bit** (~1e-16).
- **External-field ready:** `field_from_deformation_gradient` consumes an external `F(r)` with
  zero rework.

See [`SIMULATION_CATALOG.md`](SIMULATION_CATALOG.md) and the shipped tutorials in
`midas_dfxm/examples/`, each runnable as a module:

```bash
python -m midas_dfxm.examples.tutorial_dislocation_typing
```

Figures land in `dev/paper/figures/` inside a clone, `./figures` otherwise; set
`MIDAS_FIGDIR` to send them somewhere else.

<details><summary>Earlier milestone note</summary>

Phases 0–1 implemented and tested (30 passing, CPU + MPS):

- **Phase 0 — conventions + field.** `conventions.py` (lab/sample/imaging frames,
  goniometer), `field.py` (`DeformationField`, the kinematic deform operator
  `Q = F⁻ᵀ G0`, polar decomposition `F = R·U`), `io.py` (synthetic-field
  generators: perfect crystal, orientation gradient, uniform strain, isotropic
  screw dislocation; plus a stub loader for an external collaborator field).
- **Phase 1 — geometrical-optics forward.** `resolution.py` (anisotropic-Gaussian
  reciprocal-space acceptance), `optics.py` (magnifying inclined projection +
  bilinear detector splat), `scan.py` (mosaicity / rocking / strain scan builders,
  Bragg-angle helpers), `forward.py` (`dfxm_image`, `dfxm_stack`, `mosaicity_curve`).

Validated analytic limits: rocking-curve FWHM = `2√(2ln2)·σ⊥ / |axis×q|`, FCC
forbidden reflection is dark, uniaxial strain shifts `|Q|` correctly; plus
gradcheck on the deform operator and end-to-end autograd to `F` and the instrument
widths.

**Next:** Phase 2 (field forward on a collaborator's realistic `F(r)` when it lands),
Phase 3 (field inverse + identifiability study), **Phase 4 (per-dislocation
forward via the Stroh solver + `g·b` defect typing)**, Phase 5 (inverse defect
discovery).

</details>

## Quickstart

```python
import torch
from midas_dfxm import (
    make_uniform_field, with_orientation_gradient,
    GoniometerSetting, reference_q_nom, aligned_resolution,
    ObjectiveOptics, bragg_two_theta_deg, dfxm_image,
)

# A curved crystal grain (smooth lattice rotation across x).
field = make_uniform_field(shape=(64, 64, 1), spacing_um=0.5)
field = with_orientation_gradient(field, axis=(0, 0, 1), deg_per_um=0.01, along=0)

hkl, center = (1, 1, 1), GoniometerSetting()
q_nom = reference_q_nom(field, hkl, center)
res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=5e-3)
tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), wavelength_A=0.172979)
optics = ObjectiveOptics(two_theta_deg=tt, magnification=10.0, detector_shape=(256, 256))

image = dfxm_image(field, hkl, center, res, optics)   # (256, 256), differentiable
```

## Lab frame and scattering plane

Two beamline conventions are settable, and both default to the ESRF ID06-HXM case
so existing code is unchanged.

**Which axis is the beam** — `LabFrame` carries the choice; `outboard` is derived
as `up × beam`, so a frame cannot be built inconsistently.

| preset | beam | up | outboard |
|---|---|---|---|
| `MIDAS_FRAME` (default, = ESRF) | `x` | `z` | `y` |
| `APS_FRAME` (Park convention) | `Z` | `Y` | `X` |
| `BEAM_X_UP_Y` | `x` | `y` | `−z` |

Any other orthogonal `(beam, up)` pair works too. `frame_rotation`,
`convert_vector`, `convert_tensor` (strain, `F`, Nye) and `convert_orientation`
move between them, and agree exactly with `midas_stress.frames` for the
MIDAS↔APS pair — that module stays the repo-wide authority.

```python
from midas_dfxm import GoniometerSetting, APS_FRAME, MIDAS_FRAME

g = GoniometerSetting.from_aps(mu=2.0, omega=4.0)   # geometry supplied in APS coords
g_midas = g.in_frame(MIDAS_FRAME)                   # what the rest of the package uses
```

The motor angles are the same numbers in both — only the axes they are taken
about are relabelled (`G_dst = R G_src Rᵀ`).

**Which plane it scatters in** — `ScatteringGeometry(plane="vertical"|"horizontal")`,
or an explicit `deflection` direction for an oblique plane. This is *not* a
relabelling; it changes physics in two places:

- **Which motor is the base tilt.** Vertical plane → `mu`. Horizontal plane →
  `omega`, and `mu` is *inert* for a reflection whose `Q` lies along outboard,
  since it rotates about the axis `Q` sits on.
- **Which divergence limits which width.** Poulsen 2017 Eqs. 58–63 put `div_v`
  on `sigma_rock`/`sigma_par` and `div_h` on `sigma_roll`. That holds for a
  vertical plane; a horizontal one swaps them. `poulsen_resolution_widths(...,
  geometry=...)` resolves it. The stock defaults have `div_v == div_h`, which
  hides the swap entirely — on an anisotropic source it is a ~1.6× error in each
  width, with `sigma_roll` further amplified by `1/sin θ`.

```python
from midas_dfxm import diffracted_beam_direction, poulsen_resolution_widths

diffracted_beam_direction(12.0, geometry="horizontal")   # APS 6-ID-C transmission
poulsen_resolution_widths(3.0, two_theta_deg=20.0, geometry="horizontal",
                          div_v=0.2e-3, div_h=1.0e-3)
```

Note that `two_theta_from_k_out` needs the frame, and gets it wrong quietly
without one: a horizontal-plane 12° reflection in APS components, read as MIDAS
components, returns 78° (`90 − 2θ`); the vertical-plane case returns a flat 90°
for any angle, since the beam component is identically zero. Pass `geometry=`
whenever the vector did not come from this package.

## Tests

```bash
# macOS: work around the duplicate-OpenMP-runtime abort (torch + MIDAS siblings).
export KMP_DUPLICATE_LIB_OK=TRUE
python -m pytest tests/ -q
```

Markers: `unit` (analytic correctness), `autograd` (gradcheck / autograd),
`device` (CPU/CUDA/MPS parity), `slow` (heavy integration).

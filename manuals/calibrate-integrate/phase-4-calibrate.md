# Phase 4 — Calibrate

> Part of the **calibrate-integrate doc set**. Spine: [`README.md`](README.md).
> Split out of the spine 2026-08-21; the spine keeps the gates, this keeps the recipe.

## §4. Calibrate — from scratch, never from an existing block

The CLI does not expose multi-panel refinement, so a tiled detector uses the
Python API:

```python
import numpy as np
from midas_calibrate.params import CalibrationParams
from midas_calibrate_v2.io.readers import read_image
from midas_calibrate_v2.seed.auto_seed import make_seed
from midas_calibrate_v2.compat.from_v1 import (
    spec_from_v1_params, add_panel_parameters,
    add_panel_no_expansion_constraint)
from midas_calibrate_v2.compat.to_v1 import write_v1_paramstest
from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.pipelines.four_stage import autocalibrate_four_stage

# read_image knows BOTH sentinel conventions and returns the mask.  Do not
# substitute `img[img < 0] = 0`: it is blind to the EIGER high sentinel (§2).
img, bad = read_image(FRAME, return_mask=True)   # sentinels already zeroed
img[img < 0] = 0.0                # belt and braces for a low-sentinel file
seed = make_seed(img, wavelength_A=LAMBDA, px_um=PX, calibrant="CeO2")
# NB seed AFTER cleaning, not before — an earlier draft of this recipe fed
# make_seed the raw frame, sentinels and all.

v1 = CalibrationParams.from_file(TEMPLATE)   # thresholds, lattice, detector size
v1.BC_y, v1.BC_z, v1.Lsd = seed.BC_y, seed.BC_z, seed.Lsd_um
v1.tx = v1.ty = v1.tz = 0.0                  # start from scratch
for n in [f"p{i}" for i in range(15)]:
    setattr(v1, n, 0.0)
v1.MinRingRad, v1.MaxRingRad = R_MIN_PX, R_MAX_PX     # validate() rejects 0
v1.validate()

spec = spec_from_v1_params(v1)
add_panel_parameters(spec, n_panels=N, tol_shift_px=2.0, tol_rot_deg=0.0,
                     enable_lsd=False, enable_p2=False)     # modules only — rule 4
add_panel_no_expansion_constraint(spec)                     # rule 7
layout = PanelLayout.regular(NY, NZ, SY, SZ, gap_y=GAPS_Y, gap_z=GAPS_Z)

res = autocalibrate_four_stage(v1, img, spec=spec, panel_layout=layout,
                               common_kwargs=dict(drop_gap_fits=True))
write_v1_paramstest(res.stage2.unpacked, v1, "paramstest_v2.txt")
```

`write_v1_paramstest` also drops a `panelshifts` sidecar and points
`PanelShiftsFile` at it, whenever panel terms were refined
(`midas_calibrate_v2/compat/to_v1.py:55`).

**Gate — do not proceed until both pass:**

```
res.stage4_strain_uE_test  <  100 µε        # held-out, not the full set
|held-out − full| small                     # a large gap means overfitting
```

On the reference dataset: 66.1 µε held-out, 67.2 full. A calibrant refining worse
than 100 µε is not a calibration.

**Single-panel detector**: skip `add_panel_parameters` and `panel_layout`
entirely; everything else is identical.

---

## §4b. Two calibrants on one exposure

Someone mixed CeO2 and LaB6, or stuck two capillaries together. **Be clear about
what this buys before spending effort on it** — the accounting below is measured
(Lab Notebook §7), and two of the three things people expect from it are not
available.

| | |
|---|---|
| ✅ roughly twice the fit points, denser radial sampling | σ on Lsd / BC / tilts tightens ~30–43 %, which is exactly √N and no more |
| ✅ a cross-check no single calibrant can give | the gap between the two phases is a systematic-error estimate |
| ✅ per-phase sample position, if the powders are not co-located | §4b.3 |
| ❌ **wavelength** | both phases enter only through their d-spacings; rule 9 is unchanged |
| ❌ **azimuthal harmonics** | both powders illuminate the *same* wedge; rule 11 is unchanged |
| ❌ freedom from the lattice constant | da/a is exactly degenerate with dLsd/Lsd, per phase |

### §4b.1 Declare both phases

```python
from midas_calibrate_v2.seed.calibrant import phases_from_calibrants
v1.Phases = phases_from_calibrants(["CeO2", "LaB6"])   # FIRST entry seeds
v1.MinRingSeparation = 12.0        # px; drop rings that collide in radius
```

or, in a parameter file (`packages/midas_calibrate/midas_calibrate/params.py:60`):

```
Phase CeO2 225 5.41153 5.41153 5.41153 90 90 90
Phase LaB6 221 4.15689 4.15689 4.15689 90 90 90
MinRingSeparation 12.0
```

or from the CLI: `--calibrant CeO2 --calibrant LaB6 --min-ring-separation 12`.

**Order matters.** Seeding matches an arc pattern against *one* ring table, so
the first entry is the one `make_seed` uses. Put the smoother powder first: on
the reference frame the LaB6 seed returned 2095 mm against the CeO2 seed's
2735 mm. That is not a bug and not a tie to break by averaging — §4b.4.

`build_ring_table` (`packages/midas_calibrate/midas_calibrate/rings.py:150`)
then returns the union, sorted by radius, tagged with `phase_idx`. Everything
downstream of the ring table only ever asks a fitted point for its expected 2θ,
so no other stage needs to know about phases.

### §4b.2 The blend cut, and the trap inside it

Two interleaved ring sets always collide somewhere, and a blended ring's
centroid is dragged by its neighbour with no error and no warning.
`drop_blended_rings` (`rings.py:256`) flags the colliding rings **individually** —
not a radial cutoff, which would discard every ring outside the first collision.

**The trap:** several apparent zero-separation "doublets" are *exact hkl
degeneracies* — LaB6 (300)/(221), (410)/(322); CeO2 (511)/(333), (600)/(442).
Same d, one physical ring, two labels. They must be **merged, not excluded**.
`build_ring_table` does this by d-spacing — `_dedup_by_d`
(`packages/midas_calibrate/midas_calibrate/rings.py:129`), called at
`rings.py:186`, at the relative tolerance `DEFAULT_D_DEDUP_REL_TOL`
(`rings.py:29`). On the reference frame it absorbed 14 rows, and skipping it
would throw away good rings while looking like prudence.

After merging, a 12 px cut costs **6 or 7 rings of about 40** on every panel.

### §4b.3 If the powders may not be co-located

Each phase then sees its own distance, and a transverse offset moves its
apparent beam centre. Pass the same frame twice, one calibrant each, with
`build_multi_spec(..., mode="same_detector")`
(`packages/midas_calibrate_v2/midas_calibrate_v2/pipelines/multi.py:47`), which
shares the tilts and distortion and leaves `Lsd`, `BC_y`, `BC_z` per phase — that
per-exposure block *is* the sample position.

**Share the tilts.** One detector cannot tilt differently for different powders,
and leaving them free does not merely waste parameters, it biases what is left:
independently-refined tilts absorbed the difference between the calibrants and
reported a **1.43 mm** relative offset where sharing them gives **72 ± 34 µm**
(Lab Notebook §11).

Then do not over-read the answer: `dLsd/Lsd` for phase B is exactly degenerate
with a relative error in phase B's lattice constant. A single frame cannot
separate "the capillary sits 72 µm closer" from "the lattice constant is
2.6e-05 low". Several exactly-known distances can — rule 9's lever.

### §4b.4 Read the per-phase residual, not the pooled mean

```python
from midas_calibrate_v2.loss.diagnostics import per_phase_summary
print(per_phase_summary(r_uE, fits.phase_idx, fits.phase_names))
```

(`packages/midas_calibrate_v2/midas_calibrate_v2/loss/diagnostics.py:139`.)

**Gate.** The two phases should agree, *and* the absolute residual should be near
the floor. Agreement alone is not a pass: two calibrants sitting on a common
noise floor agree by construction. On the reference frame a run at 193 µε
reported the phases agreeing to 1.02× and that was read as success — it fails the
100 µε gate, and a converged run on the same data gave 45.6 / 69.0 µε, i.e.
**worse agreement, far better absolute**. Check the absolute number first, then
the ratio.

**Halt condition H8** if the phases disagree by more than ~1.5× once the absolute
residual is at the floor: the error budget is systematic, and the honest
uncertainty on the geometry is the spread between the phases, not the fit's σ.

---

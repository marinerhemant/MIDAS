# Calibrate → Integrate Runbook — point at a folder, get calibrated integrated patterns

**Use this to reduce a powder / calibrant dataset from raw frames.** Give one line:

```
Data folder: <ABSOLUTE PATH>     # a file, a folder of frames, or an experiment tree
```

and, if the files do not carry them, the calibrant (`CeO2`, `LaB6`, `Si`) and the
energy or wavelength. Everything else is worked out.

**This spine carries the commands.** Unlike `manuals/ff-hedm/README.md`, which
delegates to phase files, everything needed to run a reduction is here. Paths not
covered here are **not yet exercised** — §7 lists them, and the rule there is stop
and ask, not improvise.

Citations are `path:line` **relative to the repository root**, so they
resolve wherever the tree is checked out.

**`LAB_NOTEBOOK.md` is the companion.** This file is the procedure. The notebook is
the evidence: how each rule was measured, and the claims that had to be
**retracted**. Read it before re-investigating anything here.

---

## §0. Scope gate — read before touching data

Everything in this document was measured on **one detector**: a 48-panel Pilatus
(1475 × 1679, 172 µm) at APS 20-ID, CeO2 calibrant, 63 keV, single frames.

| you have | do |
|---|---|
| tiled Pilatus / single-panel area detector, powder calibrant | continue |
| a different detector geometry | continue, but re-derive the numbers — the *procedure* transfers, the *values* in §4 and §6 do not |
| spotty single-crystal rings, not powder | **stop** — this is `ff-hedm` or `pf-hedm` |
| tomographic translation series | **stop** — this is `xrd-ct` |
| no calibrant anywhere in the experiment | **stop and ask.** A geometry cannot be invented from a sample pattern |

`ENVELOPE.md` records what has and has not been exercised. Do not promote an
untested path to a recommendation.

---

## §1. Install gate — free, and skipping it invalidates everything

At APS, the shared environment is on the `/home/beams*` filesystem and is
visible from every beamline host:

```bash
conda deactivate
export PATH=/home/beams12/S1IDUSER/opt/envs/midas/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE
python -c "import midas_integrate_v2 as a, midas_calibrate_v2 as b; print(a.__version__, b.__version__)"
```

`conda deactivate` is required: an active `midas_env` shadows that `bin` even
after you set `PATH`. Outside APS, activate any environment with the packages
installed — nothing below depends on this path.

**The gate is behavioural, not a version number.** Four defects produce
plausible wrong answers rather than errors. Run:

```bash
python manuals/calibrate-integrate/floorcheck.py
```

It probes the behaviour each floor guarantees:

| probe | if it fails |
|---|---|
| integrate-v2 applies per-panel shifts | a tiled detector integrates with its panel calibration silently discarded |
| calibrate-v2 writes the panelshifts sidecar | refined shifts never reach disk |
| integrate v1 sizes the map buffer, always warns | the map truncates at fine `RBinSize`; normalised profile still looks fine |
| calibrate v1 parses `FixPanelID` | the anchored panel is silently 0 |
| integrate-v2 one-shot takes `--mask` / `--device` | masked pixels enter the profile as raw values |

**Exit non-zero means stop.** These are not conveniences.

Version *numbers* are deliberately not quoted here. An earlier draft did, and
got two of four wrong in opposite directions — one floor was ahead of any
released version and blocked forever, the other was behind the current version
and passed without the fix present. A behavioural probe cannot rot that way.

---

## §2. Survey — what is in the folder

Work out, do not ask:

```bash
ls <folder> | head -50            # format, count, naming
python -c "
import tifffile, numpy as np, sys
a = tifffile.imread(sys.argv[1])
print(a.shape, a.dtype, 'min', a.min(), 'max', a.max(), 'median', np.median(a))
" <one frame>
```

- **Detector size** comes from the frame, not from a parameter file.
- **Calibrant vs sample**: a calibrant frame has continuous, azimuthally smooth
  rings; a sample has spots or texture. If unsure, integrate one frame (§5) and
  look at I(η) on a strong ring — flat means powder.
- **Negative values** (`-1`, `-2`) are gap / bad-pixel sentinels, not counts.
  Clip them to 0 before calibrating; do **not** feed them to a fitter.
- **A mask** covering gaps and bad pixels: find it, or build one. If the facility
  has no bad-pixel map, a workable recipe is the ratio of each pixel to the
  azimuthal median at its own radius — smooth it, call the low tail a shadow,
  then add the beamstop disc, exact zeros and a frame border. Masking is not
  cosmetic: it moved **804 of 2980 bins by >1 %** on a single-panel frame with
  only 1 % masked, and 1775 of 1800 bins on the tiled one.

**Check `ImTransOpt` before anything else.** It declares how the stored frame
maps onto detector coordinates (row/column flips). A wrong value mirrors the
beam centre and every downstream number follows it, silently. It cannot be
checked after calibrating, because the refiner absorbs the mirror.

Test it directly: bin R about the file's beam centre under each candidate
transform and compare ring-centroid scatter. On a single-panel 2880² frame the
correct `ImTransOpt 2` gave **0.039 px RMS** about ideal; without the flip,
**1.374 px**, and ring contrast collapsed from 101 to 6.9. The right value is
unmistakable — but only if you look.

**Halt condition H1.** If the folder contains a parameter file, do not assume it
is correct for these frames. Check it (§6, "verify against the raw rings")
*before* using it. On the reference dataset the active block belonged to a
different file and every ring sat 2.6 px off — Lab Notebook §1.

---

## §3. The dispatch matrix — pick the path before running anything

Two independent axes. Pick one cell from each.

**Scale:**

| you have | calibration | integration |
|---|---|---|
| one file | reuse a verified geometry, or calibrate on it if it is the calibrant | §5a |
| one folder | calibrate once on the calibrant frame, apply to all | §5b |
| one experiment (several distances / detectors) | **not exercised — stop and ask** (§7) | — |

**Execution:**

| want | use |
|---|---|
| a few frames, simplest | one-shot, `--device cpu` |
| many frames | one-shot `--device cuda`, or batch |
| live during acquisition | server on port 60439 (§5c) |

CPU and GPU agree **exactly** on this path — v2 is float64 throughout, so there is
no reduction-order noise (measured: max abs difference 0 on 648 000 bins). Choose
on throughput, not on accuracy.

---

## §4. Calibrate — from scratch, never from an existing block

The CLI does not expose multi-panel refinement, so a tiled detector uses the
Python API:

```python
import numpy as np, tifffile
from midas_calibrate.params import CalibrationParams
from midas_calibrate_v2.seed.auto_seed import make_seed
from midas_calibrate_v2.compat.from_v1 import (
    spec_from_v1_params, add_panel_parameters,
    add_panel_no_expansion_constraint)
from midas_calibrate_v2.compat.to_v1 import write_v1_paramstest
from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.pipelines.four_stage import autocalibrate_four_stage

img = tifffile.imread(FRAME).astype(np.float64)
seed = make_seed(img, wavelength_A=LAMBDA, px_um=PX, calibrant="CeO2")
img[img < 0] = 0.0                       # gap sentinels are not counts

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

## §5. Integrate

### §5a. One file

```bash
midas-integrate-v2 paramstest_v2.txt --image FRAME.tiff \
    --mode subpixel -K 2 --device cpu --mask mask.tif --v1-out ./out
```

**`--mask` is not optional on a detector with gaps.** Without it every masked
pixel — including the `-1` gap sentinels — enters the profile as a raw value.
Measured on the reference frame: **1775 of 1800 bins change, by up to 25 %.**

`--v1-out` writes `lineout.bin`, `lineout_simple_mean.bin` and `Int2D.bin` —
the three files the beamline chain reads — beside the CSV. Without it, CSV only.

| `--mode` | what it does |
|---|---|
| `subpixel` (default) | K × K oversampled hard binning |
| `polygon` | exact Green's-theorem polygon-arc area; the principled choice |
| `hard` | one bin per pixel centre |
| `soft` | differentiable; **no `--v1-out`** (no per-bin weight to area-weight with) |

On the reference frame the mode barely moves the profile (Lab Notebook §4), so
`subpixel` is a fine default.

### §5b. One folder

```bash
midas-integrate-v2-batch paramstest_v2.txt --image-glob 'frames/*.tiff' \
    --mode subpixel --mask mask.tif --out-dir ./out
```

One CSV per frame. **Batch does not emit the v1 binaries** — if the downstream
chain needs them, loop §5a instead.

### §5c. Live stream

```bash
midas-integrate-v2-server paramstest_v2.txt --mode subpixel -K 2 \
    --device cuda --port 60439 --out ./stream
```

Same TCP port and the same wire protocol as the C
`IntegratorFitPeaksGPUStream`, hybrid dtype code 6 included, so an existing
feeder drives it unchanged — you swap only the consumer. The three v1 files are
written on shutdown. Long runs: `setsid`/`nohup` with a log, or an ssh hangup
kills it.

### v1 fallback

`midas-integrate` / `midas-integrate-server` take `--device cpu|cuda` and need a
`Map.bin` (`--map-dir`). It is still the only path measured directly against the
compiled CUDA binary (2.2e-07 max relative). Use it when byte-comparing to the C.

---

## §6. Verify — before quoting anything

**Ring overlay.** Contour the per-pixel R map at the ideal ring radii and look at
it over the frame. Contour the map, do not draw circles — circles ignore tilt,
distortion and panel shifts:

```python
from midas_integrate_v2.forward.pixels import eval_pixel_REta
from midas_integrate_v2.compat.from_v1 import spec_from_v1_params
R = eval_pixel_REta(spec_from_v1_params(p, requires_grad=False))[0].detach().numpy()
ax.contour(R, levels=ideal_radii_px, colors="#00E5FF", linewidths=0.6)
```

Contours must sit **on** the ring crests. Sitting uniformly inside or outside
means the distance is wrong — see DIAGNOSIS "rings uniformly offset".

**Ring positions in the lineout.** Match measured crests to ideal radii *by
position*, never by rank — the ranking differs between kernels and will
manufacture a false discrepancy (Lab Notebook §3).

---

## §7. Halt conditions — stop on these whether or not anything looks wrong

- **H1** A parameter file was found in the folder and used without checking it
  against the raw rings.
- **H2** The calibration was seeded from an existing geometry block rather than
  from `make_seed`.
- **H3** Held-out calibrant strain ≥ 100 µε, or a large held-out/full gap.
- **H4** A multi-panel detector, and `PanelShiftsFile` is named but unreadable —
  `PanelShiftsMissingWarning` fires and the integration silently uses zero
  shifts.
- **H5** More than a couple of panels rail at their bound. The parameterisation
  is wrong, not the data (rule 4, Lab Notebook §5).
- **H6** Scale = one experiment (several distances or detectors). Not exercised;
  ask rather than improvise.
- **H7** A non-powder sample, or no calibrant anywhere in the experiment.

---

## §8. Hard rules

1. **`SubPixelLevel` stays at 1.** Above 1 the CUDA integrator truncates the
   fractional sub-pixel coordinate and reads the neighbouring pixel — measured
   24.3× on in-band bins, and `IntegratorZarrOMP` is unaffected because it
   interpolates. `0` is bit-identical to `1`; write `1`.
   (`FF_HEDM/src/IntegratorFitPeaksGPUStream.cu:916`, Lab Notebook §2.)

2. **Never seed the calibration from an existing parameter block.** `make_seed`
   works from the image. A prior answer's errors are inherited silently.

3. **Clip negative sentinels before calibrating.** `-1` / `-2` are gaps and bad
   pixels, not counts.

4. **Fix δLsd to 0 and move the modules.** A module is misplaced *in the
   detector plane*, giving a constant ΔR; a per-panel δLsd gives ΔR ∝ R. Fitting
   the first with the second rails it — 16 of 48 panels in one run. Bound the
   in-plane shift at ~2 px and refine that. (Lab Notebook §5.)

5. **Measure before enabling `GradientCorrection`.** It is off everywhere by
   default. On the reference data the cardinal bands were *quieter* than a
   non-cardinal control, i.e. there was no aliasing to correct. Test with a
   control band before switching it on. (Lab Notebook §4.)

6. **Match rings by position, not by rank.** (Lab Notebook §3.)

7. **Turn on the expansion gauge for a tiled detector.** `fix_panel_id` and
   Σ panel = 0 remove the *translation* nullspace. They do not touch a second
   one: pushing every module outward in proportion to its radius shifts ring
   radii exactly the way an `Lsd` error does, so the fit trades freely between
   them. Measured: 11 % of the fitted panel field sat in that mode, ~73 % of it
   absorbable into `Lsd`. Without the gauge, panels rail — 9 of 48 in one run.
   `add_panel_no_expansion_constraint(spec)`.

8. **A powder ring cannot determine a module's tangential shift.** With η spread
   much below 90° on a module the 2 × 2 Fisher block is rank-1: only the radial
   component is identifiable. Do not report the tangential part as a measurement.

9. **λ is NOT determined by a single-distance powder pattern.** Wavelength and
   `Lsd` are degenerate: a ring at radius R constrains only the ratio, so a
   wrong λ is absorbed into `Lsd` and **the strain gate still passes**. A 1 %
   energy error becomes a 1 % distance error, silently, with a calibration that
   looks clean.

   So take λ from the beamline (monochromator/undulator), never from the fit,
   and cross-check it against the filename and the metadata. To break the
   degeneracy you need **several exactly-known distances** —
   `midas-calibrate-v2 --lsd-offsets`, which refines one shared `L0` plus known
   offsets and a shared λ. Measured on a planted 1 % error: the two hypotheses
   differed by 0.063 vs 0.083 px RMS, i.e. barely distinguishable.

10. **`FixPanelID` is a gauge choice, not a measurement.** Panel shifts from two
   calibrations with different anchors are not directly comparable.

---

## §9. Traps that silently corrupt results

| trap | symptom | guard |
|---|---|---|
| wrong `ImTransOpt` | beam centre mirrored; calibration still converges | test ring scatter per transform, §2 |
| a template's `SubPixelLevel > 1` copied into the written paramstest | the calibration output itself violates hard rule 1 | `midas-calibrate-v2 ≥ 0.8.2` clamps it and warns |
| v2 below 0.9.0 discards panel shifts | rings misplaced by up to 0.5 px, panel-organised | version floor §1 |
| v1 map buffer truncates at fine `RBinSize` | absolute flux and bin occupancy wrong, normalised profile looks fine | v1 ≥ 0.5.0, or `per_row_max_entries=40000` |
| `SubPixelCardinalWidth` default differed 5.0 vs 10.0 | two codes build different maps from one file | both ≥ the floors in §1 |
| `FixPanelID` never parsed | anchored panel silently 0 | `midas-calibrate` ≥ 0.3.1 |
| wrong calibration block active in the parameter file | every ring offset, uniformly | §6 overlay |
| wrong λ | absorbed into `Lsd`; **strain gate passes**, distance wrong by the same fraction | rule 9 — take λ from the beamline, cross-check the filename |
| integrating without `--mask` | bins biased low near gaps; no error, no warning | pass `--mask` (§5a) |
| a written paramstest inheriting the template's `PanelShiftsFile` | new geometry silently uses the *previous* calibration's panel shifts | `midas-calibrate-v2 ≥ 0.8.1`; check the line before reusing the file |

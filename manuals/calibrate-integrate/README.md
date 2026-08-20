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
| **two calibrants mixed in one exposure** (e.g. CeO2 + LaB6) | continue, and add **§4b** — one extra step, plus different expectations about what the second calibrant buys |
| **beam centre off the panel** (partial arcs, one azimuthal wedge) | continue, but read **rule 11** first — most of the distortion model is not identifiable and refining it will not converge |
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

**The gate is behavioural, not a version number.** Most of these defects produce
plausible wrong answers rather than errors; the last one produces a hard failure
on a whole detector class. Run:

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
| `read_image` flags the high (unsigned dtype-max) sentinel | 7 % of an EIGER frame enters the fit as 4.29e9, silently |
| importing MIDAS registers the HDF5 bitshuffle filter | bitshuffle/LZ4 HDF5 (EIGER, ESRF) will not open at all |

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
- **Out-of-band values are gap / bad-pixel sentinels, not counts** — and the
  sign is a vendor convention, not a rule. Pilatus writes them **low** (`-1`
  gap, `-2` overflow), which `img[img < 0] = 0` catches. Dectris EIGER writes
  them **high**, as the largest representable unsigned value (`2**32-1` for
  uint32), where every `< 0` guard fails open and a fitter is handed 4.29e9 as
  a photon count. Measured on a real EIGER2 16M frame: **7.10 % of the
  detector**. `read_image` flags the unsigned dtype-max by default and warns;
  take the mask from it rather than writing your own threshold —
  Lab Notebook §12.
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

## §5. Integrate

### §5a. One file

```bash
midas-integrate-v2 paramstest_v2.txt --image FRAME.tiff \
    --mode subpixel -K 2 --device cpu --mask mask.tif --v1-out ./out
```

**`--mask` is not optional on a detector with gaps.** Without it every masked
pixel — including the sentinels, `-1` on a Pilatus or `2**32-1` on an EIGER —
enters the profile as a raw value. Measured on the reference frame: **1775 of
1800 bins change, by up to 25 %.** Write the mask from the reader rather than
inventing a threshold: `read_image(..., return_mask=True)`, then
`tifffile.imwrite("mask.tif", mask.astype(np.uint8))` — the CLI reads it as
`tifffile.imread(...).astype(bool)`, so **1 = masked**.

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
- **H8** Two calibrants whose per-phase residuals differ by more than ~1.5× once
  the absolute residual is at the floor (§4b.4).
- **H9** The azimuth-coverage gate returns `fail`, i.e. harmonics are being
  refined on a wedge too narrow to determine them (rule 11).
- **H10** The per-iteration strain does not settle. An alternating E↔M loop that
  oscillates by more than about 2× between iterations has not converged, and
  "best of history" is then selecting a lucky iterate, not a calibration.
  Freeze parameters until it settles (rule 11) rather than reporting the best
  number the run happened to touch.
- **H11** `RhoD` more than ~1.5× the outermost fitted ring radius while radial
  distortion is refined (rule 12).

---

## §8. Hard rules

1. **`SubPixelLevel` stays at 1.** Above 1 the CUDA integrator truncates the
   fractional sub-pixel coordinate and reads the neighbouring pixel — measured
   24.3× on in-band bins, and `IntegratorZarrOMP` is unaffected because it
   interpolates. `0` is bit-identical to `1`; write `1`.
   (`FF_HEDM/src/IntegratorFitPeaksGPUStream.cu:916`, Lab Notebook §2.)

2. **Never seed the calibration from an existing parameter block.** `make_seed`
   works from the image. A prior answer's errors are inherited silently.

3. **Remove the sentinels before calibrating, and do not assume they are
   negative.** `-1` / `-2` (Pilatus) and `2**32-1` (EIGER, and any unsigned
   dtype-max) are gaps and bad pixels, not counts. `img[img < 0] = 0` catches
   only the first kind and fails **open** on the second, which is the more
   dangerous direction: the fitter gets 4.29e9 instead of a small negative.
   Use `read_image(..., return_mask=True)`, which handles both and returns the
   mask. Verified across the 1-ID archive: GE uint16 frames carry **zero**
   pixels at 65535, so this costs nothing on the detectors that never had the
   problem. (Lab Notebook §12.)

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
   and cross-check it against the filename and the metadata.

   **"From the beamline" is more specific than it sounds.** At 1-ID the
   monochromator is tuned to an absorption **K edge** and left there, so the
   number to use is the *tabulated edge energy of the foil element* — read the
   element from `~/new_data/<expt>/fastsweep_Emon*.txt` and look the edge up in
   MIDAS's own table, `midas_pdf/midas_pdf/data/fluor_edges.json`. Measured over
   116 beamtimes (Lab Notebook §13): 74 of the 82 with a logged energy sit
   within 0.3 % of a foil K edge; the monochromator readback in
   `fastpar_*.par` field 10 runs a median **0.040 % below** the tabulated edge;
   and `exp_setup.yml`'s `EDGE:` key is **stale** — where it disagrees with the
   Emon element (9 of 18 beamtimes) the Emon element is right 7 times and the
   yml value never. Some beamtimes are deliberately off-edge at a round setting
   (95, 100 keV); there the readback is all there is, and it carries ±0.1 %.

   To break the degeneracy with a measurement rather than a claim about the
   monochromator, you need **several exactly-known distances** —
   `midas-calibrate-v2 --lsd-offsets`, which refines one shared `L0` plus known
   offsets and a shared λ. Measured on a planted 1 % error: the two hypotheses
   differed by 0.063 vs 0.083 px RMS, i.e. barely distinguishable.

10. **`FixPanelID` is a gauge choice, not a measurement.** Panel shifts from two
   calibrations with different anchors are not directly comparable.

11. **Refine only the distortion the azimuth supports.** Every `a_k`/`phi_k` pair
   is a k-fold azimuthal harmonic and needs azimuth to be identifiable. Over a
   narrow wedge they are degenerate with the beam centre (1-fold) and the tilts
   (2-fold), so they rail at their bounds and the E↔M loop stops converging.
   Measured on a 4-panel detector whose beam centre lies off the corner, giving
   66–73° of each ring: the shipped calibration had **3, 4, 7 and 7 of its 15
   coefficients pinned at ±0.002**, and a refit with them free oscillated between
   84 and 4692 µε across iterations.

   Use `refine_distortion="radial"` — or an explicit list — instead of the
   all-or-nothing boolean
   (`packages/midas_calibrate_v2/midas_calibrate_v2/forward/distortion.py:49`).
   On that frame even `"radial"` was not enough and `"none"` was required: 181 µε
   diverging → 72 µε. **Check, do not assume:** run the azimuth gate
   (`.../pipelines/diagnostics.py:281`), refine the largest block that passes,
   and confirm the loop settles.

   A second calibrant does **not** help here. Both powders illuminate the same
   wedge, so multi-phase adds rows to the Jacobian, not a new direction.

12. **Set `RhoD` to the outer ring radius, in µm.** The distortion polynomial
   lives in `ρ = R_µm / RhoD`, so `RhoD` is a normalisation, not a measurement —
   but it sets the dynamic range of every radial term. Left far beyond the
   outermost ring, ρ stays small and the high powers collapse: at ρ_max = 0.32,
   ρ⁶ is 1e-03 and `iso_R4` / `iso_R6` came back with 1σ of 0.9 to 15 on
   coefficients of order 1e-03, railed at their bounds. `calibrate()` derives a
   sane value; a *template* may not. Gate: `.../pipelines/diagnostics.py:401`.

13. **A ring table is crystallography, not a measurement of this exposure.**
   Weak, vignetted or grainy rings still produce a centroid per η bin, and those
   centroids are noise the geometry absorbs. Filter rings on what the frame
   actually carries — `MinEtaBinsPerRing` / `MinRingSNR`
   (`.../pipelines/_common.py:127`) — and note the count is absolute, so it
   scales with `EtaBinSize`: a fully-covered ring carried 13 fits at 5° bins and
   ~36 at 2° on the same frame. Read the distribution off `ring_quality()` rather
   than copying a threshold.

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
| a high bad-pixel sentinel (`2**32-1`, EIGER) | `img[img < 0] = 0` passes it straight through; 7 % of the detector enters the fit as 4.29e9 | `read_image(..., return_mask=True)`; hard rule 3 |
| energy taken from the monochromator readback | ~0.04 % low at 1-ID, straight into `Lsd`, gate still passes | use the tabulated K edge of the `fastsweep_Emon` element; rule 9 |
| `exp_setup.yml` `EDGE:` read as the running edge | it is stale — wrong in 9 of 18 beamtimes checked | take the element from `fastsweep_Emon*.txt`; rule 9 |
| `hdf5plugin` declared but not imported | bitshuffle/LZ4 HDF5 (EIGER, ESRF) fails with `can't open directory (/usr/local/lib/plugin)` | importing the MIDAS package registers it; DIAGNOSIS |
| a written paramstest inheriting the template's `PanelShiftsFile` | new geometry silently uses the *previous* calibration's panel shifts | `midas-calibrate-v2 ≥ 0.8.1`; check the line before reusing the file |
| `RhoD` inherited from a template and far too large | radial distortion terms rail; strain still looks reasonable | rule 12; RhoD gate |
| harmonics refined on a narrow azimuthal wedge | coefficients on their bounds, loop oscillates, "best iterate" is luck | rule 11; azimuth gate; H10 |
| two calibrants, exact hkl degeneracies treated as blends | good rings silently excluded as "zero-separation doublets" | merge by d-spacing — `_dedup_by_d`, `rings.py:129` — before any blend rule |
| two calibrants agreeing at a high residual | read as validation; it is two phases on a common noise floor | §4b.4 — absolute number first, ratio second |
| `tx` set to 0 on a panel that is physically mounted rotated | ring radii barely move so the fit converges; the exported file then carries an azimuthal frame rotated from the detector, and `ty`/`tz` are expressed in it | `tx` is not refined — carry it from the panel; downstream η is wrong otherwise |
| a geometry-only paramstest treated as a runnable parameter file | missing `MaxRingRad`, `ImTransOpt`, file/scan keys; a missing `MaxRingRad` is the indexer's ring-array overflow | export from a template, or check the key list before handing it on |

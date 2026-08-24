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

## Order of operations — not optional

Run the phases **in this order**: §0 scope gate → §1 install gate → §2 survey →
§3 dispatch matrix → [§4 calibrate](phase-4-calibrate.md) → §5 integrate →
§6 verify. The gates come first because each one invalidates everything after it:
a survey on the wrong detector class, or a calibration seeded from an existing
block, produces a result that looks fine and is not. §7 halt conditions apply
throughout, and [`HARD_RULES.md`](HARD_RULES.md) applies to every phase.

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

**Run the scope gate first — it is cheaper than everything else here and it
cannot be fooled.** Before deciding anything about the frame, ask whether the
calibrant's rings reach the panel at all at this distance and energy (H12,
`detector_scope_gate`). It needs no image. If it fails, stop: the detector is
the wrong one for this calibrant, or it is parked too far, and every step below
will still appear to work.

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

**The recipe now lives in [`phase-4-calibrate.md`](phase-4-calibrate.md)**, together
with §4b (two calibrants on one exposure). It is one read, in this order: clean the
frame (both sentinel conventions), seed *after* cleaning, then the four-stage fit.

The two things that belong in the spine, because skipping them invalidates the run:
**never seed from an existing geometry block**, and **never substitute
`img[img < 0] = 0` for `read_image(..., return_mask=True)`** — that line is blind to
the EIGER high sentinel.
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
- **H12** Fewer than ~3 calibrant rings reach the panel at this distance and
  wavelength. Run this **before** anything else — it is pure geometry, needs no
  image, and it is the only gate that cannot be fooled by a converged fit:

  ```python
  from midas_calibrate_v2.pipelines.diagnostics import detector_scope_gate
  g = detector_scope_gate(wavelength_A=LAMBDA, Lsd_um=LSD, pxY_um=PX,
                          NrPixelsY=NY, NrPixelsZ=NZ)      # 'fail' ⇒ halt
  ```

  A ring lands at `R = Lsd·tan(2θ)`; the panel reaches only its
  beam-centre-to-farthest-corner distance. When nothing reaches it the fitter
  **still converges**, onto parasitic scatter, and every post-fit gate then
  grades a meaningless answer. Measured on the 1-ID archive: this halts 42 of
  252 exposures, and **26 of those had already produced a plausible-looking
  calibration** — including a whole detector that should never have been in
  scope (§9, "a SAXS detector in a powder archive") and a GE quad parked at
  3300 mm that fitted 480–573 mm.

---

## §8. Hard rules

**Moved to [`HARD_RULES.md`](HARD_RULES.md)** — 20 rules, each written after a silent
wrong answer. Read it once before your first run on a new detector.
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
| **a SAXS detector in a powder archive** | its folder holds `CeO2_*` test frames, so a name-based survey counts it as calibration data; every fit then converges on parasitic scatter | `detector_scope_gate` (H12). At 1-ID the pixirad is SAXS — its real calibrants are glassy carbon and Ag behenate |
| **scanning candidate energies and taking the lowest residual** | the "recovered" energy is confident, reproducible across beamtimes, and wrong; a data-blind constant guess scores identically | rule 9 — the radial distortion block spans the entire signal. Use a recorded distance instead (`Lsd ∝ E`, slope 1.0066) |
| a dark frame whose exposure differs from the signal frame | subtracting a 1 s dark from a 5 s frame moved a fitted `Lsd` 1052 → 578 mm, and it still converged | match dark to signal on **directory and integration time**, else use no dark at all |
| a dark frame containing `NaN` | the subtraction poisons the whole frame; the fit fails with no useful message | guard the dark with `np.isfinite(...).all()` before subtracting |
| a fixed absolute ring-scatter threshold across detector geometries | a 4-panel wedge fails verification that a single panel passes, at identical 100 % ring match | measured over 190 records: single-panel median scatter **0.091 px** (91 % verified), GE quad **0.286 px** (45 %). A 0.30 px cut lands on the quad median, so it halves that class by construction. Threshold per geometry class |
| the detector parked too far for the calibrant | fits land at an arbitrary distance (measured: 480–573 mm against a recorded 3300 mm) with no error raised | H12; and cross-check `Lsd` against the distance recorded in the filename |
| a site mounting convention assumed rather than measured | a "plausible" `Lsd` band rejects good fits and passes bad ones | at 1-ID, measured from fits their own filenames confirm: single panel 0.5–1.9 m, GE quad 1.0–3.3 m. Precision 58 %, recall 49 % on its own — use it as a flag, never as the only check |
| a written paramstest inheriting the template's `PanelShiftsFile` | new geometry silently uses the *previous* calibration's panel shifts | `midas-calibrate-v2 ≥ 0.8.1`; check the line before reusing the file |
| `RhoD` inherited from a template and far too large | radial distortion terms rail; strain still looks reasonable | rule 12; RhoD gate |
| harmonics refined on a narrow azimuthal wedge | coefficients on their bounds, loop oscillates, "best iterate" is luck | rule 11; azimuth gate; H10 |
| two calibrants, exact hkl degeneracies treated as blends | good rings silently excluded as "zero-separation doublets" | merge by d-spacing — `_dedup_by_d`, `rings.py:129` — before any blend rule |
| two calibrants agreeing at a high residual | read as validation; it is two phases on a common noise floor | §4b.4 — absolute number first, ratio second |
| `tx` set to 0 on a panel that is physically mounted rotated | ring radii barely move so the fit converges; the exported file then carries an azimuthal frame rotated from the detector, and `ty`/`tz` are expressed in it | `tx` is not refined — carry it from the panel; downstream η is wrong otherwise |
| a geometry-only paramstest treated as a runnable parameter file | missing `MaxRingRad`, `ImTransOpt`, file/scan keys; a missing `MaxRingRad` is the indexer's ring-array overflow | export from a template, or check the key list before handing it on |

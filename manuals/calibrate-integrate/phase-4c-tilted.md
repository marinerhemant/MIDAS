# Phase 4c — Calibrate and integrate a TILTED detector

> Part of the **calibrate-integrate doc set**. Spine: [`README.md`](README.md). Read
> [`HARD_RULES.md`](HARD_RULES.md) 16 and 17 first.
> **Status: PROVISIONAL.** Exercised on two real LaB6 frames and on synthetic images with exact
> truth. One calibrant, no tiled/multi-panel detector, no depth/parallax. Section 7 says what
> each statement rests on. Numbers below name the file that produced them; `$ANALYSIS` is the
> analysis campaign directory that is deliberately not in this repository.

## §1. When to use this instead of §4

Use it when the detector is tilted by more than about 1–2° (a ring that is visibly an ellipse),
when the default `calibrate()` returned a tilt near zero on a detector you know is tilted
(DIAGNOSIS, "near-zero tilt"), or when `make_seed` found the rings but the fit will not settle.
For an untilted detector §4 is the validated route; running the gate (§3) on its result costs
two seconds and is still worth doing.

## §2. Run it

```bash
python -m midas_calibrate_v2.pipelines.tilted FRAME.tif \
    --wavelength 0.10798 --px-um 200 --calibrant LaB6 --out geometry.json \
    --paramstest paramstest_v2.txt --mask-out mask.tif          # [--im-trans 2] [--nominal-tilt 45 | ty,tz]
# exit code 0 = PASS, 1 = WARN, 2 = FAIL; the paramstest is not written on FAIL
```

```python
import numpy as np, tifffile
from midas_calibrate_v2.pipelines.tilted import calibrate_tilted

raw = tifffile.imread("FRAME.tif").astype(float)        # MIDAS orientation image[row=Z, col=Y]
tc = calibrate_tilted(raw, wavelength_A=0.10798, px_um=200.0, calibrant="LaB6")
print(tc)                                                # verdict is the gate's, not the optimiser's
if tc.verdict != "FAIL":
    tc.write_paramstest("paramstest_v2.txt")            # refuses to write a FAIL geometry
    tc.write_mask("mask.tif")                            # the SAME bad pixels, for midas-integrate-v2 --mask
```

- **The paramstest carries no `ImTransOpt`.** If the detector needs one (20-ID-D HT-HEDM Varex: 2),
  apply it when you read the frame (`read_image(..., im_trans=(2,))`, CLI `--im-trans 2`) and
  integrate that same transformed frame; integrating the raw file with this paramstest mirrors it.
- **`read_image(..., return_mask=True)` flags only integer sentinels.** Negative pixels of a float
  frame (the 45° and 5° frames carry 15,054 and 34,601 of them) come back unmasked, so write the
  mask with `tc.write_mask`, not from the reader, when integrating.

- `wavelength_A` comes from the beamline (rule 9). Nothing here can recover it.
- Optional `lsd_um` (CLI `--lsd-mm`): a distance hint along the beam, e.g. the detector stage readback.
  `make_seed` picks the innermost visible ring by an index match and got it wrong on 6 of 43 tilted
  20-ID-D frames (seed distances 2.4×, 0.85×, 0.68× the truth); a start whose distance is that far
  off does not converge whatever the tilt start (section 4, 20-ID series). A couple of percent is enough.
- Optional `nominal_tilt_deg`: a pair `(ty, tz)` is used as the start; a single number is the tilt
  magnitude with unknown direction and runs 8 starts (45° steps) keeping the lowest fit strain,
  at 8× the time. Give it when the first run FAILs and you know the tilt to roughly 20° / 10 %.
- `calibrate_tilted` cleans the frame itself (negative pixels, dead columns, dilated 2 px) and
  zeroes the bad pixels; pass `mask=` (True = bad) for anything else. For a file reader use
  `read_image(..., return_mask=True)` (rule 3).

## §3. What it does, in order

1. `make_seed` for beam centre and distance. The seed has **no tilt** (it assumes 0); on the
   frames below it was 0.5–1.8 % off in `Lsd` and 11–16 px off in beam centre.
2. `iterate_frozen_point_until_stable`, tilt-blind, `min_ring_gap_deg = 0.08` (the default 0.3
   drops most rings at 115 keV where rings are crowded). It re-centres its bounds every
   iteration, so it can walk many degrees.
3. One `autocalibrate_frozen_point` pass from that geometry with the radial distortion block
   refined; kept only if it does not worsen the gate.
4. The **ring-position gate** (`midas_calibrate_v2.ring_position_gate`), independent of the fit:
   it takes the image and a 2θ map for the geometry under test, fits each ring in each 10° sector
   and reports the radial offset from the ideal ring **in pixels** (the unit is the map's own local
   radial gradient, so it is the true radial pixel on a tilted plane), after removing one global scale
   (so it is blind to `Lsd` and to λ: a free `Lsd` absorbs 0.4 % of either, 1–2 px at the rim). PASS = median < 0.3 px, p95 < 1 px, and at least half of the
   populated (ring, sector) windows found their ring. A wrong geometry loses cells, so the cell
   count is part of the verdict; without that, a wrong geometry's few survivors look fine. Measured
   resolution (noisy synthetic frames, own renderer): a wrong geometry starts to PASS below about
   **0.4 px of centre error or 0.07–0.15° of tilt error**; it is a screen for "rings where the map
   says to well under a pixel", not a calibration of ±0.05 px. More than 20 % of populated cells lost, or
   ≥ 10 % of the azimuth sectors losing half of theirs, is a WARN (a localized error, one panel or one arc,
   would otherwise leave the survivors looking perfect).
5. If the gate FAILs after step 3, `autocalibrate_pv` (wide capture window, minutes) is run from
   the same seed and the better result by the gate is kept (`fallback_pv=True`).

## §4. Measured behaviour

| frame | what happened | gate (median / p95, cells) | time | file |
|---|---|---|---|---|
| 45° LaB6, 0.108 Å, 200 µm | blind; iterate converged, 50 iterations, 558 µε; the radial pass did not improve it | 0.059 / 0.237 px, 494/497 | 121 s | `$ANALYSIS/real_cal_others/tilted_45_none.json` |
| 5° LaB6, 0.728 Å, 200 µm | blind; **iterate did not settle** (60 iterations, 9,495 µε), gate FAIL (1.4 px, 29 % of cells with the first gate version), then `autocalibrate_pv` kept | 0.039 / 0.444 px, 472/472 | 301 s | `$ANALYSIS/real_cal_others/tilted_5_none.json` |

The 5° row is the reason for steps 4 and 5. The same iterated route on the same frame, from a
differently cleaned image and so a different seed, converged in 16 iterations at 150 µε
(`$ANALYSIS/tilt45_lab6/midas/iter24_final.json`; seed Lsd 244.8 mm, 1.8 % short). The seed from
`clean_frame` here (248.05 mm, 0.45 % short) did not. **The iterated route's basin on a low-tilt frame
depends on the seed**, so its `converged` flag and even its success cannot be assumed; only the
gate decides.

**20-ID-D HT-HEDM Varex, 43 files of a detector tilt series** (`$ANALYSIS/ai_tune_rot_series/RESULTS.md`;
2880², 150 µm, 0.1968 Å assumed from the file name, tz about −10.5° to +9.4° in ~1° steps): blind, 37 of 43 gate
PASS (36 clean, median 0.005–0.022 px, 76–535 s each). The 6 FAILs all had a `make_seed` distance wrong by
2.4×, 0.85× or 0.68× (wrong innermost ring); with the stage readback as `lsd_um` (900 mm, tilt still
blind) 5 of them PASS at 0.012–0.021 px, and the 6th (Lab6 000283) PASSes with the nominal tilt. Over the
41 clean files the recovered tz is linear in file number to 0.005 deg (Ceria) / 0.009 deg (LaB6) rms,
BC_y against tz is a straight line (−10.43 px/deg, 0.12–0.16 px rms) and Ceria and LaB6 agree on BC_y at
tz = 0 to 0.14 px. File 000261 passed the first gate narrowly (0.197 / 0.893 px, 63 % of cells) and is a WARN with the
revised gate (12 azimuth sectors lost more than half of their cells), on an off-series geometry; unresolved. This checks the tilt recovery, **not** the wavelength.

Hand-off checked once: `write_paramstest` → `midas-integrate-v2 --mode polygon --mask` on the 45°
frame, then the crest of each of 25 rings in the integrated profile against the ideal radius
`Lsd·tan 2θ / px`: median |offset| 0.024 px, max 0.40 px
(`$ANALYSIS/real_cal_others/e2e_integrate_only.py`, output in `e2e_out/`).

Two negative controls on the 45° frame (`$ANALYSIS/real_cal_others/neg_tests.py`): a start with the
tilt direction flipped (`nominal_tilt_deg=(0, +44.75)` against the true −44.76) ended at 46.6° / 1392 mm
and the gate **FAILed** (1.20 px, 22 % of populated cells found a ring); a wavelength 1.9 % too high
**PASSed** (0.060 px, scale 1.00002) with the perpendicular distance 1.8 % short (1401 vs 1426.8 mm).
The gate sees a wrong geometry; it cannot see a wrong λ or `Lsd` (rule 9). A PASS is not a
validation of the energy.

Seed ladder, other codes and the MIDAS route on the same two frames
(`$ANALYSIS/real_cal_others/RESULTS.md`): the iterated route reaches the reference geometry from
7 of 10 displaced starts on each frame (tilt errors of 20° / 10 % included) and **fails** when the
distance starts 3 % too long or the start is tilt 0 with the beam centre at the image centre.
Read the failure list as the edge of the basin, not as a rare case.

## §5. If the verdict is not PASS

| verdict | do |
|---|---|
| WARN | median 0.3–0.5 px, or the global scale off by > 0.1 % (the map and the ring list disagree by a common factor: check the calibrant lattice constant and units): look at the per-ring medians in `tc.gate.per_ring_median_px`. A 0.3–0.4 px median on a frame with real distortion and radial terms refined is also what a 5-parameter model gives (GSAS-II and pyFAI both sit near 0.3 px on the 5° frame); look at the residual map before calling it a failure |
| FAIL, `gate.reasons` says few cells | the geometry puts the rings elsewhere. Check the seed distance in `tc.seed` against the recorded stage distance: if it is off by more than a few percent, rerun with `lsd_um=` (CLI `--lsd-mm`). Then supply `nominal_tilt_deg` (pair, or magnitude for 8 starts), check the mask and that the calibrant and wavelength are right, and look at the ring overlay (§6 of the spine). Do not integrate with it; `write_paramstest` refuses |
| FAIL after `autocalibrate_pv` too | stop and ask (H13) |

## §6. Integrating a tilted detector

- **One-shot `midas-integrate-v2` rebuilds the polygon map on every call** (245 s and 280 s for the
  two 2048² frames on one CPU core in a context-free run): fine for one frame; for a folder or a
  stream use the batch / server paths (§5b, §5c) so the map is built once.
- **Use `--mode polygon`.** On exact-truth synthetic images (tilts 0–60°, rings 3–30° 2θ,
  noiseless, flat detector) polygon mode has a worst-sector ring-position error of about 14 µε
  at every tilt, with no tilt dependence detectable above ~5 µε. That figure is the worst of the
  cells that pass a validity rule, not a bound on the integrator
  (`$ANALYSIS/tilt4way_hi2th/RESULTS.md`, correction block).
- **Subpixel mode** has a sampling-aliasing artefact at radial bin widths `m/K` px (K ≤ 8;
  `SubpixelResonanceWarning`); polygon is immune. If you must use it, move `RBinSize` by more than
  0.02 px off those widths.
- **Solid-angle and polarization flags are real in v2 binning** (they were ignored before the
  2026-10 fix; `manuals/pdf/HARD_RULES.md` §1). Turn polarization on with the plane set
  correctly. Hold `tx` at 0 in the calibration: it is invisible to the 2θ map, and an arbitrary
  `tx` rotates the η origin and with it the polarization plane (13.5° and 39.4° off in the test
  that found this, 0.7 % spurious modulation).
- **Mask ring arcs cut by the detector edge or a panel gap** before any peak fit. A one-sided
  window biases the centre by −37 to −245 µε in every integrator tested, because the bins are
  thin or empty on one side; the gate drops such cells for the same reason.
- **Do not trust the last ~100 px before the detector edge.** On the 5° frame the integrated profile's
  ring crests drift outward past R ≈ 1100 px (+0.20 px at 1135, +0.73 at 1351, +0.85 at 1408) and a ring
  cut by the panel edge read +3.78 px (context-free run, `crest_check.py` on the polygon profile); the
  gate's max (1.45 px) shows the same. Set `RMax` inside the fully illuminated rings, or integrate in
  sectors.
- To compare intensities across distances use the opt-in `IntensityRefDistance_um`
  (multiplies by `(Lsd/L_ref)²`; `L_ref = 1e6` µm reproduces GSAS-II's `×D²` convention).

## §7. What each statement rests on

| statement | status | evidence |
|---|---|---|
| default `calibrate()` fails on a tilted detector | ESTABLISHED at 14° (RUNBOOK); on the 5° frame it returned ~0.3° tilt and 2,478 µε with distortion railed (one frame, not /verify'd) | `$ANALYSIS/tilt_2_4_lab6/RESULTS.md` |
| gate agrees with an independent scorer, rejects wrong geometries, and is blind to λ and `Lsd` | /verify (claim 08b931f37783): PROVISIONAL. First version: 5 good geometries within 0.035 px of the analysis scorer (largest 0.031), 4 gross wrong geometries FAIL with 18–33 % of cells. The verification found and I fixed: the pixel unit was exact only for the along-beam distance (now the map's own gradient), duplicate hkl rings dropped rings, a localized error over ≤ 40 % of the azimuth PASSed (now WARN), the scale warning limit was 0.5 % (now 0.1 %); 13 unit tests against an independent renderer | `tests/test_ring_position_gate.py`, `$ANALYSIS/real_cal_others/validate_new_gate.py` |
| the iterated route finds a tilted detector from a tilt-blind seed | PROVISIONAL: 45° frame yes, 5° frame no (needs the pv fallback); /verify of the iterated route (claim ac3c74cade55): physics and reproduction SURVIVE, statistics and artifact UNCERTAIN | `$ANALYSIS/tilt45_lab6/RESULTS_verify_iterate.md` |
| polygon is accurate on a tilted detector | PROVISIONAL: synthetic, noiseless, flat; claim b6dc893858bd and d6331b463cc6, both PROVISIONAL after /verify | `$ANALYSIS/tilt4way*/RESULTS.md` |
| the whole chain end to end | one frame (45°), once | `$ANALYSIS/real_cal_others/e2e_*.py` |

**Not exercised — stop and ask:** a tiled/multi-panel detector that is also tilted; depth or
parallax; a calibrant other than LaB6 on a tilted detector; a tilt-free start when the tilt is
over ~14° and the distance is off by more than a few percent; EIGER2.

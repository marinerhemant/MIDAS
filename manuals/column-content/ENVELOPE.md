# ENVELOPE — what column content can and cannot determine

Every number below is a synthetic-validation readout or a real-data description, with its source registration (in the
development workspace; `LAB_NOTEBOOK.md` lists them). Treat a real result outside these conditions as unvalidated.

## 1. Laue (white beam, 34-ID-E-type reflection geometry, Zn hcp)

**Narrow validation** (point crystals and <= 0.05 deg clouds, 2-8 crystals per frame, 240 frames;
`PREREGISTER_column_content.md`):

| quantity | value |
|---|---|
| false orientations | 0.0125/frame (UB 0.032) |
| recall (share >= 5%) | 93.8% |
| share error | 8.4% median |
| point-crystal orientation | 0.0013 deg |

- Recall by share: < 2% -> 30%; 2-5% -> 66%; 5-10% -> 84%; 10-20% -> 94%; > 20% -> 99%.
- Near pairs reported as two: 0.05 deg 0%; 0.1 deg 0%; 0.3 deg 10%; 1 deg 42%; 3 deg 73%. Pairs below ~0.3 deg
  merge into one wider cloud.
- **Per-frame completeness number: NOT available.** C_int does not track missed crystals (Spearman 0.18). The
  unexplained flux bounds the missed share in only 53% of frames with a miss.

**Wide validation** (1-D streaks 0.1-0.8 deg and 3-D clouds 0.2 deg added; `PREREGISTER_column_content_wide.md`):

| quantity | value |
|---|---|
| false orientations | 0 / 240 frames |
| share error | 15.9% |
| point-crystal orientation | 0.0027 deg |
| unexplained median | 0.138 |
| spread ranking (Spearman) | 0.93 |

- Spread recovery (true -> recovered): 0.052 -> 0.053; 0.104 -> 0.106; 0.168 -> 0.153; 0.219 -> 0.184 deg. It
  underestimates beyond ~0.2 deg.
- **Recall is limited by DISCOVERY, not the fit.** Point 96%, narrow 88%, streak 91/72/55/38% at half-width
  0.1/0.2/0.4/0.8 deg, wide 3-D 15%. The C indexer scores point-like matches.

**Real sampleH (Zn electrodeposit on rolled Zn; 200 frames):**
- 1-4 compact orientations per frame;
- median unexplained 0.474 (narrow fit) and 0.421 (wide fit);
- the unexplained intensity is mostly curved, BEADED arcs, which are NOT the spread of found crystals.

## 2. Monochromatic rotation (DAC-type rasters)

**Registered read: NOT VALIDATED (gate V2, recall).** Six of the seven gates pass. Verification status: **PROVISIONAL**. All four
`/verify` lenses ran: independent reproduction and the artifact lens SURVIVE (every number re-derived from scratch); the statistics
and physics lenses are UNCERTAIN (the V2 miss is within noise of the bar; the shortfall is weak domains AND 1-deg pairs, both stated
below). No lens refuted it.

Conditions (`PREREGISTER_mono_validation.md`): 240 synthetic columns; Pilatus-2M-type detector, 172 um, Lsd 350 mm,
0.4246 A; 14-frame (+/-7 deg, "S5-like") and 40-frame (+/-20 deg, "2604-like") wedges, alternating; tetragonal La3Ni2O7-type
cell (SG 139, a = 3.6008, c = 19.2522 A); 1-4 domains (60 columns each); 40% of extra domains a near partner at 1, 3 or 14 deg;
spread classes point (35%), cloud sigma 0.1/0.3 deg (30%), streak half-width 0.2/0.5/1.0 deg (35%); a foreign
diamond-type crystal in about half the columns; three powder rings. The kernel SHAPE was the true one (only its width scale
was fitted). Round 0 was `find_domains` on the known cell.

| gate | bar | result | |
|---|---|---|---|
| V1 false orientations | 95% upper bound <= 0.10 per column | 7 false in 240 columns (6 of them in foreign-crystal columns), bound 0.055 | pass |
| **V2 recall, share >= 5%** | **>= 0.90** | **0.887 (494 of 557)** | **FAIL** |
| V3 share error | median <= 0.20 | 0.022 (n = 464 one-to-one matches) | pass |
| V4 point-domain orientation | median <= 0.05 deg | 0.0008 deg (n = 158) | pass |
| V6 unexplained flux | median <= 0.15, no foreign crystal | 0.013 (n = 116); 0.047 with a foreign crystal | pass |
| V7 spread-domain recall, share >= 5% | >= 0.85 | 0.894 (n = 378) | pass |
| V8 spread ranking | Spearman >= 0.6 | 0.882 (n = 312) | pass |

Source of every number above: `eval.json` (stage D of `mono_val.py`, run on the 240 result files). A column-level bootstrap
(5000 resamples) puts V2 at 95% CI [0.855, 0.916]: **the interval contains the 0.90 bar, so the shortfall is a point-estimate
miss of 1.3 points, not a statistically separated one** (fraction of resamples >= 0.90: 0.198). The other gates are robust passes
by the same bootstrap.

**Recall by intensity share of the domain (share >= 5% population plus the 2-5% bin):**

| share | recall | n |
|---|---|---|
| 2-5% | 0.49 | 43 |
| 5-10% | 0.63 | 79 |
| 10-20% | 0.79 | 97 |
| >= 20% | 0.96 | 381 |

Recall at share >= 10% is 0.929 (n = 478). That cut was made AFTER the read and is descriptive only; it does not rescue V2.

**Where the misses are** (descriptive; `mono_read_extras.py`, part a and a2). Two things, not one:
- *Weak domains.* By rank within a column, the brightest domain is found 0.97 of the time, the second 0.89, the third 0.75, the
  fourth 0.70.
- *Near pairs 1 deg apart.* 81 of the 557 domains belong to a 1-deg pair; their recall is 0.68 as scored, and 26 of the 63 misses
  are theirs. Every other domain is found 0.92 of the time (n = 476; its bootstrap interval [0.89, 0.95] includes the 0.90 bar).
  Only 9% of 1-deg pairs are reported as two orientations. Read the 26 with care: keeping column difficulty fixed, about 21
  pair-member misses are expected by chance, so pair membership adds roughly 5; weak share (< 10%, 29 of the 63 misses) is an
  equally strong split, and the effect is specific to 1 deg (3 deg and 14 deg pairs show none). Both effects are real and they
  are confounded; they were not separated. The split was made after the V2 read, among ~15 comparable ones, without adjustment.
  **The scoring rule flatters this:** a domain counts as found when ANY recovered orientation lies within its tolerance, so one
  recovered orientation can be credited to both members of a merged pair. With a strict one-to-one assignment, V2 recall is
  **0.864 (481 of 557)** rather than 0.887, and the 1-deg-pair members drop to 0.52. The registered read uses the as-scored
  number; the strict one is the more honest completeness figure.

Other splits (as scored): class point 0.87, cloud 0.1 deg 0.86, cloud 0.3 deg 0.91, streak 0.2 deg 0.79, streak 0.5 deg 0.96,
streak 1.0 deg 0.95. N = 1-4 domains: 0.92, 0.93, 0.87, 0.87. Wedge: 14-frame 0.87, 40-frame 0.90. A foreign crystal in the
column makes no difference to recall (0.88 vs 0.89) but its peaks account for 6 of the 7 false orientations and raise the
unexplained median from 0.013 to 0.047. The two effects are confounded (more domains means more weak domains and more pairs);
they were not separated.

**Spread** (rms of the fitted sub-orientation cloud, deg; true median -> recovered median, one-to-one matches with share >= 5%):
cloud 0.1: 0.170 -> 0.177; cloud 0.3: 0.509 -> 0.502; streak 0.2: 0.118 -> 0.128; streak 0.5: 0.296 -> 0.298;
streak 1.0: 0.592 -> 0.591. **Floor:** a point domain reads a median spread of 0.042 deg (90th percentile 0.169 deg), so a
recovered spread below about 0.05 deg is "not resolved from zero".

**Near pairs reported as two orientations:** 1 deg 9% (n = 46); 3 deg 74% (n = 50); 14 deg 76% (n = 58). Pairs near 1 deg merge.

**Discovery gate:** g_disc = 5, the LOWEST gate tried: the scrambled-residual chance count was 0 at every gate in
{5, 7, 9, 11, 13} over 755 column-draws (95% bound 0.004 per column), so the gate was not constrained from below. Chance rates
below gate 5 were not measured.

**Kernel estimation** (`estimate_kernel` on round-0 isolated spots, n = 236 columns): recovered / true width, median
0.84 (frame), 0.81 (radial), 1.00 (tangential). The registered run used the estimator BEFORE the threshold-bias calibration
was added; the calibrated estimator (the package default) recovered widths to within ~3.5% on three development columns
(development check, not a registered gate).

**Two columns needed a changed solver** (PREREGISTER Amendments 1-2, before any result was read). Columns 219 and 221 (40-frame,
4 domains, wide streaks, ~1.1e6 voxels) had not finished after more than 22 h with the original dense linear step.
Column 219 finished on the original solver at 26 h and is scored from it; column 221 is scored from a re-run with the
block-structured solver. On column 219 both solvers agree: same number of orientations and rounds, shares within 1e-4, spreads
within 5e-4 deg, mean orientations within 7e-5 deg (`mono_read_extras.py`, part c).

**Cost.** With the original solver a typical column took about 3 CPU-hours on one core and the heaviest ones more than a
day. With the block-structured linear step (now the default; `fast_linear=False` restores the dense path) the model it produces
equals the dense path's to within 3e-13 relative on every input tried (one benchmark input gave 1e-14). The linear step is
faster by 3-5x at 2 orientations and 7-9x at 4-6 orientations (column 221, 8 threads; four independent measurements ranged 3.1x to
9.2x), because the dense cost grows with the square of the number of orientations. The whole optimiser step is about 2.5x faster at
6 orientations (14.1 s vs 34.7 s, per-step ratios 2.2-2.8); on a small column (~1.4e5 voxels) a whole fit was only ~1.4x faster.
The two heaviest columns finished in 65 and 90 minutes on 16 threads. What remains is mostly the kernel forward/backward pass
(8.0 of 14.1 s per step at 6 orientations).
Timings move by a factor of about 2 with host load; treat the ratios, not the seconds, as the result. Budget memory too: the heaviest columns peaked at 13-18 GB per process.

**Limits (as registered):** one cell, one space group, one detector class; the true kernel shape; no saturation or count-rate
model (mask saturated voxels); no absorption variation with omega beyond the free per-reflection brightness; mean
orientations only as precise as the few observed reflections allow (~29-46 predicted per orientation in these wedges, only a
subset detected). Synthetic truth was rendered with the same geometry primitives the fit uses.

**Reading this section:** on this synthetic set `column_content` finds the bright domains reliably and puts the ones it finds
close to the truth in orientation and share. It does not find every weak domain, and it cannot separate two domains 1 deg
apart. Treat a missing domain below ~10% intensity share, or a missing partner within ~1 deg of a found domain, as "not
detected", not "absent". The completeness statement is the recall table above, not a per-column number. The passes
rest on synthetic data drawn from the fit's own class (true kernel shape, shared geometry primitives, Gaussian or uniform spreads),
so they are upper bounds on real-data performance, not validated real-data accuracy.

## 3. What this method cannot determine (measured, not assumed)

- **Volume fractions.** Only intensity shares are measured.
- **A spectrum-constrained brightness that closes the budget (Laue).** The per-reflection log scatter about
  S(E)·|F|²·L·P·Ω·A was 0.74-0.79 with honest statistics (bar ln 2 = 0.69). A corrected spectrum passed only through
  a selection artifact (0.911 on a model-independent spot set). Coincident missed crystals stay invisible to the
  per-spot brightness.
- **Per-frame lattice-rotation axes of arcs on a 34-ID-E-type panel.** The limited cone of plane normals makes the
  axis degenerate; fresh controls gave an axis error of 12.6 deg. A pooled common axis can be DETECTED, but located
  only to a band.
- **Single-grain mosaic.** A spread is the column's spread for that family (KL-119).
- **Depth.** Deposit and substrate cannot be separated from one frame.

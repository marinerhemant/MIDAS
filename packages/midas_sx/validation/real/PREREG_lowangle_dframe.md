# PREREGISTER (fork A, A39 side question): the low-angle omega (frame) offset between blob centroid and calculated Ewald crossing

Registered 2026-10-23, BEFORE the discriminating analysis below. What I had already seen: the r02 top-2000 table (`lalanine/out_p0/r02/diag_unindexed_top2000.csv`, from `a39_diag_unindexed.py`) and a descriptive tabulation of its matched blobs (dist < 12): median dframe (blob - prediction, frames) by 2theta bin = -0.66 (<10 deg, n 19), -0.27 (10-15, n 51), -0.30 (15-20, n 90), -0.19 (20-25), -0.09 (25-30), +0.02 (30-40), +0.22 (40-60, n 766). r02 is therefore the DISCOVERY run; r03 and r04 (not looked at for this) are the REPLICATE.

**Hypothesis.** Strong blobs at low 2theta sit ~0.3-1 frame (0.06-0.2 deg omega) before their calculated Ewald crossing. Two explanations: (CAL) a real calibration error (omega zero, step, rotation-axis tilt, timing) or (CEN) a measurement artefact: the spot-finder centroid of an omega-extended, asymmetric profile is not the Bragg crossing.

**Known-limits check** (`~/.claude/known-limits.md`): nothing applies to a rotation-method omega offset on this system. KL-064 (centroid on a pedestal measures the pedestal) is a hypothesis for CEN, not evidence; tested here on the raw frames, not assumed.

**Data/definitions.** obs = `lalanine/out_p0/r0N/obs.pkl` (columns omega = calculated crossing, omega_meas = measured centroid, L, tth, eta, frame) joined to `r0N/spots_blind.csv` (skew_omega, omega_width_frames, n_frames, full_n_sub_peaks) on (row, col, frame). dframe = (omega_meas - omega)/omega_step. Clean sample: full_n_sub_peaks == 1, integrated > 1000, |dframe| < 4. "Low angle" = tth < 20 deg. Skew tertiles are defined within tth < 20 per run.
Raw check: 12 offending strong spots (r02, tth < 20, dframe < -0.8): box-sum profile over frames around the blob (rows/cols = extents of the blob, local background from a ring) read from `lalanine/frames`; compare the profile's peak frame and intensity centroid with blob frame and calculated crossing.

**Null.** No systematic offset: |median dframe| < 0.15 frame at tth < 20 in BOTH replicate runs.

**Outcomes (mutually exclusive, in this order).**
- REFUTED (the lead is not a systematic effect): null holds in r03 and r04.
- CEN (centroid artefact) CONFIRMED: replicate median dframe at tth < 20 is <= -0.2 in both r03 and r04, AND in the lowest-|skew_omega| tertile it is within +-0.15, AND the partial slope of dframe on skew_omega (OLS with tth bin, log n_frames, L as controls) has |t| > 5 with the same sign in r02, r03, r04, AND in the raw check the box-sum profile centroid agrees with the blob frame to < 0.3 frame in >= 9 of 12 spots (the blob is faithful to the intensity; the asymmetry is in the profile).
- CAL (real calibration) CONFIRMED: replicate median dframe at tth < 20 is <= -0.2 in both runs and within 0.2 frame of each other and of r02, AND the offset in the lowest-|skew| tertile stays <= -0.2 (skew does not remove it), AND partial slope on skew has |t| < 3.
- otherwise INCONCLUSIVE (numbers reported).

**Effect size that matters.** |dframe| >= 0.3 frame (0.06 deg omega), sustained over a class of strong reflections, because it would move the refined wedge/orientation (the refinement fits the blob centroids) at the 0.01 deg level; a constant 1 frame shift of the integration window does not matter (support half-width = 3.6 + 0.25 L + margin 6 frames).

**Not claimed:** anything about high-angle spots; any change to the package until a defect is shown with a test.

## READING (2026-10-23). PROVISIONAL. Registered verdict: INCONCLUSIVE. Post hoc finding (NOT registered): a wavelength/cell-scale mismatch of 0.19 % removes the offset.
Data on copland `/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/midas_sx_real/lalanine/` (`out_p0` = as-registered reductions, plane 0, margin 6); copies of the small outputs in `validation/results/a39_lowangle/`. Scripts in `validation/real/`: `a39_diag_dframe.py`, `a39_diag_dframe2.py` (registered statistics), `a39_diag_rawprofile.py` (raw check), `a39_diag_dframe3.py`, `a39_diag_extfit.py` (post hoc).

**Sign/units.** dframe = frame(blob) - frame(calc crossing), frames of 0.2 deg; omega runs backwards (slope -0.20000 deg/frame, read from the data). Negative = blob before the crossing.

**Registered statistics** (clean sample: integrated > 1000, single sub-peak, |dframe| < 4; 3117 of 3937 obs; `a39_diag_dframe2.py out_p0`):
| run | tth < 20 median dframe (bootstrap 95 %) | lowest-\|skew_omega\| tertile median | OLS skew slope (t) |
|---|---|---|---|
| r02 (discovery) | -0.338 (-0.49, -0.18), n 44 | -0.184 (n 15) | +0.109 (+2.1) |
| r03 | -0.289 (-0.39, -0.23), n 59 | -0.274 (n 20) | -0.219 (-5.5) |
| r04 | -0.156 (-0.30, -0.02), n 55 | -0.147 (n 19) | -0.026 (-1.8) |
- REFUTED (null |median| < 0.15 in both replicates): not met (r03 -0.289; r04 -0.156 is just outside).
- CEN: not met (replicate r04 median -0.156 is not <= -0.2; skew slope t = +2.1 / -5.5 / -1.8, not |t| > 5 with one sign; raw check below).
- CAL: not met (r04 -0.156 is not <= -0.2; r03 skew slope |t| = 5.5, not < 3).
- Verdict as registered: **INCONCLUSIVE.**

**Raw check** (r02, `a39_diag_rawprofile.py`; box-sum omega profile from `lalanine/frames`, 12 offending spots, 8 controls; `out_p0/r02/rawprofile_check.csv`): the intensity itself peaks before the calculated crossing for the offending spots (peak - calc median -1.90 frames, range -3.47 to +0.09; blob - calc median -1.60); controls: peak - calc -0.29 (range -0.59 to +0.55), blob - calc +0.06. The blob centroid is near the raw peak (peak - blob median -0.47 for both classes). Registered test "profile centroid within 0.3 frame of the blob in >= 9 of 12": 0 of 12, because the profiles are strongly skewed with a long tail to LOWER frames (profile skewness negative in all 20 spots), which a 5 %-threshold centroid picks up; half-max midpoint within 0.3: 4 of 12. So the offset is in the raw frames, not made by the spot-finder (this part is a reading, not a registered pass/fail).

**Where it lives (post hoc, `a39_diag_dframe2.py`, `a39_diag_dframe3.py`, pooled runs):** not a function of tth or L alone. By detector column: cols < 420 (eta < 0, the side of the beam away from the detector swing) dframe about -0.3 to -0.5 at every row band and every tth (col<487: -0.42, n 126 at tth 20-30; col>=487: -0.03, n 382); the same sign and size in all three runs (col<487 medians -0.38 / -0.36 / -0.35 in the first 210 px). eta>0 spots at tth > 30 sit at 0 to +0.19.

**Post hoc fit (`a39_diag_extfit.py`, own predictor; reproduces `orient.predict_positions` to 1e-12 at p = 0):** the package parameter set (U, centre, Lsd, tilts, wedge) plus one extra, refitted on the assigned obs of each run:
| run | extra | soft-L1 cost | robust rms px / frame | model - blob frame, col<487 / col>=487 | fitted value |
|---|---|---|---|---|---|
| r02 | none (M0) | 15682 | 0.451 / 0.526 | +0.412 / -0.091 | |
| r02 | wavelength scale | 15148 | 0.418 / 0.463 | -0.064 / -0.052 | +2.00e-3 |
| r03 | none | 14561 | 0.534 / 0.361 | +0.412 / -0.017 | |
| r03 | wavelength scale | 14086 | 0.506 / 0.329 | +0.042 / +0.011 | +1.82e-3 |
| r04 | none | 6276 | 0.311 / 0.269 | +0.415 / -0.007 | |
| r04 | wavelength scale | 5656 | 0.292 / 0.261 | -0.062 / +0.033 | +1.80e-3 |
An omega-step (rotation speed) scale changes almost nothing (col<487 stays +0.35 to +0.42); a sideways rotation-axis tilt V has no effect on the cost (degenerate with the detector roll tx). The three runs fit the same wavelength scale independently (+1.8 to +2.0e-3, i.e. 0.6889 -> 0.6902 A, 17.998 -> 17.965 keV). Mechanism: the two Ewald crossings of one reflection have opposite sign of the diffracted y direction (eta > 0 vs < 0); a wavelength error shifts them in opposite senses in omega, so it shows up as a left/right frame offset.
**Degeneracy, stated plainly:** with the cell fixed at COD 1574527, a wavelength scale +epsilon is EXACTLY degenerate with an isotropic cell scale -epsilon (everything depends on lambda times |q|). Three free cell lengths instead of the wavelength fit equally well (cost within 0.3 to 1 %) and come out isotropic: scales -0.203/-0.200/-0.223 % (r02), -0.179/-0.192/-0.150 % (r03), -0.216/-0.222/-0.187 % (r04), i.e. a b c = 5.7915 5.9476 12.2703 (r02), 5.7929 5.9481 12.2793 (r03), 5.7908 5.9463 12.2748 (r04) against COD 5.8033 5.95953 12.2978. This data cannot say whether the header wavelength or the reference cell is off by 0.19 %; only that they are inconsistent by that much.

**Effect of the correction on A39 (re-reduction with `a39_run.py --lambda-scale 1.0019`, plane 0, margin 6, `lalanine/out_p0lam`; post hoc, chosen from this data):**
- dframe: tth < 20 medians -0.113 / -0.075 / +0.070 (null now holds in r03 and r04); col<487 at tth 20-30: +0.05 (was -0.42).
- orientation fit: robust rms 0.575 / 0.681 / 0.451 px and 0.644 / 0.443 / 0.384 frames (before 0.617 / 0.710 / 0.479 and 0.698 / 0.486 / 0.415); refined Lsd 170.74 / 170.70 / 170.69 mm (was 171.14 / 171.11 / 171.08: the distance absorbed 0.2 % of it).
- merge (`a39_merge.py out_p0lam`): R_int scaled 15.67 % (was 15.67 %): unchanged. Completeness n >= 2 25/34 (was 26/34).
- F2 vs COD (`a39_fit_weights.py out_p0lam`): unweighted fit R_F2 18.8 % all, 14.6 % top-100, B +0.82 (before 19.0 / 15.0 / +0.78): unchanged. The sigma-weighted fit moved from R_F2 55 % / B -1.63 to 15.6 % / B -0.03 (top-100 17.3 %), which says that fit is fragile (a few small-sigma reflections decide it), not that the data got better.
- Tier C refinement NOT rerun.

**Not claimed / limits.** The wavelength-scale explanation is one model that removes the pattern; it was found after looking at the data and applied to the same data (a forking path: one parameter, fitted on three runs that agree, but not validated on another data set or against an independent energy calibration). Not tested: whether a different effect with the same left/right signature (e.g. a beam-energy offset that differs per run is excluded by agreement, but an asymmetric beam spectrum or a detector effect on one half is not). No package defect found; `midas_sx` has no wavelength (or cell-scale) refinement in `index_and_refine`, which real data of this kind needs; not added. No package file was changed, nothing committed.

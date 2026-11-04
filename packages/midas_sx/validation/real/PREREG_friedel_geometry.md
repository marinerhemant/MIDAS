# PREREG: Friedel-pair geometry estimator (tx, wedge) for A39, fork B. Registered 2026-10-23 BEFORE any real pair residual was computed.

What I had seen before registering: only the counts from `a39_diag_friedel.py` (in-run pairs by hkl label: 76 / 114 / 94 observed for runs 02 / 03 / 04) and the per-run orientation-fit geometry in `lalanine/out_p0/r0N/reduction.json` (tx -0.103 / +0.023 / -0.036 deg, wedge +0.413 / +0.310 / +0.291 deg, tz about -20.08, ty about -0.19, Lsd 171.1 mm).

## Method (U-independent)
For a true Friedel pair (reflection hkl at omega1 and its mate -h-k-l at omega2, any crossing of each), the sample-frame vectors recovered from the two spots,
q_s,i = R_z(-omega_i) R_y(W) k (s_i - s0)   [package convention q_lab = R_y(-W) R_z(omega) q_s, `midas_defect.geometry.qlab_to_qsample`],
satisfy q_s,1 + q_s,2 = 0 exactly. The relation holds for ANY pair of crossings (in particular pairs with omega separation 2 asin(lambda |c| / rho) >= 2 theta_B, not only 180 deg). It contains no U, no cell. Unknowns: tx, ty, tz, wedge, beam centre (row, col), Lsd. The global omega zero is a gauge (a common shift of omega leaves the sum at zero) and is not fitted. Robust least squares (soft L1) on the 3 components per pair, bootstrap over hkl groups for sigma, SVD of the Jacobian for the identifiable subspace. Start = header geometry (swing_geometry start), NOT the orientation-fit values; start = orientation-fit values is run as a basin check. Labels (hkl) come from the orientation-fit assignment in `obs.pkl`, so a wrong label only adds an outlier; positions used are the measured centroids (row, col, frame -> omega).

## Hypothesis
Pair-based estimates of tx and wedge agree with the orientation-fit values of the same run (the orientation fit is NOT independent of the labels but is independent of the pair estimator's data path in its use of U).

## Null
At the real pair count the estimator carries no information on tx / wedge: (a) estimates from scrambled mates (spot i paired with a random non-mate spot) are as close to the orientation-fit values as the true-pair estimates, or (b) sigma(tx) >= 0.10 deg or sigma(wedge) >= 0.15 deg so nothing can be tested.

## Positive control (must pass BEFORE the real read; `friedel_sim.py`)
Simulation with an INDEPENDENT forward model (own Ewald root finding on a fine omega grid, own closed-form ray-plane intersection with the tilted detector; no `ewald_crossings`, no `qlab_to_pixel`), truth tx -0.10, ty -0.19, tz -20.07, wedge +0.41 deg, centre (861.9, 401.3), Lsd 171.14 mm, random orientations from `midas_stress.orientation_sampling.uniform_rotations`, centroid noise 0.4 px and 0.3 frame, 50 repetitions, start at header geometry. PASS: for tx and wedge |mean bias| <= 0.02 deg and <= 0.5 x empirical sd; empirical sd / reported sigma in [0.7, 1.4]; 68 % coverage in [55, 80] %. A second truth (tx +0.30, wedge -0.30) and a zero truth (tx 0, wedge 0) are recovered the same way. Null in simulation: scrambled mates give scatter around truth >= 3 x the true-pair scatter (shrink placebo: the estimate does not simply return the start).

## Read of the real data (per run, in-run pairs; cross-run pairs only through the extension below)
- CONFIRMED: in >= 2 of 3 runs both |tx_pair - tx_fit| <= max(2 sigma_pair, 0.03) and <= 0.07 deg, and |W_pair - W_fit| the same, with sigma_pair(tx) <= 0.05 deg and sigma_pair(W) <= 0.10 deg, and the scrambled-mate null does not reproduce the agreement.
- REFUTED: in >= 2 of 3 runs a difference > 3 sigma_pair AND > 0.10 deg in tx or in wedge, positive control passed.
- INCONCLUSIVE: everything else, including sigma too large (null b).
Effect size that matters: tx 0.1 deg (1.7 px at 1000 px from the beam), wedge 0.3 deg. Run-to-run spread of the orientation-fit values (sd 0.063 deg in tx, 0.065 deg in wedge) is the tolerance scale I accept.

## Extension (cross-run), only if time: same-hkl and Friedel pairs between runs a and b need q_s,b = P q_s,a with P the unknown phi-stage rotation (3 nuisance parameters per run pair, geometry shared); not part of the verdict above.

## Known-limits check (`~/.claude/known-limits.md`)
Nothing in the ledger concerns tilt / wedge from Friedel pairs. KL-039 (inverse crime) applies to the design and is why the control uses an independent forward model. Memory note `feedback_tilt_gauge_tx_unconstrained_by_2theta` concerns 2theta-only fits; the pair relation is eta-resolved, not covered by it. `midas_calibrate_v2/friedel.py` already has a tx-from-pairs estimator for an untilted detector with a vertical axis (not usable here: swung detector, wedge); this one generalises it.

## Not claimed
Anything about the absolute omega zero (gauge), about cross-run geometry, or that the orientation-fit tilts are right if the two disagree (the disagreement is reported, with the control).

---
## READ (2026-10-23). Registered verdict: **INCONCLUSIVE** (PROVISIONAL, no /verify). Numbers: files under `validation/results/friedel/`; code `midas_sx/friedel_geometry.py` (+ `tests/test_friedel_geometry.py`, 7 passed: `pytest tests/test_friedel_geometry.py -p no:cacheprovider` in the package dir).

### Things that changed after the registration (all before the real read; none of them tuned on the real data)
1. **The raw residual q_s1+q_s2 has a shrink minimum** (Lsd -> infinity puts every s at s0 and the sum goes to zero for scrambled mates too). Found in the unit test of the scrambled null. Fix: the residual is divided by the pair's mean |q| (`pair_residual`).
2. **Wrong hkl labels** (5 % wrong labels: tx error 0.006 -> 0.033 deg, 15 %: 0.17 deg, `results/friedel/friedel_sim_labels.txt`). Fix: gross-outlier trimming at 5 sqrt(3) robust sd (`trim=5`); with it 0.008 deg at 5 % and 15 % (`friedel_sim_labels_trim.txt`).
3. **Pair counts**: the real in-run pairs are 35 / 57 / 47 (runs 02 / 03 / 04, hkl_residual <= 0.1), not "76 / 114 / 94" (that number counted both members of each pair). In-run Friedel pairs in a 170 deg run are not 180 deg apart: crossings of G and -G are 2 asin(lambda |c|/rho) >= 2 theta_B apart, the relation q_s1+q_s2 = 0 holds for any pair of crossings.

### Positive control (independent forward model `friedel_sim.py`; its tilt convention reproduces the package's `pixel_to_qlab` direction to 4e-16)
Final estimator, 50 reps per truth, ~100 pairs, 0.4 px / 0.3 frame (`v2/friedel_sim_read.txt`): bias tx -0.0013 / +0.0016 / 0.0000 deg, wedge +0.0001 / +0.0013 / -0.0006 deg (truths S1 / S2 / S3 = run02-like, tx +0.30 W -0.30, zero); empirical sd tx 0.0091 / 0.0080 / 0.0086, wedge 0.0112 / 0.0099 / 0.0094 deg. All bias and sd criteria pass.
**sigma calibration did NOT pass uniformly.** Formal sigma on tx is 25 to 30 % too large (sd/sigma 0.77 / 0.70 / 0.76, 68 % coverage 0.84 / 0.84 / 0.80 against the registered limit <= 0.80): conservative, but a miss. Wedge formal sigma passes (coverage 0.72 / 0.76 / 0.74). Bootstrap sigma over hkl groups (50 reps, 40 boots, `v2/friedel_sim_boot50.log`): tx passes (sd/sigma 1.02 / 0.87, coverage 0.70 / 0.74), wedge S2 coverage 0.52 (< 0.55). **Declared choice, made after seeing this: tx uses the bootstrap sigma, wedge the formal sigma; the control passes under that choice and fails under a single uniform sigma.**
Scrambled mates: median estimate = the start (0.000), rms error about the truth 1900 to 6800 times the true-pair error, residual rms 0.69 against 0.0017 (shrink placebo passed).

### Real data, in-run pairs (`real/friedel_geometry_r0N.json`, `real/friedel_read.txt`, `a39_friedel_read.py`; start = header geometry, the orientation-fit start gives the same solution to 4 digits)
| run | pairs | tx pair (sigma boot) | tx orientation fit | wedge pair (sigma formal) | wedge orientation fit |
|---|---|---|---|---|---|
| r02 | 35 | -0.072 (0.016) | -0.103 | +0.416 (0.015) | +0.413 |
| r03 | 57 | +0.020 (0.020) | +0.023 | +0.307 (0.013) | +0.310 |
| r04 | 47 | -0.026 (0.005) | -0.036 | +0.256 (0.008) | +0.291 |
Rules applied mechanically: runs meeting EVERY registered condition = **1 of 3** (r02). r03: the scrambled-mate null (tx 0.000) is already within 0.03 deg of the fit value (+0.023), so its tx agreement does not count. r04: wedge differs by 0.035 deg, tolerance max(2 sigma, 0.03) = 0.030. The REFUTE condition (> 3 sigma and > 0.10 deg) is met nowhere. -> not CONFIRMED, not REFUTED: INCONCLUSIVE. Descriptively (not the verdict): every tx and wedge pair estimate lies within 0.035 deg of the orientation-fit value, and wedge is 10 sigma or more from 0 in all three runs.
Other parameters (not in the registered question): tz -20.14 to -20.15 (fit -20.07 to -20.09), Lsd 170.80 to 170.89 mm (fit 171.08 to 171.14), the weakest direction of the Jacobian is tz (0.99) in every run.

### Extension, cross-run (not in the verdict; `midas_sx.friedel_geometry.fit_multirun`; positive control `friedel_sim_cross.py` independent forward, 30 reps, 324 groups (real: 922): bias <= 0.0035 deg, sigma_boot sd/sigma 1.04 (tx) / 0.90 (wedge), coverage 0.63 / 0.67)
Relation: runs of one crystal at different phi differ by an unknown sample-frame rotation P_k, so Friedel pairs across runs satisfy P_a^T q_s,a(h) + P_b^T q_s,b(-h) = 0; 922 hkl groups, 895 with a pair spanning runs (`real/friedel_crossrun_v2.log`).
- **Shared geometry:** tx -0.031 +/- 0.0045, ty -0.183 +/- 0.005, tz -20.158 +/- 0.010, wedge +0.331 +/- 0.006, centre (861.94, 401.71) +/- 0.03, Lsd 170.858 +/- 0.047 mm (bootstrap sigma, 80 boots).
- **tx and wedge free per run (the geometry is NOT shared):** reference r02 tx -0.092, wedge +0.446; deviations r03 tx +0.117 +/- 0.009, wedge -0.106 +/- 0.011; r04 tx +0.035 +/- 0.007, wedge -0.185 +/- 0.013. Per-run values tx -0.092 / +0.024 / -0.057 and wedge 0.446 / 0.340 / 0.261 (orientation fit: -0.103 / +0.023 / -0.036 and 0.413 / 0.310 / 0.291). Residual rms 0.00084 -> 0.00077.
- Control for this arm (`friedel_sim_dev.py`, 24 reps each, `v2/friedel_sim_dev.log`): identical geometry in the three runs -> deviations 0.0014 / -0.0034 / 0.0006 / -0.0010 deg, |dev| > 3 sigma in 0 to 4 % of reps; injected (+0.12 / -0.11 and +0.03 / -0.19) recovered with bias <= 0.005 deg.
PROVISIONAL reading: the run-to-run spread of tx and wedge in the orientation fits is not noise of the orientation fit; the U-free pair estimator sees the same spread. Cause NOT diagnosed (candidates, untested: the omega axis tilts differently at each phi setting, crystal off the rotation axis, a detector move between runs).
The tz and Lsd offsets (pair -20.158 / 170.858 against fit -20.08 / 171.11) are 5 to 7 sigma in the joint fit. UNTESTED idea: the orientation fit holds the COD cell, so a 0.15 % cell difference would move Lsd by 0.15 % (171.11 vs 170.86 is 0.146 %); refining the cell with the pair-based geometry would test it.

### 360 deg scans (`friedel_sim_360.py`, independent forward, A39 detector, 12 reps, `v2/friedel_sim_360.log`)
Every reflection has its mate in the same run (1538 pairs available at d >= 1.0 A). 100 pairs: empirical sd tx 0.0078, wedge 0.0041 deg (formal sigma 0.0046 / 0.0051: bootstrap needed for tx); all pairs: tx 0.0019, wedge 0.0013 deg, bias <= 0.0011. Usable without the cross-run step, provided the hkl labels (from an orientation fit) are right; NOT tried on Rochelle A37 (its lattice / labels were never established, A37 is INCONCLUSIVE).

### Not claimed
That the orientation-fit tx / wedge are right in an absolute sense (the pair relation shares the pixel-to-direction map and the omega scale with the fit); anything about the omega zero (gauge) or omega step; the cause of the per-run differences; the Lsd / cell idea.

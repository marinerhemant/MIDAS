# PREREGISTER A39-FLOOR (2026-10-23, fixed BEFORE any ablation below is run)

Data: L-alanine 100 K, Diamond I19-1 Pilatus, runs 02/03/04 (lalanine/ on copland, /gdata/dm/MPE/OrthrosJr/analysis/sharma_work/midas_sx_real). Baseline = `lalanine/out_p0` (plane 0, margin 6, mean background, header wavelength).
Baseline numbers (provenance: `real/a39_diag_budget.py lalanine/out_p0`, `real/a39_merge.py`): merged R_int 15.67 % (scaled, n >= 2), systematic floor f = 14.2 % relative F2 overall (counting prediction 2.7 %), f = 10.8 % at L < 1.5 and 23.5 % at L > 5, 11-13 % at d < 1.1 A and 18-22 % at d > 1.1 A, run 04 f = 17.4 % vs 11.8 / 11.3 % for runs 02 / 03; 10 rows rejected by the refinement, among them (1 0 2) (F2o 14.7, F2c 13044, n = 2).

## Hypothesis (one sentence)
The systematic floor of the A39 merge (14 % in F2 against 2.7 % from counting) is dominated by correctable effects in OUR processing (integration support truncating omega tails, strong-reflection count-rate losses, a drifting intensity scale, failed integrations) and not by the sample or the beam, so that fixing them brings R_int to <= 8 % and the model-free R_F2 against the reflections embedded in COD 1574527 to <= 10 %.

## Known-limits check
`~/.claude/known-limits.md`: nothing applies to single-crystal reduction. KL-010 (tilts constrain the axial ratio, never the cell scale) is the same kind of degeneracy as the wavelength/cell-scale one, not this question. Package limits that stand: no extinction, no absorption, no anomalous dispersion in the package unless a step adds them.

## Metrics (definitions fixed now)
- f: `a39_diag_budget.py` floor estimator (leave-one-out log deviation of equivalents, robust sd, minus the counting prediction, divided by sqrt 2), overall and by L, d, run. R_int: package definition, scaled (scale_groups), all n >= 2.
- TUNING objective, reference-free: f and R_int of the merge of runs 02 + 03. HOLD-OUT: the same for run 04 (its own R_int in reduction.json) and for the merge of all three after the final configuration is fixed.
- REFERENCE readout, untouched until the end and read ONCE per final configuration: R_F2 of the merged F2 (n >= 2) against the reflections embedded in the COD CIF (`_shelx_hkl_file`, merged to the same unique set with F2 meaning intensity as in the CIF), fit = one scale only, UNWEIGHTED, R = sum|Fo2 - k Fc2| / sum Fo2 (the current value 14.7 % is the physics lens's, to be reproduced by `real/a39_ref_hkl.py` before any ablation); tier C positions (median/max, aniso, 10 starts) are the second readout.
- No decision uses the reference readouts. A change is kept only by the tuning/hold-out rule below.

## Ablations, in this order, each applied on top of the previous kept ones
1. Failed integrations (the 10 rejected rows, (1 0 2) first): inspect the raw frames; if a package defect, fix at the root with a test. KEEP if the rows integrate sensibly afterwards and f does not worsen.
2. Support truncation: margin 8, 10, 14 (existing option), then profile fitting if the margin sweep cannot plateau. 
3. Strong-reflection count-rate: (a) exclude observations with window maximum > 1e4 counts, (b) a dead-time correction of the peak pixels from the Pilatus Tau (124 ns), whichever is justified.
4. Smooth scale vs omega per run from the equivalents (no I0 record exists). PLACEBO: the identical procedure with the omega labels permuted must give |delta f| < 1 point.
5. Absorption / extinction from equivalents, only if f is still > 8 % after 1-4.

## Per-step rule (fixed)
- GAIN: f on the tuning merge (02 + 03) drops by >= 2 percentage points (absolute, relative-F2 units) AND run 04 / the 3-run merge do not get worse by > 0.5 point. KEEP.
- NO GAIN: |delta f| < 1 point on the tuning merge. DROP, report.
- In between, or tuning and hold-out disagree: INCONCLUSIVE, drop, report.
- STOP after two consecutive NO GAIN steps (or at step 5).
- Reproducibility control: the baseline configuration rerun unchanged must reproduce f and R_int to the printed digits.

## Verdict words for the campaign
CONFIRMED: kept steps give R_int <= 8 % on all runs' merge AND model-free R_F2 <= 10 %. REFUTED: after all steps R_int > 12 % AND f > 10 % (the floor is not processing). Otherwise INCONCLUSIVE with the numbers. Effect sizes that matter: R_int 8 %, R_F2 10 %, f 5 %.
Every positive reading goes through /verify (4 lenses) before it is called established.

## AMENDMENT 1 (2026-10-23, written AFTER step-1 inspection and after the post hoc diagnostics below, BEFORE any confirmatory implementation)
What was seen, with provenance (all reference-free, `real/a39_diag_*.py` on copland `lalanine/out_p0*`):
- Step 1: the reduction is blob-driven (integration box centred on the measured blob; calculated crossing only for the corrections, `reduce.py`). (1 0 2) has two observations that are weak noise blobs accepted at hkl_residual 0.12 / 0.13 (I = -3 +- 284 and 189 +- 253 where about 72 000 counts are expected); the real reflection is not integrated. (0 0 8): obs F2 4015, 4183, 2825, -1.2; merged F2 = 834.6 +- 793 because the sigma-weighted mean is dominated by the failed observation (small sigmaF2): merge has no outlier rejection among equivalents and Poisson weights are biased low. (`a39_diag_rejected.py`)
- Step 2 (post hoc, margins already run): floor f overall 12.2 % (margin 1), 14.2 % (6), 15.8 % (10); L-dependence present at every margin (L<1.5: 9.7/10.8/12.3 %; L>5: 25.3/23.5/23.6 %). Larger margins do NOT lower the floor. Wavelength-scale variant: 14.2 %, no change. (`a39_diag_budget.py`, out_p0m1, out_p0, out_p0m10, out_p0_lam)
- L evaluated at the measured spot instead of the calculated crossing: median ratio 1.000, robust sd 0.001, floor by L unchanged. REFUTED as a cause. (`a39_diag_leval.py`)
- Step 3 (post hoc): excluding observations with blob peak counts > 1e4 / 3e4 / 1e5 (43 / 12 / 4 obs) changes f 14.2 % -> 14.0 / 14.1 / 14.2 %. NO GAIN. (`a39_diag_scalefit.py`)
- Step 4 (post hoc, in-sample): a piecewise-linear per-run log-scale vs frame, K = 8 knots fitted from the equivalents: f 14.2 -> 5.8 % overall (runs 02+03: 11.9 -> 5.5; run 04: 17.4 -> 6.2; L > 3: 20.9 -> 8.0), fitted log-scale range within a run 0.35 / 0.48 / 0.57; PLACEBO (frame labels permuted within each run) f = 13.3-13.7 % (5 seeds). So the scale drifts by +-20-30 % across a scan and the package has no model for it. This is a strong LEAD, not yet a result: it was fitted and read on the same observations.

New steps (registered now, before implementation):
- 1b Merge estimator: (i) robust rejection of observations inconsistent with their equivalents (leave-one-out, using sigma' that includes an error-model term), (ii) weights from an error model sigma'^2 = sigma^2 + (g * mean)^2 not driven by the observation's own value (removes the downward bias of 1/sigma^2 weights), g estimated from the data. Metric (reference-free): R_split = merge runs {02, 03} vs run 04 on the common unique reflections, sum |F_a - F_b| / sum (F_a + F_b)/2, and R_int; improvement = R_split drops >= 1 point and R_int does not rise.
- 1c Predicted-position integration: integrate every predicted crossing (refined geometry + U, existing `predicted_crossings`) with the same box/background, not only the found blobs; compare with blob-driven on the strong common reflections (must agree), on completeness and on weak-reflection bias; GAIN if the unmatched strong predicted reflections (like (1 0 2)) are recovered AND f and R_split do not worsen.
- 4 (confirmatory): implement the smooth scale in the package (`merge.scale_groups`, knots K = 8 FIXED now, knot variable = frame), with a CROSS-VALIDATED evaluation: fit the scale from a random half of the unique reflections, evaluate f and R_split on the other half (5 random splits); placebo as above; a no-drift control (simulated observations without drift must give |log scale| range < 0.03 and no change in R_int); a drift control (injected smooth drift of +-25 % recovered to within 0.03 in log scale). GAIN rule unchanged (>= 2 points on the tuning merge, no hold-out worsening) but evaluated on the held-out HALF of reflections.
Order: 4 and 1b are independent and run first (steps 1c in parallel as a fork); 2 stays NO GAIN / INCONCLUSIVE (margin 1 lowers f but raises R_int: not adopted); 3 NO GAIN. Everything else in the registration stands. Reference readouts are still untouched and read once at the end.

## AMENDMENT 2 (2026-10-23, before any ablation was read against the reference)
`real/a39_ref_hkl.py` on the BASELINE merge (`lalanine/out_p0/merged.csv`, 716 common n >= 2) gives R_F2 = 18.66 % with a least-squares scale (k = 0.593), but 14.18 % with the ratio-of-sums scale, 14.15 % with the L1-optimal scale; the physics refuter's 14.7 % is of that kind. The least-squares scale is pulled up by the strong reflections (Fo2/(k Fref2) median 0.81-0.88 in every shell). The registration said only "one scale only, unweighted": the registered definition is now the scale that MINIMISES R_F2 itself (L1-optimal). Baseline = 14.15 %. Per shell (baseline, LSQ scale): R 5.6-6.3 % for d < 0.9 A, 10-13 % for 0.9-2.5 A, 24 % beyond 2.5 A (17 reflections). Only the baseline has been read against the reference so far.

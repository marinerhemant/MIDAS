# /verify of claim 49b86bbe41de (midas_sx.workflow composes the validated chain), 2026-10-21: PROVISIONAL

Lenses: physics UNCERTAIN, statistics UNCERTAIN, artifact UNCERTAIN, reproduction SURVIVES -> PROVISIONAL (mechanical rule). Refuter workdirs: /Volumes/T5 EVO/verify_workdirs/Desktop/analysis/midas_sx/refute_49b8_{physics,statistics,reproduction,artifact}/

## What holds
- reduce_twin90_scan + refine_twin90_structure reproduce run_pm2.py + refit_pm.py bit for bit with d_min passed EXPLICITLY: seed 3 (478 rows, my check), seed 8 (459 rows) and seed 21 (462 rows) (reproduction lens, strict NaN-safe comparison of all 12 columns, F2 sigF2 sep_px max relative difference 0.0). Twin law, RZ90, TWIN_90_C, step order, crowd_px, neighbours, centre_rc (row, col), omega_start (start edge) all match or are consistent.
- The refit result for seed 3 matches the saved Tr arm (GooF 1.0875, m 0.3051, sigma 0.0324, 1 rejected) from a different start (truth, not the 0.10 A displaced start); seeds 8 and 21 give m 0.4175 +- 0.18 and 0.004 +- 0.41: the m +- 0.032 precision is seed 3's, not general.

## What does NOT hold as worded
1. DEFAULT d_min. The workflow derives it from the farthest detector CORNER (2theta 47.3 deg, d_min 0.515 A, HKL pool 4702, 1482-1674 twin targets) while the validated chain used the simulator's inscribed edge (2theta 37.2 deg, d_min 0.648 A, pool 2354, ~1190-1309 targets); 362-379 extra targets lie beyond the inscribed radius, where the simulated frames have no signal. The check passed d_min explicitly, so the default was never exercised. -> default changed to the validated definition (edge midpoints inset 3 px), see below.
2. reduce_twin90_scan is in NO unit test (77/119 statements of workflow.py covered; all of reduce_twin90_scan and _tth_max_deg missed). The equivalence rests on three seeds of one variant (P11).
3. Seeds 3, 8 and 21: seed 3 has 0 c2 rows, so the U2 / c2 integration branch is not exercised there (seeds 8 and 21 were not checked for c2 rows).
4. Silent behaviour: pair-model non-convergence falls back to the centroid orientation with no warning (only pair_orientation_used = False); allow_unconverged=True is silent; the workflow RAISES on an unconverged centroid orientation where the chain carries on, and lets a build_profile RuntimeError propagate where the chain writes a CSV note. DacModel() defaults to mu_per_cm = 0 (no anvil absorption, silently) and PolarizationModel to 0.99 horizontal.
5. A read-only numpy array (pandas 3 to_numpy) makes torch warn once per process in refine.py, so the unit test 'a clean fit must not warn' depends on test order.
6. refine_twin90_structure on a minority-domain U0 (true m = 0.7) returns twin.m on the 0.5 bound, converged False, with a generic message.
7. reduce_scan has no numerical equivalence check with run_c2.py (only the small Cu recovery test: orientation 0.018-0.021 deg and median F2 error 0.3-0.6 % on two seeds, 2.4x and 9x headroom to the test thresholds).
8. One refuter used pkill -f against the rules (it matched only its own job).

## Amendments made after the verification (tested; not re-verified)
see PREREGISTER / DESIGN: default d_min = validated definition; warnings for the fallback and for allow_unconverged; a hint at twin.m = 0.5; a unit test for the default d_min; np.array copy of hkl in refine.py.

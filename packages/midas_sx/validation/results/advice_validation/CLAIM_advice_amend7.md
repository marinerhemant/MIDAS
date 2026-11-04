# CLAIM (repaired D2/D3 of the advice module), 2026-10-29, midas-17. PROVISIONAL until /verify returns.

Engineering claim about `midas_sx/advice/quality.py` after PREREG_advice.md Amendment 7 (written after the first /verify, claim a0c60827daf1, found D2/D3 non-discriminating).

## Claim
1. `dynamic_range` returns a `limiting` finding whose title contains "saturated" on the simulated high-count frames clipped at a declared cutoff (D2: 217 of 1452 observations clipped, fraction 0.149), and returns NO such finding on the same scene unclipped with a cutoff declared above its maximum (D2c: 0 blobs clipped) nor with no cutoff declared (D2n: 1 blob at the maximum, fewer than 3).
2. `omega_sampling` returns a note on the 1.0 deg step scene (single-frame fraction 0.50) and NO finding on the 0.25 deg scene (clean), at the registered bar `single_frame_fraction` 0.25.
3. On the real L-alanine runs 02-04 (cutoff 791927) neither check raises a new finding: no saturated finding (0 observations clipped), no omega_sampling finding, and the 14 earlier expectations still pass.
4. Unit tests: tests/test_advice_quality.py (20) and test_advice_coverage.py (19) pass; the full midas_sx suite passes (253, 28:41).

## Not claimed
That a real detector's saturation is visible to this test (see the question of what `spots.peak_counts` is); that the 0.25 bar is right for other spot widths, exposures or mosaicities; behaviour on steps between 0.25 and 1.0 deg; the oversampled rule; other crystals.

## Disclosed
The 0.25 bar was chosen after the first verify had shown 0.508 (1.0 deg step) against 0.013 (clean). The repair and the controls were written after seeing the failure.

## Evidence
- validation/real/PREREG_advice.md (Amendment 7 and its READING)
- validation/results/advice_validation/amend7/amend7.json, amend7/a39_all_amend7.log
- script validation/real/advice_amend7.py; code midas_sx/advice/quality.py (check_dynamic_range, check_omega_sampling), midas_sx/advice/__init__.py (THRESHOLDS single_frame_fraction, clip_min_counts)
- tests/test_advice_quality.py
- the first verify's evidence: validation/results/advice_validation/CLAIM_advice.md
- full suite log: /private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/full_suite_final.log

## OUTCOME (2026-10-29, midas-17): verdict PROVISIONAL (claim a56630731be7). Lenses: statistics SURVIVES, reproduction SURVIVES, artifact SURVIVES, physics UNCERTAIN.
Every number reproduced (per-number table in the reproduction lens, byte-identical re-run of advice_amend7.py). The physics lens left it UNCERTAIN on scope: the clipping test (peak >= 0.99 x cutoff) is blind when the background exceeds about 1 % of the cutoff (a uniform background of 2500 counts on a 226888 cutoff: 437 raw pixels at the cutoff, 0 spots flagged), and the 0.25 single-frame bar is specific to the simulator's spot width (real L-alanine spots are about 3.5 x broader; the check would not fire there until the step is well above 1 deg). Limits found by the lenses and NOT fixed (they are in the docs, not in the code):
- Declared cutoff wrong (2 x the true one, or 1e6): no saturated finding; the tie rule runs only when no cutoff is declared.
- Peaks 2 % below the cutoff are missed, 0.5 % below fire. clip_min_counts (1e4) is an absolute guard: low-count scenes with no cutoff declared are not flagged.
- The 2 % bar counts observations, not intensity: 24 clipped blobs (1.6 %) held 25.5 % of the integrated intensity and gave no finding (cutoff 20 x the median peak).
- Real L-alanine r04: one blob at 1.0000 of the cutoff (a real clipped Pilatus pixel: frame pgw240050_04_00416, row 760, col 257) was rejected as crowded, so the observation-level check reports nothing; only blobs_clipped_per_run shows it.
- A gain map or background gradient reduces what the tie rule sees; a cutoff declared below the data makes "saturated" fire with the inconsistent-cutoff note beside it.
- Specificity: 0 of 15 clean scenes (12 fresh + 3), single structure, effective n below 15; Wilson 95 % upper bound 0.20 (0.24 for 12). D3 curve: single-frame fraction 0.008, 0.07, 0.18, 0.30, 0.38, 0.43, 0.50 at steps 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 deg: the bar is crossed near 0.58 deg, a 0.5 deg step is not flagged.
Wording fixed after the verify (title only, no logic): the saturated title no longer says "identical" or "sit at" for ties and for peaks above a declared cutoff.

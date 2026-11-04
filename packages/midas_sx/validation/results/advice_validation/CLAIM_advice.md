# CLAIM (advice module), written 2026-10-28 by midas-17 before /verify

Status: PROVISIONAL until the four lenses return. Engineering claim about code (midas_sx.advice), not about the world.

## Claim, one falsifiable statement

`midas_sx.advice.assess` (checks in `advice/coverage.py`, `advice/quality.py`), run on frames from the independent simulator `validation/simulate.py`
(S2 bilayer, orientation seeds 3-5, `n_int_max = 2e7`, `d_min = 0.66` target, cell aperture declared via `Acquisition.dac`):

1. raises no blocking or limiting finding on clean frames (scenarios C and C6hi);
2. when exactly ONE defect is injected into the rendered signal, fires the named check at the registered severity:
   - D1 smooth 25 % per-run drift: `intensity_drift` limiting, placebo gain 0.011 vs real gain 0.937;
   - D2 saturation at 4x the median peak: `dynamic_range` present, 40.2 % of reflections above 25 % of the cutoff;
   - D3 1.0 deg step instead of 0.25: `omega_sampling` present, median width below 3 frames;
   - D4 40 deg scan: `scan_range` limiting when no allowance is declared, note when the allowance equals the scan;
   - D5 one run: `completeness` and `redundancy` limiting;
   - D6 rotation axis tilted 0.4 deg: refined wedge 0.402 / 0.408 / 0.401 deg (clean: 0.014 / 0.007 / 0.002);
   - D7 200 extra spots on 2theta = 12.0 deg: contamination ring found at 11.88 deg;
   - D8b independent per-frame lognormal scale sigma 0.25: `systematic_floor` limiting (0.145) and `intensity_drift` NOT limiting;
3. 25 of the 26 registered conditions pass on this table; the one that fails (D4_set: 40 deg scan inside a 76 deg allowance expected "note", got "limiting") is a registration error, the check is right;
4. on the real L-alanine runs 02-04 it reproduces 14 of 14 hand-measured numbers (smooth-scale log ranges 0.365 / 0.467 / 0.578, refined wedge 0.4129 / 0.3104 / 0.2908 deg, in-run Friedel pairs 38 / 57 / 47, cross-run mates 1802 of 3120, 12 / 9 / 11 blobs above 1e5 counts per run, run-scale-only floor 0.142, no blocking finding).

## What it does NOT claim

The advice wording, the quoted gains and the actions were not tested. Only which finding fires and at which severity. One simulated structure, three orientations, single clean defects, one real crystal (L-alanine). Not tested on A37/A38.

## Disclosed forking paths (all made after seeing results)

As registered (first run) the verdict was REFUTED (C6, D4_set, D8 failed; and the clause "no other limiting finding" in D1, D2, D3, D6, D7, D8). Six amendments followed:
two module repairs (cell aperture; resolution survivorship), one new option (`d_min` target), and several scenario changes (D8 sigma 0.10 -> 0.25, clean base 2e5 -> 2e7 counts, d_min 0.66 for all scenes because the simulator truncates at 0.648 A). The thresholds and scenarios in claim point 2 are the AMENDED ones. Full text: `validation/real/PREREG_advice.md`.

## Evidence

- `validation/real/PREREG_advice.md` (registration, 6 amendments, READING)
- `validation/results/advice_validation/table_final_highcount.txt`, `final_highcount/*.json|txt`, and the earlier tables in `first_run/`, `run2_aperture/`, `run3_partial_dmin/`, `run4_lowcount_dmin/`
- scripts `validation/real/advice_validate.py`, `advice_validate_eval.py`, `a39_advice_all.py`
- code `midas_sx/advice/{__init__,coverage,quality}.py`; tests `tests/test_advice_coverage.py` (19), `tests/test_advice_quality.py` (20 after Amendment 7; 18 and 37 pass at the time of the claim), 39 pass after the repair (2026-10-28, 24 s)
- simulator `validation/simulate.py`

## OUTCOME (2026-10-28, midas-17): verdict PROVISIONAL, 4 of 4 lenses UNCERTAIN (physics, statistics, reproduction, artifact)
Every number in the claim reproduced (reproduction lens), but D2 and D3 as worded were not detections: clean frames gave the same output (D2: fraction 0.4019 clean vs 0.4015 clipped; D3: note on the clean 0.25 deg scan). Points 2 for D2 and D3 are WITHDRAWN as worded. The checks were repaired (PREREG_advice.md Amendment 7: clipping test for dynamic_range, single-frame fraction for omega_sampling) and pass their registered discriminating controls on three cached seeds; that repair is NOT re-verified. D1, D4, D5, D6, D7, D8b are differential against the clean scene (lenses agree). Further caveats: PREREG_advice.md "READING of Amendment 7".

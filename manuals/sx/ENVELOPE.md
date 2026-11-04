# sx envelope: what has been verified, what has not, and what the data cannot say

Status words are copied from the claim log (`~/.claude/bin/skill-log`) and from the registered readings. They are never promoted here. "ESTABLISHED" means it survived a four-lens /verify in a fresh context, with the scope written in the row. Everything real-data is ONE crystal (L-alanine, Diamond I19-1, three runs) unless the row says otherwise.

## ESTABLISHED (four-lens /verify)

| Claim id | What | Scope and limits | Source |
|---|---|---|---|
| f55319a6e282 | Smooth per-run frame scale plus robust merge: R_int 15.67 -> 6.77 % (6.88 % with the scale fitted on half the reflections), R_split 21.4 -> 10.1 %, model-free R_F2 against the independent reference reflections of COD 1574527 14.15 -> 6.33 % | One crystal, three runs. Both steps contribute (R_F2 9.31 % smooth scale + default merge, 11.33 % run scale + robust, 6.33 % both). Error-model g = 0.073 on the smooth-scaled data. The d < 0.8 A shell does not improve. The mechanism (cause of the +-20 % scale) is NOT established. | `validation/results/a39_floor/CLAIM_floor.md` |
| a1a8381d5796 | Tier C refinement of positions on the baseline merge | LOCAL refinement only (basin about 0.43 A per atom). Needs studentized outlier rejection (cap 10). Hand and origin come from the start model and fixed COD hydrogens. 10 starts = one result. Formal esds are understated. ADPs not validated. | `validation/results/a39/CLAIM_tierC.md` |
| 2f5c452c9687 | Tier C on the smooth + robust merge: GooF 1.64, R1 3.02 %, non-H positions median 0.0014 A and max 0.0031 A from COD 1574527 | Same limits as above. The GooF comparison with the baseline mixes the error-model change with the merge change: compare R1 and positions. The 10 rejected rows are strong, well-measured reflections the model misfits (extinction is a guess). The isotropic-U arm missed its bar (GooF 3.14 against 2.5). | `validation/results/a39_floor/CLAIM_tierC_cfgA.md` |
| 210cb198b6b6 | Polarization correction, differentiable | simulated and unit-level | `PREREGISTER.md` |
| 9c505c031bbd | `orientation_grid`, `index_by_grid`, device policy | wording caveats in the log | `PREREGISTER.md` |
| 38c79b5ee7fe | Per-parameter bound test plus Poisson-quantile clip | limits in the log | `PREREGISTER.md` |

## PROVISIONAL (not verified, or verified with a lens left UNCERTAIN)

| What | State |
|---|---|
| The advice module (`midas_sx.advice`) | Claim a0c60827daf1: 4 of 4 lenses UNCERTAIN, verdict PROVISIONAL. Every number reproduced. Checks that DISCRIMINATE a defect from clean frames in simulation: intensity_drift (D1), scan_range (D4), completeness and redundancy (D5), geometry_consistency wedge (D6, always a note), contamination ring (D7), systematic_floor (D8b). D2 (dynamic_range) and D3 (omega_sampling) were NOT detections as first written; they were repaired afterwards (clipping test, single-frame fraction) and re-verified: claim a56630731be7, 3 lenses SURVIVES, physics UNCERTAIN, verdict PROVISIONAL. Scope limits: the clipping test is blind above about 1 % background, the 0.25 single-frame bar is specific to the simulator's spot width. See `ADVICE.md` and `validation/real/PREREG_advice.md` (7 amendments, all after seeing results). |
| `refine_wavelength` (cell-scale/wavelength as an 11th parameter) | PROVISIONAL. With the cell fixed, +eps in wavelength equals -eps in isotropic cell scale: it is not an energy calibration. |
| Readers, conventions, setting match, wedge refinement | Unit-tested and used on L-alanine only. |
| Thiourea tier A and B2 readings (claim 0a2b9eb0d997) | PROVISIONAL (2 lenses SURVIVES, 2 UNCERTAIN on wording). Corrected wording: blind lattice 98.9 %, null max 10.7 %, R_F2 7.5 % only on the 46 reflections with I/sigma > 10 with an unphysical B, 23.6 % on all merged. |
| Ice-ring "consistency" (d = 3.903 A etc.) | A d-spacing coincidence check, not an identification. |
| Ring-coverage denominator for completeness (a reflection is expected only if at least 50 % of its Debye ring is on the detector) | A design choice made by midas-17; not tested against an alternative. |

## INCONCLUSIVE, REFUTED, NO GAIN (kept so nobody repeats them)

- INCONCLUSIVE: A36 (the default background stays "mean"); A37 Rochelle salt; A38 thiourea overall; A39 B1/B2 on the baseline merge; the low-angle d-frame test; Friedel-pair geometry (`friedel_geometry`: agrees with the orientation fit within 0.035 deg, the registered rule met in 1 of 3 runs); A39-FLOOR step 1c.
- NO GAIN: `integrate_predicted` (integrating every predicted crossing): no gain on the systematic floor, +6 % unique reflections, opt-in.
- REFUTED as worded: the pooled bound test (97f96459b1e5, fixed); the registered A39 tier A as written; the CeO2 first-ring control (63140c038bf4, an expected REFUTED, 4 of 4 lenses); the advice validation as registered (C6, D4_set and D8 failed as written; see `ADVICE.md`).

## What the data cannot determine, whatever the package does

- c* (the reciprocal axis near the beam) in a DAC or any scan where it sits near the incident beam; reflections near the beam are not measured.
- Intensities below about 1e4 counts of integrated signal are noise-limited; the outer shell of a weak crystal is simply absent (see `completeness` and `resolution_limit`).
- Hand and origin are taken from the start model; they are not determined by the refinement.
- Anisotropic displacement parameters are NOT validated. Absolute structure is not claimed.
- The cause of a smooth per-frame scale (centring, illumination, beam drift) cannot be told from the frames. The scale is inferred from equivalents, not measured. Record I0 per frame.
- An energy: wavelength and an isotropic cell scale are degenerate. Measure a calibrant.
- A twin law: this package takes a declared one. It does not discover it.

## Scope of the real-data evidence

One crystal verified (L-alanine, 100 K, Pilatus 2M, 170 deg per run, three runs). The Rochelle and thiourea sets did not meet their registered bars. Simulated validation covers two structures (a Cu-like cell and a layered oxide bilayer cell) and three orientations.

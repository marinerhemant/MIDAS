# ID03 forward test: registered run, results and how to re-run

The registered leave-one-plane-out test behind `forward_id03_maps_to_frames.ipynb`.

The `PREREGISTER*.md` files named below (the registrations, their amendments and the results read against them) are
kept with the author's analysis notes and are not in this repository: the repository keeps registrations out. The
results, logs and run scripts listed here carry the numbers, and the verdicts of record are quoted in this file.

| file | what |
|---|---|
| `PREREGISTER.md` | the registration, its two amendments (sections 10-11, written after a smoke run on a design ROI and before the registered run) and the result read against it (section 12) |
| `run_forward_loo.py` | the run: `python run_forward_loo.py umich_school_raw_8planes.h5 OUTDIR --n-perm 200 --nsub 40000` |
| `results.json` | every number the notebook quotes in markdown |
| `run.log` | the log of the registered run |

The run needs the whole 8-plane cube in memory (about 17 GB of float32 for the 786 k grain pixels) plus the same again in
transient copies, so it was run on a compute host with ~2 TB of RAM. It took about 40 minutes. The three `maps_fold*.npz`
residual maps (36 MB each) are not kept here.

**v2 verdict: INCONCLUSIVE** (see `PREREGISTER.md` section 12).

## v3: the flank-plane fix

| file | what |
|---|---|
| `PREREGISTER_v3.md` | the second registration (flat-top obpitch profile), the result (section 13) and the registration error found in it |
| `run_forward_loo_v3.py` | the run, with the v2 Gaussian pipeline rescored on the same pixels |
| `results_v3.json`, `run_v3.log` | numbers and log |

**v3 verdict of record: REFUTED by the rule as written, because of a mis-indexed clause (my error; section 13).** Read post hoc with the right
folds, the flank failure drops from R² -6.6 and -11.2 to -0.31 and -0.13 and the primary planes rise from 0.547 to 0.695, but the flank planes stay slightly
below zero. The code is `midas_dfxm/forward_profile.py` (6 tests).

## Physical forward with a flat-top acceptance

`ResolutionFunction(par_flat=(width, softness))` is a new optional erf-box longitudinal acceptance (default `None`, behaviour unchanged);
`forward_thin_slab.resolution_for_plane/render_weights/match_marginal_widths` take `flat_top_ob_deg=(width, softness)`.
`equiv_check.py` and `equiv_results.json` hold the registered equivalence check against v3 on two 112 x 112 px ROIs
(`PREREGISTER_v3.md` sections 14 and 14b). **Verdict: NOT CONFIRMED.** The pooled primary score agrees within 0.02 and the Gaussian acceptance is 4-5 R^2 units worse on the
flank folds, but single folds differ by up to 0.36 and per-pixel correlation is 0.81-0.98, so v3's full-grain scores do not transfer to the physical forward.

## v4: the physical flat-top forward on the whole grain

`PREREGISTER_v4.md` (registration and result), `run_physical_full.py`, `results_v4.json`, `run_v4.log`. **Registered verdict: INCONCLUSIVE**: 6 of 7 criteria
met (no loss vs v2, flank folds -0.66 and -0.34 against v2's -6.6 and -11.2, far folds at zero, block null, Gaussian-acceptance control, matched synthetic); the
pooled primary score is 0.624 against v3's 0.695, a gap of 0.071 for a registered tolerance of 0.05. The flat-top acceptance matters on the flank folds and makes almost no
difference on the primary planes. Provisional; not through `/verify`. New helper code: `forward_thin_slab.physical_references`, `physical_score` (covered in `tests/test_forward_thin_slab.py`).

## v5: the three candidates, then the confirmatory run

`PREREGISTER_v5.md` (part A diagnostic with its decision rule fixed first, part B registration, both results), `diag_v5.py`, `diag_v5.json`, `run_physical_full.py --no-couple --per-pixel-widths --version v5`, `results_v5.json`, `run_v5.log`.
**Registered verdict: CONFIRMED.** Physical flat-top forward with the obpitch-rock coupling off and per-voxel widths: primary pooled R^2 0.699 against v3's 0.695 (per-pixel correlation 0.999), flank folds -0.34 and -0.20, all seven criteria met.
It is v3 re-expressed in the package's geometry, not an independent confirmation of v3; the coupled geometry scored 0.07 lower and why is not decided. `/verify` PROVISIONAL (see the paragraph below).

**`/verify` of v5: PROVISIONAL** (reproduction and physics SURVIVE, statistics and artifact UNCERTAIN; details at the end of `PREREGISTER_v5.md`). Quote the primary R^2 as about 0.70 +-0.05; physical vs empirical flat-top differ by +0.003 (about 1 % of the SSE);
fold 6 below zero is not statistically firm; the coupling strength is not pinned by the data.

## v6: the coupling strength and the amplitude, measured, then one registered run

`PREREGISTER_v6.md` (part A decision rules fixed first, part 1 and part 2 results, a disclosed design exploration, part B registration and result), `coupling_measure.py` + `coupling_v6.json`, `amp_diag.py` + `amp_diag_v6.json`, `calib_explore.py`, `run_physical_v6.py` + `results_v6.json`, logs.
- **Coupling, measured model-free on the whole grain:** alpha_hat = 0.153, 95 % block-bootstrap [0.088, 0.227] (synthetic alpha = 0, 0.5, 1 controls recover 0 : 0.50 : 1; shuffle control 0). It varies about 4x across regions (0.10 to 0.47), so it is not a fixed instrumental constant; a depth gradient along the beam would look the same and cannot be separated.
- **Amplitude oracle decomposition** (excluded ROIs): per-plane scalar 0.036, per-pixel amplitude 0.087, per-pixel centre 0.083, profile 0.031; amplitude and centre are the same held-out error seen two ways.
- **v6 (alpha 0.153 + local-pooled amplitude), whole grain: INCONCLUSIVE, no improvement:** primary 0.6994 against v5's 0.699; the nested search never smooths (L = 1 in every fold). Provisional.

**`/verify` of the coupling measurement: PROVISIONAL** (reproduction SURVIVES; physics, artifact and statistics UNCERTAIN). The slope 0.0725 deg/deg reproduces, with an honest interval of about [0.015, 0.13]; but an equal-size slope on the chi axis (where the geometry predicts none), a complete obpitch = plane = time confound, edge-dominated and spatially structured values, mean it is an effective plane-dependent shift, not a demonstrated coupling. The amplitude decomposition claim was not sampled for verification and stays PROVISIONAL, unverified.

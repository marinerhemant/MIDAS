# Implementation Plan — grain-based tx/Wedge geometry refine (lightweight)

## Goal
Close the `tx=0` gap: after the FF pipeline reconstructs grains with `tx=0`
(powder is blind to `tx`), refine `tx` (and `Wedge`) from the single-crystal
grain spots, then re-reconstruct. Standalone in `midas_joint_ff_calibrate`,
lightweight path first (refine only `tx`/`Wedge` + grain orient/pos; everything
else frozen). Full-joint (powder+grains) layered on the same entry point after.

## Why this works
A symmetric powder ring is invariant under rotation about the beam, so `tx` is
unconstrained by calibrant data. Single-crystal grain spots break that
degeneracy — their (Y,Z,ω) positions move with `tx`/`Wedge`. We already have a
differentiable HEDM spot residual (`midas_fit_grain.hedm_spot_residual`) and an
LM minimiser (`midas_peakfit.lm_minimise`); we only need a thin entry point with
the right refine-mask.

## New module: `midas_joint_ff_calibrate/grain_refine.py`

### `refine_geometry_from_grains(...) -> GrainGeomRefineResult`
Inputs:
- `paramstest` — v1 paramstest the pipeline ran with (`tx≈0`, full v2 geometry).
- `layer_dir` — pipeline output dir (`Grains.csv` + `SpotMatrix.csv`/`ExtraInfo.bin`).
- `refine_params=("tx", "Wedge")` — geometry blocks to thaw (configurable).
- `with_powder=False` — lightweight HEDM-only by default; `True` adds the
  powder pseudo-strain residual (the full-joint path, reusing `joint_residual`).
- `max_grains`, `max_iter`, weights, device/dtype.
- `out_paramstest` — write a corrected paramstest for the pipeline re-run.

Steps (reusing existing primitives — no re-port):
1. Load geometry via `midas_calibrate.params.CalibrationParams.from_file`.
2. Load grains via `midas_process_grains.io.csv` (Grains.csv → OM/pos/lattice)
   → Euler/pos/lattice init tensors.
3. Build `HEDMForwardModel` (`midas_diffract`) + per-grain `ObservedSpots` /
   `MatchResult` (`midas_fit_grain`) from SpotMatrix. The one piece of glue not
   yet in a library is the SpotMatrix→observations mapping; factor it from
   `runners/run_real_phase3_joint.py` into a shared helper
   (`grain_observations.py`) and point the runner at it too (avoid dual-tree).
4. Build spec: start from the geometry spec; **freeze every geometry block
   except `refine_params`**; `build_joint_spec(..., refine_grain_orientation=True,
   refine_grain_position=True, refine_grain_strain=False)`. Tight bounds on the
   thawed geometry (`tx`,`Wedge` ± a few °).
5. Residual: `hedm_spot_residual` only (lightweight) or `joint_residual`
   (`with_powder=True`).
6. `mp.lm_minimise` → refined `tx`/`Wedge`.
7. Write `out_paramstest` = input params with `tx`/`Wedge` overwritten; return a
   result object (refined values, cost before/after, n_grains, n_spots, rc).

### CLI
`midas-joint-ff-calibrate-grain-tx` (entry point in pyproject) →
`refine_geometry_from_grains` with `--paramstest --layer-dir --refine tx,Wedge
--out --max-grains --max-iter [--with-powder]`.

## Refactor (anti-dual-tree)
Lift the SpotMatrix→`ObservedSpots`/`MatchResult` builder and the Grains.csv→
init-tensors reader out of `run_real_phase3_joint.py` into package modules; have
the runner import them. Net: one definition, used by both. (Runner behaviour
unchanged — pure extraction, covered by re-running its smoke path.)

## Tests
- **Synthetic recovery (headline):** take a known geometry with `tx=tx_true`,
  forward-simulate grain spots for a few grains, zero `tx`, run
  `refine_geometry_from_grains(refine=("tx",))` → recover `tx_true` to <1e-3°
  and cost ↓. Proves the degeneracy is broken by grain spots. (Mirrors the
  synthetic-proof approach you asked for on FIX-3.)
- **Refine-mask unit test:** only `tx`/`Wedge` + grain blocks are in
  `spec.refined_names()`; Lsd/BC/ty/tz/distortion frozen.
- **Round-trip I/O:** `out_paramstest` re-parses with corrected `tx`, all else
  byte-equal to input.
- **Extraction parity:** the factored loaders reproduce the runner's
  observations/matches on a small fixture.

## Out of scope
- The pipeline re-run itself (user re-runs with `out_paramstest`).
- Distortion/Lsd/BC refinement from grains (full-joint covers it via
  `with_powder=True`; not the lightweight default).
- `ty`/`tz` are powder-constrained already; not thawed by default.

## Sequence
1. Factor loaders → `grain_observations.py` (+ point runner at them).
2. `grain_refine.py` lightweight path + refine-mask + paramstest writer.
3. Synthetic recovery test + unit tests (green).
4. CLI entry point.
5. `with_powder=True` full-joint path on the same function.

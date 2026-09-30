# midas-fit-grain

PyTorch single- and multi-grain refiner. Drop-in replacement for the C executables
`FitPosOrStrainsOMP` / `FitPosOrStrainsGPU` in MIDAS FF-HEDM.

Status: 0.2.x — sparse-output pre-allocation fix (trailing-skipped seeds
no longer truncate `OrientPosFit` / `FitBest` / `ProcessKey`) and
vectorized pixel-residual fast path (~N_g× fewer kernel launches per LM
iteration vs the per-grain Python loop). Park22 + datasetC CP-Ti real-data
validated. The C path remains the ff_MIDAS default;
this package is opt-in via `--refine-backend python`.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_fit_grain/notebooks).

## What it does

For each grain in `SpotsToIndex.csv`:

- Reads matched spots from `ExtraInfo.bin` and the seed orientation from
  `BestPos_*.csv` (the indexer's per-spot output).
- Refines 12 parameters: position (3) + Bunge Euler (3) + lattice (6).
- Writes byte-identical `OrientPosFit.bin` / `FitBest.bin` / `Key.bin`
  consumed by the existing `ff_MIDAS.py` merge stage.

## Solvers

`--solver {lbfgs,adam,lm,nelder_mead}` — default `lbfgs`.

## Loss functions

`--loss {pixel,angular,internal_angle}` — default `pixel`.

| Loss | Residual | Equivalent C function |
|------|----------|------------------------|
| `pixel` | `(y, z)` pixel positions on detector | `FitErrorsPosT` (`FitPosOrStrainsOMP`) |
| `angular` | `(2θ, η, ω)` in radians | `optimize_single_grain` (midas-diffract) |
| `internal_angle` | angle between `ĝ_pred` and `ĝ_obs` (rad) | `CalcInternalAngle` (`FitOrientationOMP`) |

## Fit modes

`--mode {iterative,all_at_once}` — default `iterative`.

- `iterative`: position → re-match → orientation → re-match → strain → re-match
  → joint polish (matches `FitPosOrStrainsOMP` default behavior).
- `all_at_once`: 12 params jointly, association computed once at entry, no
  mid-fit re-match.

## Spot weights (c-omp refiner)

Two opt-in weights on each matched spot's contribution to the stage objectives, both off by default and bit-identical
to the unweighted refiner when off (verified on LSHR, bt_20id_sep26b and datasetH):

* `RelFitRMSEWeightR0 r0` — scalar weight `1/(1 + rel/r0)`, `rel` = the merged spot's peak-fit `FitRMSE/IMax`, read
  from `RelFitRMSE.bin` (one float64 per `ExtraInfo.bin` row; the pipeline's refinement stage writes it when the key
  is set). `r0 = 0.69` improves grain positions against EBSD (LSHR layer 1, ESTABLISHED 2026-09-28); it does not
  reduce strain noise.
* `SpotWeightsDirectional 1` — experimental radial / along-ring / omega weights from `SpotWeights.bin`
  (3 float64 per row, `midas_transforms.io.csv.write_spot_weights_bin`). Combined with `RelFitRMSEWeightR0` the
  orientation and position stages stay scalar and only the strain stage is directional.

A set option with a missing, short or invalid file stops the refiner with a message rather than running unweighted.
Reported errors (`FitBest*`, `DiffPos`, `DiffOme`) stay unweighted. Measurements and preregistrations: ff-hedm
`LAB_NOTEBOOK.md` §14.

## Backends

`MIDAS_FIT_GRAIN_DEVICE` and `MIDAS_FIT_GRAIN_DTYPE` follow the same precedence
contract as `midas-index` (`cuda > mps > cpu` auto-detect; `f64` on CPU,
`f32` on accelerators). Per-grain refinement is batched into a single forward
call across the block, so scaling depends on `B × S` (grains × spots/grain),
not per-grain Python overhead.

## CLI

```
midas-fit-grain paramstest.txt <blockNr> <numBlocks> <numLines> <numProcs> \
                [--solver lbfgs] [--loss pixel] [--mode iterative]
```

Argv shape mirrors the C binary so the ff_MIDAS subprocess line is one-for-one.

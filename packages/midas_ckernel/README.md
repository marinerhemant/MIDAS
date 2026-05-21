# midas-ckernel

Shared C kernels for the MIDAS **c-omp** indexer (`midas_index`) and refiner
(`midas_fit_grain`). One copy of the diffraction forward model and the
linear-algebra / optimizer primitives, so the two pipelines can no longer drift.

## Why this package exists

Historically the orientation indexer and the position/strain refiner each
carried their **own** copy of the diffraction forward model:

| | indexer (`IndexerUnified.c`) | refiner (`CalcDiffractionSpots.c`) |
|---|---|---|
| function | `CalcDiffrSpots` | `CalcDiffractionSpots` / `CalcDiffrSpots_Furnace` |
| precision | `RealType` (=`double`) | `double` |
| output cols | 16 (`sinOme`/`cosOme` at 14/15) | 9 (`GCr` at 3-5, `nrhkls`) |
| extra gating | `RingsToReject` fraction tally | `BigDetector` active-area mask |
| `v` source | `sin(θ)·\|G_hkl\|` (precomputed) | `sin(θ)·\|OM·G_hkl\|` (runtime) |

The two had diverged subtly — exactly the class of bug that makes
indexer-vs-refiner comparisons untrustworthy. This package collapses them into
a single `midas_ck_calc_diffraction_spots()`.

## Design rulings

* **R1 — fp64 everywhere.** `RealType` is fixed to `double`.
* **R2 — `v = sin(θ·deg2rad)·|G_hkl|`** from the *un-rotated* reciprocal
  magnitude, exactly as the indexer precomputes it. This is what preserves the
  indexer's locked **500/500 bit-level parity**. The newly-ported refiner
  adopts the same form (ULP-different from its old `|OM·G|`, mathematically
  identical) and becomes its own reference.
* **R3 — frozen output layout.** Columns 0-9, 14, 15 keep the indexer's exact
  offsets (its `CompareSpots` reads them). Refiner-only `GCr` lives in appended
  columns 16-18, so the indexer is bit-unaffected.
* **R4 — both gates, NULL-gated.** `RingsToReject` (indexer) and `BigDetector`
  (refiner) are both supported; passing `NULL`/`0` disables each, so one body
  serves both callers with no behavioral change.

See `c_src/forward.h` for the full column map.

## Verified parity

`tests/parity_test.c` (run via `tests/test_forward_parity.py`) drives 2000
random orientations through the unified forward and both legacy forwards:

```
[indexer] rows compared: 28044   BIT-mismatches: 0   PASS (bit-identical)
[refiner] rows compared: 28044   within 1e-7: 28044   max|Δ|: 3.463e-09  (ULP-tolerant, R2)
```

## Consuming the shared sources

Downstream scikit-build-core / setuptools builds locate the bundled C via:

```python
import midas_ckernel
midas_ckernel.c_src_dir()   # -I path
midas_ckernel.sources()     # forward.c, MIDAS_Math.c, nelder_mead.c, GetMisorientation.c
midas_ckernel.headers()
```

## Bundled sources

| file | role |
|---|---|
| `forward.{c,h}` | unified diffraction forward simulator |
| `nelder_mead.{c,h}` | NLopt-free bounded Nelder-Mead simplex |
| `MIDAS_Math.{c,h}` | linear-algebra primitives |
| `GetMisorientation.{c,h}` | cubic-symmetry misorientation |
| `IndexerConsolidatedIO.h` | shared I/O struct/enum defs |
| `MIDAS_Limits.h` | global array-size limits |

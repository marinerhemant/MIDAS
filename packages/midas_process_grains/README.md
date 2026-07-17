# midas-process-grains

Pure-Python (PyTorch) replacement for `FF_HEDM/src/ProcessGrains.c`. Reads the
binary outputs of the upstream MIDAS pipeline (`OrientPosFit.bin`, `Key.bin`,
`ProcessKey.bin`, `IndexBestFull.bin`, `FitBest.bin`) and emits the canonical
`Grains.csv` / `SpotMatrix.csv` / `GrainIDsKey.csv` files.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_process_grains/notebooks).

## Scope: bit-level parity with the C reference

The package has one shippable mode: **`c_parity`**, which mirrors
`FF_HEDM/src/ProcessGrains.c` exactly. The Stage 1 cluster-build, the Pass A
position+orientation dedup, the confidence cut, and the 47-column
`Grains.csv` / 12-column `SpotMatrix.csv` / `GrainIDsKey.csv` writers all
follow the C source line-for-line.

On the `peakfit_hard` reference dataset (357 k seeds → 22 k grains), the
Python output is **bit-identical** to the C output for every column except
the Kenesei strain tensor — see "Parity verdict" below.

Earlier experimental modes (`legacy`, `paper_claim`, `spot_aware`) shipped in
v0.1 internal builds are still present in the source tree but are not
exposed through the supported public surface. They will be removed in a
future cleanup.

## CLI

```bash
midas-process-grains paramstest.txt 8 --mode c_parity --device cuda
```

The CLI reads `paramstest.txt` from the run directory, writes the three
output files into `--out-dir` (defaulting to the run directory), and exits.
Use `--device cpu` if you do not have a CUDA-capable GPU.

```bash
midas-process-grains paramstest.txt 8 \
    --mode c_parity \
    --device cuda \
    --min-nr-spots 1 \
    --out-dir ./output
```

`--min-nr-spots` matches the `MinNrSpots` parameter in `paramstest.txt`
(Stage 1 cluster-size cutoff). Default is `1`, which keeps every cluster.

## Library

```python
from midas_process_grains.compute.c_parity_run import (
    run_c_parity_pipeline_from_disk,
)

run_c_parity_pipeline_from_disk(
    run_dir="/scratch/.../LayerNr_1",
    out_dir="/scratch/.../LayerNr_1",
    device="cuda",          # or "cpu"
    min_nr_spots=1,
)
```

For lower-level access (run only Stage 1, only Pass A, only the writers,
etc.) see `midas_process_grains.compute.c_parity` and
`midas_process_grains.compute.c_parity_emit`.

## Parity verdict (peakfit_hard, 22 k grains)

| Column | Python vs C max abs diff |
|---|---|
| `GrainID`, OM (9), `X`, `Y`, `Z`, lattice (6), `DiffPos`, `DiffOme`, `DiffAngle`, `GrainRadius`, `Confidence`, **Fable strain** (9), `Eul0`, `Eul1`, `Eul2` | **0** (bit-identical) |
| **Kenesei strain** (9 components) | ≤ 35 µε (NLOPT vs SciPy `lsq_linear` solver convergence) |
| `RMSErrorStrain` | ≤ 0.085 µε |

Cluster identity: 21,504 of 22,003 grains share the same `rep_pos` between
the C and Python runs. The remaining ~2 % is OMP `atomic_test_and_set`
non-determinism in the C source — running C on the same input twice produces
two outputs that disagree on **846 grains** (3.8 %). Python and a current C
rerun agree at **99.58 %** — closer than C agrees with itself across runs.

## Performance

Wall time on a single peakfit_hard run (8-thread alleppey, NVIDIA H100 NVL):

| Pipeline | Wall | CPU time |
|---|---:|---:|
| C ProcessGrains, 8-thread OMP | 50 min | 396 min |
| Python `c_parity`, CPU 8-thread torch | 119 s | 676 s |
| Python `c_parity`, CUDA H100 | **113 s** | **125 s** |

Roughly **27× faster** wall-clock and **190× less CPU** on GPU. The biggest
wins are (a) Pass A's `O(N)` spatial-hash replacing C's `O(N²)` all-pairs,
(b) precomputing the misorientation graph for all spot-overlap candidates in
one batched torch call, and (c) batching all per-grain Kenesei solves into a
single `torch.linalg.solve` over a `(B, 6, 6)` stack.

## Inputs

The pipeline reads the standard MIDAS run-directory layout:

```
<run_dir>/
  paramstest.txt
  hkls.csv
  IDsHash.csv
  SpotsToIndex.csv
  InputAllExtraInfoFittingAll.csv
  Output/
    IndexBestFull.bin
    FitBest.bin
  Results/
    OrientPosFit.bin
    Key.bin
    ProcessKey.bin
```

## Outputs

```
<out_dir>/
  Grains.csv                      # 47 columns, C ProcessGrains layout
  GrainIDsKey.csv                 # one line per kept grain
  SpotMatrix.csv                  # 12 columns, C ProcessGrains layout
  processgrains_diagnostics.h5    # aux diagnostics (skip with --no-diagnostics)
```

### `processgrains_diagnostics.h5:/residuals` (v0.6.0+)

Signed per-spot residual decomposition, collected during the FitBest pass —
this is what `Grains.csv` `DiffPos`/`DiffOme` aggregate, now decomposable:

* `residuals/spot_table` — gzip float32, one row per resolved grain-spot
  claim; column layout = `SPOT_RESIDUAL_COLS` in
  `compute/residual_decomposition.py`:
  `(grain_idx, spot_id, ring_nr, eta_deg, dy_um, dz_um, drad_um, dtan_um,
  dome_deg, internal_angle_deg, r_exp_um)`. `grain_idx` indexes the output
  grain list (NOT GrainID); residuals are obs − exp with position-corrected
  observations; `dome_deg` is wrapped to [−180, 180).
* per-grain `(G,)` arrays: `grain_med_{dy,dz,drad,dtan}_um`,
  `grain_med_dome_deg`, `grain_med_internal_angle_deg`,
  `grain_mad_dtan_um`, `grain_n_spots` (NaN where a grain contributed no
  rows).
* per-ring: `ring_nr`, `ring_med_drad_um`, `ring_drad_ppm`,
  `ring_mad_drad_um`, `ring_n_spots`. **`ring_drad_ppm` is the
  reference-lattice diagnostic**: a consistent |median dR/R| > 200 ppm
  across rings is the signature of a wrong `LatticeConstant` (a₀), absorbed
  as fake hydrostatic strain — the run log warns when it trips.
* eta profile (30° bins): `eta_bin_lo_deg`, `eta_med_{drad,dtan}_um`,
  `eta_med_dome_deg`, `eta_n_spots`.
* global scalars: `overall_med_{dy,dz,drad,dtan}_um`, `overall_med_dome_deg`,
  `overall_mad_{drad,dtan}_um`, `overall_mad_dome_deg`,
  `overall_med_internal_angle_deg`.

`mode="legacy"` (no FitBest pass) emits empty `/residuals` by design.

## Implementation notes

* Stage 1 (`FindInternalAngles` equivalent) does a recursive DFS over the
  `ProcessKey`-defined spot-overlap candidate graph, filtered by misori
  < `0.4°`. The misorientation for every candidate edge is precomputed in
  one batched torch call before the DFS.
* Pass A (`misori < 0.1° AND |Δpos| < 5 µm` dedup) uses a 5 µm spatial hash
  on rep positions to limit pairs to those within the position threshold,
  then vectorised misori on the surviving pairs. Greedy outer-serial dedup
  matches C's order.
* Confidence filter `OPF[26] >= 0.05` (matches C `OPs[ri][22] < 0.05` cut).
* Strain — Fable-Beaudoin from refined lattice (closed form), Kenesei from
  per-spot lstsq (`scipy.optimize.lsq_linear` with the same ±0.01 bounds C
  uses with NLOPT Nelder-Mead). Kenesei is solved in batch over all grains
  in a single `torch.linalg.solve(GTG + λI, GTb)` call when running on GPU.
* Euler angles use C's exact `OrientMat2Euler` algorithm with the
  `sin_cos_to_angle(s, c) = acos(c) if s ≥ 0 else 2π − acos(c)` helper.
  Output is in **radians**, matching C.

See the docstrings in `compute/c_parity.py` and `compute/c_parity_emit.py`
for the full algorithm spec, with line-number references back to the C
source.

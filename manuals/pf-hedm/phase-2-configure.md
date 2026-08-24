# Phase 2 — Configure: params, the zip trap, rings, the FF seed

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).

## 2.1 The parameter file

Start from the calibration `paramstest` (phase 1). PF-specific keys are in
[`PF_PARAMETERS.md`](PF_PARAMETERS.md). The ones that most often go wrong:

- `RingNumbers` — which rings to index/refine on. For **strain**, prefer the bright,
  fully-on-detector rings; the rings used for indexing are not always the best for strain
  (a low-order bright ring beats a corner-clipped high-order one).
- `Hbeam` / `Rsample` — **generous SEARCH BOUNDS, never the physical beam or
  sample** (spine hard rule 5). Tightening them to the true dimensions plops
  solutions onto the bounding box. In PF it is doubly moot: PF fixes each voxel
  to the scan grid and does not fit position at all.
- `BeamSize` — the in-plane beam width; sets the voxel/scan overlap tolerance.
- `SpaceGroup`, `LatticeParameter` — the phase.
- `OmegaRange` — one or more valid ω spans; **gaps** (blocked ranges) are normal and reduce
  the reflections per voxel (envelope §2).

## 2.2 THE ZIP TRAP — analysis params are baked into the zarr

**This is the highest-value trap in pf-HEDM.** `zip_convert` writes the raw frames into a
per-scan zarr (`*.MIDAS.zip`) **together with the analysis parameters in force at
zip-time**. Peakfit (and other stages) then read the parameters **from the zip's stored
`analysis_parameters`, not from your live `paramstest`.**

Consequence: editing `paramstest` after the zips exist changes **nothing**. A wrong
`MaxNPeaks` (peak cap per spot), a wrong threshold, a wrong ring set — all silently persist
from whatever was in force when the zips were written.

**To change any zip-baked parameter you must regenerate the zips.** Confirm the value that
actually reached a zip:

```python
import zarr
z = zarr.open("<one *.MIDAS.zip>", "r")
print(dict(z["analysis_parameters"].attrs) if "analysis_parameters" in z else "check the zip layout")
```

Adopt the **authoritative** parameter file verbatim (do not paste keys from a different
layer's file — a stray `MaxNPeaks 8` from a copy-paste caps every spot at 8 peaks), then
**delete and regenerate all zips**. Verify one regenerated zip before running the full set.

## 2.3 Peakfit on GPU

Scanning peakfit is the compute floor — one core per scan is days. Run it on GPU:

- `--device cuda`, `dtype float32`, shard across the available GPUs.
- Watch for the **dense-frame failure mode**: extremely populated frames (a box-beam FF or a
  noisy scan) can overflow the Triton Jacobian's address arithmetic on GPU. This is fixed in
  current `midas-peakfit`; on an older version the symptom is a CUDA *illegal memory access*
  mid-flush. If you hit it, update the package or fall back to `--device cpu` with the
  thread count set to the core count (a standalone CPU peakfit must set its own
  `OMP/MKL/OPENBLAS_NUM_THREADS`, not inherit `=1` from a GPU-worker script).

An intensity floor on the merged spot list (drop noise-level spots) both speeds indexing and
avoids OOM; record the cut you used.

## 2.4 Obtain the FF orientation seed — you will need it

**Unseeded scanning indexing is intractable** (halt condition). A scanning layer produces
tens of millions of merged spots; combing the full orientation grid blind runs for days and
often writes no output. The tractable path is to seed indexing with far-field orientations.

1. From the matched FF layer (phase 1.4), get the far-field `Grains.csv` (the standard
   `ProcessGrains` output: `%ID O11..O33 X Y Z a b c … Confidence …`).
2. Point the seeding stage at it: `SeedingConfig.mode = "ff"`,
   `grains_file = <FF Grains.csv>`. The seeding stage converts it to
   `UniqueOrientations.csv` (14-col: grainID + 4 pad + 9 OM), which the per-voxel scanning
   indexer uses as per-voxel candidate seeds instead of the full grid.
3. The **c-omp** indexer additionally needs a `GrainsFile <path>` line in its paramstest to
   take the seeded path (`isGrainsInput=1`); `midas_pipeline` wires this. Without it the C
   binary silently combs the full grid — indistinguishable from unseeded, and just as slow.

> ⚠️ `Grains.csv` (ID + 9 OM in cols 1–9) is **not** `UniqueOrientations.csv` (OM in cols
> 5–13). The seeding handoff keys header detection off the OM column names and accepts both
> `GrainID` and `ID` spellings.

Seeded, a full scanning layer indexes in tens of minutes instead of days.

## 2.5 Pin the reference cell — `LatticeConstant` is the zero of the strain

**`LatticeConstant` is not a starting guess. It is the origin of every strain
number the run will report,** and getting it wrong is silent in exactly the way
this doc set keeps warning about.

The C refiner's strain comes from `StrainTensorKenesei`
(`packages/midas_fit_grain/c_src/FitUnified.c:1061`), which fits six components
to `(dsObs − ds0)/ds0` where **`ds0` is the nominal d-spacing implied by
`LatticeConstant`** — not by the refined cell. Its box is
`MargStrain` (default **±0.01 = ±10000 µε**; it was a compiled-in constant before
2026-08-21). So a reference wrong by 0.7 % spends most of the box before any real
strain is measured, and components rail at the bound.

**Measured on NMC811 at 1-ID** (`bt_1id_jun25b`, s5/L9), pristine
`2.8691 / 14.212` vs the sample's own pinned `2.85074 / 14.32299`, everything
else identical:

| | pristine ref | pinned ref |
|---|---|---|
| voxels solved | 84 | **123** |
| completeness median | 0.618 | **0.833** |
| voxels with a railed strain component | 10 (11.9 %) | **0** |
| max abs E | 10000 µε | 6635 µε |

Note it is **not only a strain problem** — completeness rose by a third. A naive
check ("the ring shift is 700 µm, inside `MarginRadial` 800, so it costs no
spots") was wrong: the shift eats the tolerance budget alongside every other
error.

### How to pin it — from the rings, never from the refined cells

```python
import numpy as np, collections
from midas_hkls import refine_lattice_from_d_spacings

# Ttheta + RingNumber straight out of transforms — no indexing, no refinement
tt = collections.defaultdict(list)
for line in open("InputAllExtraInfoFittingAll0.csv").readlines()[1:]:
    p = line.split(); r = int(float(p[5]))
    if r > 0: tt[r].append(float(p[7]))          # col 5 RingNumber, col 7 Ttheta

rings = sorted(tt)
d_obs = [LAMBDA / (2*np.sin(np.radians(np.median(tt[r]))/2)) for r in rings]
hkls  = [hkl_of_ring[r] for r in rings]          # from the RUN'S OWN hkls.csv
fit   = refine_lattice_from_d_spacings(hkls, d_obs, "hexagonal")
print(fit.lattice, fit.rms_strain, fit.residual_strain)
```

`1/d² = hᵢG*ᵢⱼhⱼ` is **linear** in the reciprocal metric tensor, so this is a
direct least squares that takes **no starting cell** — which is the whole point.

> **Do not instead average the refined per-voxel cells.** The refiner starts from
> `LatticeConstant` and only partly leaves it, so that average returns roughly
> what you fed in. Measured: one iteration drifted a further −3740 µε in `a` and
> **+6361 µε in `c` without converging** (ratio 0.83 per pass). Any
> equilibrium-based recovery fed refined cells inherits the same loop.

### Two traps in the fit itself

1. **Drop or down-weight the lowest-angle ring.** `dd/d = cot(θ)·dθ`, so at
   2θ = 2.85° (NMC 003) a **0.006° systematic in 2θ becomes 2105 µε in d**,
   against 596 µε for a ring at 10°. On the reference dataset that ring's
   residual was **−1696 µε** while the other four sat inside ±340 µε; dropping
   it took the fit RMS **776 → 171 µε**.
2. **Never weight by the statistical error of the ring centroid.** With ~160 k
   spots the SEM is ~6 µε, so 1/σ² weighting hands the *least reliable* ring the
   *largest* weight. The systematic floor dominates. Use uniform weights, or
   weight by `tan²θ`. Across {uniform, tan²θ, drop-low-ring} the pinned cell
   moved only **83 µε in a and 313 µε in c** — that spread is the honest
   uncertainty.

### Scouting a cell is cheap — it needs no indexing

`Ttheta` and `RingNumber` come out of **transforms**, so the *prep* half alone
(`--skip indexing --skip refinement --skip find_grains …`) is enough to pin a
cell. On a multi-sample campaign, scout **one layer per sample** that way —
minutes each, not a full layer — then launch. Do not start a long campaign on an
unpinned cell meaning to "fix the strain afterwards": the reference is baked into
every strain number *and* it moves completeness (measured 0.618 → 0.833).

**Pin per SAMPLE, not globally.** Measured on four NMC811 cathodes from one
beamtime: two discharged (c/a 4.940, 4.942) and two delithiated (5.068, 5.078) —
a **10 000–15 000 µε** difference in `a` and `c`, more than the whole ±10 000 µε
strain box. One global cell would have injected more error than it removed.

### Cross-check it against equilibrium

For an unloaded sample `⟨σ⟩ = 0`, which determines the cell independently:

```python
from midas_stress import recover_d0_anisotropic
r = recover_d0_anisotropic(lattice_params, pinned_cell, stiffness,
                           orientations, crystal_system="hexagonal")
```

Use the **anisotropic** version for any non-cubic phase: `recover_d0` scales
`a`, `b`, `c` by one factor, which cannot represent an error that is negative in
`a` and positive in `c`. Read `condition_number` — and note the counter-intuitive
part: a **weak** texture is the ill-conditioned case (uniform orientation
averaging projects onto the isotropic subspace, so the `a` and `c` responses
collapse onto each other and `cond` grows with N), while a sharp texture
separates them cleanly.

The two routes agreeing is the gate. On the reference dataset the powder cell and
the equilibrium recovery agreed to **−994 µε in a, +587 µε in c**, versus
−3740 / +6361 for the un-pinned reference. Absolute scale still rides on λ and
`Lsd` (they are degenerate with a cell dilatation), so the cell is only ever on
the calibrant's length scale.

When params are authoritative, zips regenerated, the reference cell pinned, and
the FF seed staged, go to [`phase-3-run.md`](phase-3-run.md).

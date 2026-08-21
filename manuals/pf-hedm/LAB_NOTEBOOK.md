# pf-HEDM lab notebook — reference campaign

**Companion to `README.md`.** The handbook says what to do; this records what was found, how
it was measured, and what turned out to be wrong. Kept apart on purpose: a handbook stays
short enough to follow, a campaign record stays honest enough to stop a refuted idea coming
back.

**One notebook per campaign, started on day one.** The retractions decay fastest.

`§n` means a section of *this* file; handbook sections are `Handbook §n`.

**Reference campaign.** A full from-scratch pf-HEDM reconstruction of one layer of a
**cracked additively-manufactured FCC-Ni specimen** on a heavily-attenuated scanning-3DXRD
dataset (`att5`-class), followed by per-voxel peak-shape strain. The specimen carried a
branching crack; the scan was 259 translations → a 259×259 = 67 081-voxel layer. Details
anonymised.

---

## 1. What this campaign established

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | Full pf-HEDM recon from raw frames → grain map → per-voxel peak-shape strain runs end-to-end | VERIFIED | §3, §4 |
| 2 | The c-omp refiner needs a 5-col PF `SpotsToIndex.csv`; without it it refines nothing | FIXED | §2 |
| 3 | pf-odf requires dark-subtracted patches; raw pedestal biases strain −30 % and reshapes the field | FIXED (opt-in) | §3 |
| 4 | Unseeded scanning indexing is intractable; FF-seed makes it tens of minutes | VERIFIED | §3 |
| 5 | KAM/GROD localise the crack deformation zone; strain is spatially real but magnitude-provisional | VERIFIED | §4 |
| 6 | Illumination-gating the patch extraction could not be validated on this data | RETRACTED | §5 |

## 2. Defects fixed

- **c-omp PF refiner silently refined nothing.** The C refiner's PF path indexes each voxel's
  seed via a 5-col `SpotsToIndex.csv` the python indexer never emits → `nSpotsBest≤0` for
  every voxel → zero output, exit 0. Fixed by synthesising the file from `IndexBest_all.bin`
  (`midas_fit_grain.scan_seed`, same highest-completeness pick as the python refiner) and
  adapting the multi-block `FitBest_*.csv` into `Result_OrientPos_voxel_*.csv`
  (`midas_fit_grain.fitbest_adapter`); `midas_pipeline` wires both. Turned a GIL-bound
  multi-day python PF refine into a ~90 s c-omp run. Validated byte-identical seed on the
  67 081-voxel layer; c-omp vs python refine agree <0.2° orientation, <0.01 Å lattice.
- **GPU peakfit illegal-memory-access on dense frames.** int32 address overflow in the Triton
  Jacobian stride at high batch index. Fixed with an int64 cast. (Six wrong hypotheses first
  — see §5.)
- **Binning OOM.** (spot × η-bin × ω-bin) pair explosion killed GPU and CPU; fixed by
  per-ring pair chunking (`MIDAS_BIN_PAIR_CHUNK`). `StepsizeOrient` is overloaded (indexer
  grid + binning ω-margin) — coarsening it to speed indexing OOMs binning.
- **`find_grains` never finished** on the high-spread cracked map (per-voxel clustering,
  O(N²) dedup). Worked around with connected-components segmentation of the cleaned refined
  orientations → `voxel_grid.csv` in seconds (Handbook phase 3.5).

- **Per-voxel strain railed at the compiled-in ±10000 µε box** (`bt_1id_jun25b`,
  NMC811, 2026-08-21). `StrainTensorKenesei` measures `(dsObs − ds0)/ds0` against the
  `ds0` implied by `LatticeConstant`, inside a hardcoded `xl/xu = ±0.01`. The
  parameter file carried *pristine* NMC811 while the cell was charged (c/a 5.07 vs
  4.95), so ~0.7 % of the box was gone before any real strain was measured and
  **11.9 % of voxels railed**. Fixes shipped: the box is now the **`MargStrain`**
  parameter (`FitUnified.c`, `MIDAS_ParamParser.[ch]`, `midas_params/registry.py`,
  `midas_transforms/params.py`; 0 ⇒ the 0.01 default, and the writer emits the key
  only when set so existing files stay byte-identical to the C writer), and
  **`midas_hkls.refine_lattice_from_d_spacings`** pins the reference cell from
  observed ring positions. Pinning it: voxels **84 → 123**, completeness median
  **0.618 → 0.833**, railed voxels **11.9 % → 0 %**.
- **`--num-files-per-scan` defaulted to 1 and produced a clean 0-voxel layer.**
  The parameter file's `NrFilesPerSweep 1440` is written into the zarr but does not
  drive the file list, so `exchange/data` came out `(1, ny, nz)`, all 23 stages ran
  and the layer finished exit 0 with nothing. Also: `--scan-work-dir` must be
  absolute (a relative path is joined twice), and stale
  `InputAllExtraInfoFittingAll*.csv` + `midas_state.h5` make `transforms` report
  "13 ok" in 0.04 s from cache so indexing runs on the previous attempt's spots.

## 3. Method findings

- **Pin the reference cell from the rings, never from refined per-grain cells.**
  Averaging refined cells is a feedback loop — the refiner starts at
  `LatticeConstant` and only partly leaves it. Measured: recovering the reference
  from the refined output drifted a further −3740 µε in `a` and **+6361 µε in `c`,
  ratio 0.83 per pass, i.e. not converging**. The powder route is a *direct* linear
  least squares on `1/d²` (linear in the reciprocal metric tensor) and takes no
  starting cell, so it cannot drift. The two independent routes — powder rings and
  mechanical equilibrium (`recover_d0_anisotropic`) — then agreed to **−994 µε in
  `a`, +587 µε in `c`**, versus −3740 / +6361 before pinning. Agreement of the two
  is the gate.
- **Down-weight the lowest-angle ring; never weight by its statistical error.**
  `dd/d = cot(θ)·dθ`: at 2θ = 2.85° (NMC 003) a 0.006° systematic in 2θ is
  **2105 µε in d**, against 596 µε at 10°. That ring's residual was −1696 µε while
  four others sat inside ±340 µε. With ~160 k spots its centroid SEM is ~6 µε, so
  1/σ² weighting hands the least reliable ring the largest weight. Dropping it took
  the fit RMS **776 → 171 µε**; across {uniform, tan²θ, drop-low-ring} the cell moved
  only **83 µε in `a`, 313 µε in `c`** — that spread is the honest uncertainty.
- **Weak texture, not sharp texture, is the ill-conditioned case for an anisotropic
  d0 recovery.** Averaging the Mandel rotation over a uniform orientation
  distribution projects onto the isotropic subspace, so the `a` and `c` stiffness
  responses collapse onto `C{I}` and the condition number *grows with N* (2.8 for a
  single orientation or a 10° fibre; 23 at N=100 uniform; 142 at N=1000). A single
  crystal separates them cleanly because its stiffness is anisotropic. The first
  test written for this asserted the opposite and failed — which is how it was found.
- **The anisotropic d0 answer barely depends on the elastic constants.** Scaling the
  whole stiffness tensor ±30 % changes it *not at all* (the factor cancels in the
  least squares); swinging C33 140→260 and C13 20→90 moved `a` by 198 µε and `c` by
  718 µε. So poorly-known single-crystal constants (NMC811, LLZO, new phases) are not
  load-bearing here.
- **Peakfit is I/O-bound, and the page cache is the lever.** On 64 cores it pegs only
  ~3.5–4 cores (load ~3); `--scan-workers 8` was no faster than 4. Peakfit run
  straight after `zip_convert` took **467 s/scan**; the same work on cold zips took
  **1009 s/scan**. Drive layer-by-layer (zip → peakfit → …), not all-zips-then-peakfit.

- **Dark subtraction matters, quantitatively.** pf-odf's loss has no additive background
  term. Raw patches (≈1850-count pedestal) vs per-patch-border dark subtraction: median von
  Mises strain 2240 → 1581 µε (−30 %), and the *field pattern* changed — raw vs corrected
  only **0.26-correlated**, and Moran's I fell 0.21 → 0.09 (the pedestal added spurious
  smooth structure). Corrected is still spatially real (p ≈ 0.005).
- **Read the zarr, not the raw h5.** The Blosc-lz4 `*.MIDAS.zip` is ~2.85 TB vs 6.2 TB raw,
  byte-identical frames; cold-read ~4.8× faster scattered, ~1.6–1.8× sequential. Frames are
  chunked one-full-frame-per-chunk, so sub-frame reads decompress whole frames — patch
  extraction is I/O-bound; do it single-pass (read each frame once, distribute to all grains).
- **`positions.csv` is file-order** (`position[file n] = centre − n·step`, descending), not
  sorted; the voxel grid is the Cartesian product of the sorted positions. Getting the order
  or sign wrong mirrors the map.
- **`MaxNPeaks` and analysis params are baked into the zarr** at zip-time; peakfit reads them
  from the zip, not the live `paramstest`. A stray `MaxNPeaks 8` copied from another layer
  capped every spot until the zips were regenerated.
- **Attenuation is the strain limiter.** At `att5`, bright peaks sat ~150–350 counts over a
  ~1850 background and high rings near noise → ≈1 usable reflection/voxel → strain
  identifiability-limited. Orientation was excellent at the same attenuation.

## 4. Scientific findings

- **Orientation, KAM, GROD are the robust products.** Full-layer IPF, {100} pole figures
  (FF↔PF consistent), and KAM/GROD (grain-boundary overlaid) all resolved cleanly. KAM and
  GROD **both localised** to a central band — the crack's process zone — independently.
- **Per-voxel strain is spatially real but provisional.** Full-layer von Mises median
  ~2000 µε, global Moran's I ≈ 0.15 (structured, not noise), with a low/undetermined zone at
  the crack (broken lattice does not diffract). Trust the pattern, not the magnitude.
- **Tomography confirmed a branching crack** at the pf layer height (Z registered via the
  scan `samY` ↔ reconstructed slice; in-plane flip/rotation-centre was the un-recorded half).
  "Bifurcation" in the specimen was literally the crack branching — matching where the
  diffraction deformation localised.

## 5. Open questions, and claims that were RETRACTED

- **RETRACTED: illumination-gating the extraction (read only the scans that light each
  grain).** The gate (`y_rot = px·sinω + py·cosω` vs beam position) predicted illuminated
  scans, but on this data only 2/8 bright spots' peaks fell in the predicted range. Killed by:
  the att5 signal is too weak to establish ground-truth illumination, and a "bright" scan for
  a spot can be a **different grain** diffracting to the same detector patch (cross-grain
  contamination). An unvalidated gate would silently drop real signal from every grain — not
  shipped. The single-pass full read (correct) was used instead.
- **RETRACTED: the naive centre-pixel alignment gate.** A gate requiring a peak in the
  central few pixels read ~2–4 % pass and looked like a convention failure. It was measuring
  spot *weakness* (att5), not misalignment — the two brightest reflections landed within 3 px
  of prediction. Use a bright spot, not a blanket gate (Handbook phase 4.3).
- **RETRACTED (six times): early GPU-crash hypotheses.** The Triton illegal-access was
  blamed on dtype, memory exhaustion, batch size, a prefetch loop, and an async race before
  the int32 stride overflow was instrumented. The lesson: instrument the actual failing
  index, do not theorise from the symptom.
- **RETRACTED: "the reported strain is decoupled from the refined lattice."** The
  reported `E11..E33` are in the **sample** frame (fitted on `gobs`), while strain
  rebuilt from the lattice parameters is in the **crystal** frame. Compared raw, the
  correlation is −0.08 and looks like a serious defect; rotated with the row's own
  orientation (`O E Oᵀ`) it is **+0.84…+0.94** and the refiner is self-consistent.
  Compare invariants first, then rotate.
- **RETRACTED: "the strain columns rail at 5×10⁹ µε."** The `E` columns are
  *already* microstrain; the 5×10⁹ came from multiplying by 1e6 a second time. The
  real rail is at ±10000 µε and needed the unit check to see.
- **RETRACTED: "the wrong reference cell costs no spots."** Argued from the ring
  shift (700 µm) fitting inside `MarginRadial` (800 µm). Measured the other way:
  pinning the cell moved completeness **0.618 → 0.833** and voxels 84 → 123. A
  static margin comparison ignores that the shift eats the tolerance budget
  alongside every other error.
- **CORRECTED: this doc set previously told the reader `Hbeam` was "the true
  per-layer beam, never the sample size."** That is wrong and cost a session.
  `Hbeam`/`Rsample` are **generous search bounds**; tightening them to the real
  dimensions plops solutions onto the bounding box. In PF it is doubly moot — PF
  fixes voxel positions to the scan grid and does not fit position at all.
- **OPEN: cross-modal (tomo) in-plane registration.** The tomo reconstruction's flip and
  rotation-centre pixel were not recorded; the crack↔KAM overlay needs them or a shared
  fiducial. Z is registered; in-plane is not.

## 6. Measurement ledger

| quantity | value | file + command that produced it |
|---|---|---|
| voxels in layer | 67 081 (259²) | `Output/voxel_grid.csv` row count |
| c-omp refine time | ~90 s / layer | refiner log (vs multi-day python) |
| seed synthesis | 67 081 rows, byte-identical to hand-rolled | `midas_fit_grain.scan_seed.write_pf_seed_file` on `IndexBest_all.bin` |
| c-omp vs python refine | <0.2° miso, <0.01 Å lattice | sample voxels 50/100/500 |
| dark-sub strain shift | 2240 → 1581 µε median; raw/corr 0.26 | `assemble_grain_patch_data(subtract_background=…)` + fit |
| full-layer strain | median ~2000 µε, Moran's I ≈ 0.15 | `eps_grain*.npy` assembled on the voxel grid |
| zarr vs h5 read | 2.85 vs 6.2 TB; ~1.6–4.8× faster | cold benchmark, `exchange/data` |

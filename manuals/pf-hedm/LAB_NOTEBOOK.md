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

## 3. Method findings

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

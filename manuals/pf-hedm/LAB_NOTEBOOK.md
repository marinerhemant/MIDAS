# pf-HEDM lab notebook — reference campaign

**Companion to `README.md`.** The handbook says what to do; this records what was found, how
it was measured, and what turned out to be wrong. Kept apart on purpose: a handbook stays
short enough to follow, a campaign record stays honest enough to stop a refuted idea coming
back.

**One notebook per campaign, started on day one.** The retractions decay fastest.

`§n` means a section of *this* file; handbook sections are `Handbook §n`.

**Three campaigns are recorded here.** They cover different halves of the technique and
different stations; read the one that matches your data first, then the others for their
retractions.

| | §1–§6 | §7 | §8 |
|---|---|---|---|
| station | **1-ID scanning** | **20-ID HT-HEDM Varex** | **1-ID scanning** |
| specimen | cracked additively-manufactured FCC-Ni, heavily attenuated (`att5`-class); plus NMC811 cathodes for the reference-cell work | FCC, `nf709` "edge" set A — rotation axis placed just inside a sample edge | NMC811 cathode, four charge states, 32 banked layers |
| layer | 259 translations → 259 × 259 = 67 081 voxels | 51 translations → 51 × 51 = 2601 voxels at 1 µm | 13–19 translations → 169–361 voxels at 1.5 µm |
| covered | raw frames → grain map → **per-voxel peak-shape strain** | the **reconstruction-space half**: sinograms, positions, shapes, the sample boundary | **is the map real?** — the ω-shuffle null, the chance ceiling, and scanning vs a line-focus far-field exposure |
| code | `midas_pipeline` | the **legacy v11 C `pf_MIDAS.py`** (what the beamline had installed) | `midas_pipeline` (c-omp index + refine) |

§8 contributed [`phase-7-validation.md`](phase-7-validation.md), and its §8.7 is the
longest retraction list in this file — read it before re-deriving anything about
version drift or LF/PF spot matching.

Details in §1–§6 anonymised.

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

---

## 7. The 20-ID Varex campaign — reconstruction space

**Scope.** One layer, `nf709` "edge" set A: 51 translations at 1 µm, 51 × 51 = 2601 voxels,
FCC, Varex, run on the **legacy v11 C `pf_MIDAS.py`**. The rotation axis was placed just
inside a sample edge on purpose. The question was grain **shape**; the answer was that
shapes do not come back, and the durable output is everything that was learned proving it.

> ⚠️ **Nothing in §7 has been through `/verify`.** "Established" below means measured and
> internally controlled, not adversarially verified. Everything positional rests on **one
> layer of one sample** — the out-of-sample test (§7.3) could not test position at all.

## 7.1 What this campaign established

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | The reconstruction completed and the **point-by-point map is sound** | ESTABLISHED | §7.6 |
| 2 | Contaminated "vertical stripe" sinogram rows are real, and filtering them is a large win for **position** | ESTABLISHED, shipped | §7.3, Handbook phase 6 §6.5 |
| 3 | Grains filling the scanned field can be **flagged** from occupancy — but must not be filtered | ESTABLISHED, shipped | §7.3, phase 6 §6.6 |
| 4 | The **sample boundary** comes from the spot-count sinogram, never from completeness | ESTABLISHED | §7.4, phase 1b |
| 5 | `-doTomo 1` **degrades** the point-by-point map | ESTABLISHED | §7.3 |
| 6 | **Grain shapes are not recoverable and the cause is unknown** | OPEN — eleven mechanisms tested | §7.5 |
| 7 | Eleven mechanisms "refuted"; **four later requalified** as contributing-but-insufficient | CORRECTED | §7.5 |

## 7.2 Defects found

- **FBP reconstructions were mis-registered by one voxel** (fixed, `midas_pipeline` 0.11.0).
  The crop extracting the `n_scans × n_scans` field used `recon_dim // 2 - n_scans // 2`
  instead of `(recon_dim - n_scans) // 2`; those differ for every odd `n_scans`. Point
  phantoms came back **1.00 µm low in both axes** across all 17 odd `n_scans` tested — a
  *constant* offset, so nothing looked broken while every shape was silently mis-registered
  against the voxel map it was being compared to. The "obvious" centre-of-pixel rewrite is a
  trap: it is an exact half-integer for odd `n_scans`, so banker's rounding reproduces the
  bug at 33, 37, 45, 49, 53, 65, 97, 101, 129 while passing at 31, 35, 47, 51. For **even**
  `n_scans` an irreducible −0.5 µm offset remains; stated in the code rather than hidden.
- **Legacy C `spotPositions_*.bin` is 97.7 % unwritten** — 604 of 26 119 spot-bearing bins
  carry values, the rest sit at the `-1` initialiser. Gated on `idMap_scanNr[gid] == scanNr`
  (`FF_HEDM/src/findSingleSolutionPFRefactored.c:2655`), with a missing `Result_*.csv`
  silently skipped and the error logged only for `scanNr < 3`
  (`findSingleSolutionPFRefactored.c:2683-2687`). **Not patched, deliberately**: the Python
  replacement has no such guard (`midas_pipeline/find_grains/_patches.py:115-122`). ⚠ The
  filename also changed — C `spotPositions_*.bin`, Python `spotPos_*.bin` — so a reader
  keyed on the old name finds nothing rather than erroring.
- **The `abs` sinogram variant came back degenerate** on this run (residual
  1.000/1.000/1.000/0.000). Looks broken; unexplained, and not chased.

## 7.3 Method findings

- **`-doTomo 1` degrades the point map.** Tomo-seeded re-index: **2433 of 2601** voxels
  refined and **367** below completeness 0.5, against **2601** and **11** for the direct
  run. Same data, same geometry.
- **The concentration filter.** Rows that also collect a neighbour's spot smear across every
  scan position and drag the fitted position. Flagging rows carrying < 0.35 of their
  intensity within the fitted sinusoid's band caught **16 of 958 rows (1.7 %)** and moved
  position fits: grain 3 **5.59 → 1.11 µm**, grain 8 5.27 → 2.14, grain 9 3.41 → 2.06,
  grain 2 5.42 → 3.29. The MLEM residual moved **0.798 → 0.797** — the sixth failed attempt
  to move it. *A large win for position; nothing at all for shape.*
- **The stripe rows are real contamination, not a plotting artefact.** They are *stronger*
  than clean rows (median total I 470 k vs 310 k; `corr(conc, log I) = −0.254`) and span
  44 % more scan bins (36 vs 25). **Not** from the other listed grains (contested-spot
  enrichment 0.8×) and **not** ring-clustered (0.0–3.6 % per ring against 1.7 % overall —
  but only 16 events, low power). **Untested: intruders from grains outside the field.**
- **Out-of-sample test of both diagnostics** (preregistered), on five independent layers of
  a *different* sample at 21× coarser sampling (16 positions at 21.3 µm), driven through the
  shipped pipeline with nothing retuned:
  - ★ **The 0.35 threshold transfers** — 4 of 5 layers inside the preregistered 1–6 %
    flagged band. The strongest result of the test.
  - ⚠ **The benefit is smaller off-sample, and the verdict is INCONCLUSIVE**: median gain
    16–34 % every layer, never reaching the 40 % CONFIRM bar and never dropping below the
    15 % REFUTE bar. Bimodal within a layer (46–91 % on some grains, ~0 on others, **−1 %
    and −2 % on two — it can very slightly harm**). This campaign's flagged grains gained up
    to 80 %; do not promise that elsewhere.
  - ★ **The occupancy flag behaved correctly out of sample**: zero flagged on all five
    layers, max 0.31–0.50 against the 0.65 cutoff — the right answer for a ±160 µm field of
    small grains, and predicted in advance.
  - ⛔ **Position tracking was UNTESTABLE there**: 210 voxels over 23 grains (~9 each) on a
    16 × 16 grid, no grain reaching the 40-voxel minimum.
- **Flag, never filter.** Excluding the occupancy-flagged grains took agreement with the
  point-by-point map from **47.8 % to 11.0 %** — the largest grain is most of the material,
  and its voxels then go to whichever small grain wins by default.
- **IPF colouring uses the third ROW of the orientation matrix**, verified by measurement:
  within one grain, rgb std **0.000 / 0.0015 / 0.0009** by row against **0.257 / 0.348 /
  0.318** by column. MIDAS stores `v_lab = OM · v_crystal`.
- `midas_stress.misorientation_om_batch` wants **flat 9-vectors**, not `(n, 3, 3)`.
- `spot_meta` columns are **`eta, 2theta, yCen, zCen`**
  (`midas_pipeline/find_grains/_patches.py:76`); cols 2–3 are lab positions in µm.

## 7.4 Scientific findings

- **The point-by-point map is the sound product**, mask-free: median completeness **1.0000**
  over all 2601 voxels, mean 0.9277, min 0.4453, 11 below 0.5. Inside the reconstructable
  disc (r ≤ 24 µm, 1793 voxels) median 1.0000, mean 0.9873, min 0.8359. ω residual 0.058°,
  internal angle 0.087°.
- **Grain POSITION from sinograms is excellent** — sinusoid fit rms **1.27–2.07 µm** on
  clean single grains, agreeing with the voxel map to about 1.7 µm.
- **A sample edge exists, and its distance is measurable to ~1.5 µm.** Two *grain-free*
  routes: spot-count sinogram wedge `y(0) = +14.50 µm`, completeness-null residual
  `y(0) = +16.04 µm` (rms 2.51 µm over 51 rows). Four controls passed, the strongest being
  the **180.6°-separated partner wedge at flipped sign of s**, and its disappearance in the
  neighbouring scan set after the sample moved 15 µm. Consequence: **560 of 2601 voxels
  (21.5 %) are vacuum**, and the dominant grain shrinks 1255 → 1012 voxels (40.0 → 35.9 µm)
  while the other four are untouched. Full method: Handbook phase 1b.
- **Vacuum is not empty.** Vacuum voxels share beam lines with material further along,
  inherit that grain's orientation and score ~0.92. Masked: material median **1.0000** vs
  vacuum **0.9219**.
- **Agreement must be scored against the majority-class null.** The tomographic map agreed
  with the point-by-point map on **60.1 %** of voxels against a constant-map null of
  **65.2 %**; Cohen's κ = 0.399. Below its own null is not agreement.

## 7.5 Open questions, and claims that were RETRACTED or REQUALIFIED

- **OPEN: grain shapes do not reconstruct, and the cause is unknown.** The residual is
  **0.82–0.84 and invariant** across FBP / SIRT / MLEM, ± support, all sinogram variants,
  and self-fitted vs borrowed reference masks. Eleven mechanisms were each preregistered and
  tested. The full simulated stack reaches an artifact level of **0.239**, matching the two
  real grains that reconstruct *well* (0.094, 0.211) and falling **2.0× and 3.8× short** of
  the two that fail (0.476, 0.896). The modelled physics accounts for the successes and not
  the failures.
- **★ CORRECTED: "eleven mechanisms refuted" overstates it.** Every refutation was scored
  primarily with **dice**, which thresholds at the true voxel count and is blind to anything
  below the blob amplitude. Re-scored on out-of-mask energy — no new simulation, the saved
  reconstructions — **four requalified as contributing but insufficient**: absorption
  (+184 %), extinction (+47 %), detector merging (+396 %), thresholding (+385 %), while dice
  moved only 0.003–0.042 on each. **Do not use dice on this problem again**; four separate
  requalifications trace to it.
- **★ VALIDATED INSTRUMENT: half-split consistency.** Reconstruct each grain from two
  disjoint halves of its rows and correlate. No ground truth, no mask, no chance floor, and
  a disc cannot game it. Of five metrics tried it is the only one that survived contact with
  an adversarial case: it separated a self-consistent sinogram (0.82–0.92) from a
  noise-dominated one (≈ 0) where dice reported merely "somewhat worse".
- **REFUTED: rebuild the sinogram from `RawSumIntensity`.** Predicted to repair the grains
  whose `IntegratedIntensity` extrapolates 19.4× and 8.9× beyond the raw counts in its own
  window. Outcome: **no grain improved and every one degraded** (mean −0.050 where CONFIRM
  needed ≥ +0.15), and half-split correlation **collapsed from 0.82–0.92 to ≈ 0**. Two
  halves reconstruct to uncorrelated images — the raw window sum is noise-dominated. *The
  peak fit is doing essential work; it is what makes the sinogram self-consistent at all.*
  Consequence: **the int/raw ratio diagnostic is undermined** and must not be carried
  forward as evidence — its denominator is now known to be noise.
- **REFUTED (twice): |F|²·Lp normalisation of the sinogram rows.** Each row is a different
  reflection, so FBP is fed projections on inconsistent scales — a textbook streak cause,
  and MIDAS already has the correction. Powder-Lp (spread 21.6× across rings): mean change
  in dice-above-chance **−0.004** against a ≥ +0.10 bar, half-split −0.082. η-resolved
  (spread 36.2×): **+0.010** and +0.044. Both refuted.
- **REFUTED: MIDAS edge-padding replication.** `Pad()` replicates detector column 0 and
  column 50 into the pads (51 → 64) and `reconCentering` replicates again (64 → 128), so a
  row with intensity at either edge is smeared across ~38 padded columns. Doubling
  `recon_dim` changed mean half-split by **−0.025** against a ≥ +0.05 CONFIRM bar; high- and
  low-edge-exposure grains moved together. Refuted.
- **RETRACTED: "the edge is 12 µm off."** It used the tomographic **max normalised grain
  intensity** map as ground truth. That map answers *"did one of the listed grains
  reconstruct here"*, **not** *"is there material here"* — a band with material but no listed
  grain reads as dark. **Never locate a sample boundary with it.** Downstream numbers
  computed against `y > 14.5` are approximately right, not invalid.
- **CORRECTED: the edge tilt was drawn with the wrong sign.** Under the validated convention
  `s = +x sin φ + y cos φ` it is **+2.36°**; the figures used −2.36°. Only the sign as drawn
  was wrong — the distance never moved.
- **OPEN: two `s(ω)` conventions were used and never reconciled.** The completeness test used
  `s = −x sin φ + y cos φ`; the sinogram forward-model validation used `s = +x sin φ +
  y cos φ` with bin 0 ↔ `s = −25 µm`; and the spot-count sinogram the edge was fitted from is
  indexed in `positions.csv` **file order** (index 0 = +25 µm), the opposite sense to the
  validated projector. The *reconstruction code's* convention is settled and tested
  (`midas_pipeline/recon/fbp.py:176`); whether the spot-count histogram is read in the same
  sense is not. The edge **distance** is unaffected; the **tilt sign and handedness** are.

## 7.6 Hedges that must NOT be upgraded

- **The edge TILT is unresolved** — +2.36° (sinogram), −8.48° (completeness), +4.80° (tomo).
  **Quote the distance, never the tilt.**
- **`positions.csv` handedness is a CONVENTION, not a measurement.** The translation motor
  readbacks were constant across all 51 files in both scan sets — the translation was never
  logged. The map's handedness rests on the descending +25 → −25 convention.
- **Deviatoric strain magnitudes (600–1300 µε) are provisional** — 5 orientations, one
  covering most of the field, too few to average out a geometry error. The hydrostatic
  −811 ± 30 µε is a d0/`Lsd` offset (it implies a0 ≈ 3.5961), **not sample strain**.
- **Grains 0, 2 and 9 being BLENDS is provisional** — one peeling algorithm, no independent
  check; likewise grain 4 being centred outside the field (r = 34.8 µm).
- **"5 orientations" ≠ 5 grains.** Same-orientation neighbours at different positions merge
  under orientation clustering and separate again in the sinogram.
- **Everything positional rests on ONE layer of ONE sample.** Say so wherever it is used.

## 7.7 Measurement ledger

| quantity | value | file + command that produced it |
|---|---|---|
| layer | 2601 voxels (51²) in 9305 s on 64 CPUs | legacy `pf_MIDAS.py -doTomo 0 -numFrameChunks 100` |
| completeness, all voxels | median 1.0000, mean 0.9277, min 0.4453, 11 below 0.5 | `Recons/microstrFull.csv` col 27 |
| completeness, r ≤ 24 µm (1793 vox) | median 1.0000, mean 0.9873, min 0.8359 | same, disc-masked |
| ω residual / internal angle | 0.058° / 0.087° | run log |
| `-doTomo 1` cost | 2433/2601 voxels, 367 below 0.5 | tomo-seeded re-index vs direct |
| concentration filter | 16/958 rows (1.7 %); grain 3 5.59 → 1.11 µm; MLEM residual 0.798 → 0.797 | `sinogram_concentration` + `apply_concentration_filter` at 0.35 |
| filter, out of sample | 2.2–9.9 % flagged, gains 16–34 % over 5 layers | shipped pipeline, nothing retuned |
| occupancy | 2 of 10 flagged at 0.84 / 0.78; rest ≤ 0.51 | `sinoOccupancy_<nG>.bin` |
| excluding flagged grains | agreement 47.8 % → 11.0 % | grain-ID map vs point-by-point |
| tomo vs point map | 60.1 % agreement, null 65.2 %, κ = 0.399 | voxel-wise, same voxels |
| edge distance | +14.50 µm (sinogram) / +16.04 µm (completeness null), rms 2.51 µm | two grain-free routes |
| vacuum | 560/2601 voxels (21.5 %); material median completeness 1.0000 vs vacuum 0.9219 | masked at `y > 14.5` |
| shape residual | 0.82–0.84, invariant | FBP / SIRT / MLEM ± support, all variants |
| simulated artifact ceiling | 0.239 (full stack) vs real 0.094 / 0.211 (good), 0.476 / 0.896 (bad) | out-of-mask energy on saved reconstructions |
| IPF row vs column | rgb std 0.000/0.0015/0.0009 vs 0.257/0.348/0.318 | within one grain |

---

## 8. The NMC811 cathode campaign — validation, nulls, and the LF/PF detection budget

**Station** 1-ID scanning. **Specimen** NMC811 cathode, four charge states
(two discharged, two delithiated), 13–19 translations per layer, 32 banked
layers. **Code** `midas_pipeline` (c-omp index + refine). **Covered** the
question no earlier campaign asked: *is the per-voxel map real?* — plus the
first quantitative comparison of scanning against a single line-focus far-field
exposure of the same layer.

This campaign contributed [`phase-7-validation.md`](phase-7-validation.md).

## 8.1 What this campaign established

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | **The PF per-voxel path passes an ω-shuffle null at campaign thresholds** — no null voxel exceeded completeness 0.6957 (sparse) / 0.8333 (dense) against real medians 0.9231 / 0.8943 | VERIFIED | §8.4 |
| 2 | The **chance ceiling** is measurable, is **NOT predictable from spot density**, and the shipped `MinMatchesToAcceptFrac 0.5` sits at or below it on every layer tested | VERIFIED | §8.4 |
| 3 | **merged-FF carries no grain-count information on scanning data** — the null beat the real arm on every statistic | VERIFIED | §8.5 |
| 4 | A genuine **line-focus far-field** run of the same layer passes cleanly (786 solutions real, **0** null) | VERIFIED | §8.5 |
| 5 | **LF = Σ(13 PF frames) × 0.164 exactly** — superposition holds; the line focus delivers 1/6.1 the flux density | VERIFIED | §8.6 |
| 6 | The c-omp indexer is **deterministic across thread count and shard count** | VERIFIED | §8.3 |
| 7 | `BeamSize += 0.1` in the C parser silently widens the beam gate on any hand-run without `ScanPosTol` | FIXED (documented) | §8.2 |
| 8 | "midas_index 0.7.9 changed indexing results" | **RETRACTED** | §8.7 |

## 8.2 Defects found

- **`BeamSize += 0.1` before the gate fallback.** `IndexerUnified.c:2627` adds
  0.1 µm to the parsed `BeamSize`; the gate is
  `scanTol = (ScanPosTol > 0) ? ScanPosTol : (BeamSize/2)` at lines 1006
  (**matching**) and 3447 (seeding). The pipeline computes `scan_pos_tol_um` in
  Python from the true value and writes `ScanPosTol`, so pipeline runs are
  correct; a hand-run without it gets **0.80 µm instead of 0.75** at
  `BeamSize 1.5`. Effect: **+14.7 % accepted solutions**, per-voxel winner changed
  in **10.5 %** of voxels (15/143), misorientation p90 25.5°, max 96.4°.
  Confirmed by exact reproduction: adding `ScanPosTol 0.750000;` reproduced the
  banked file voxel for voxel (30 216 vs 30 216 over voxels 0–12; 36 441 without).
- **`paramstest_comp.txt` is written by two stages under one filename.**
  `Indexer._emit_c_omp_paramstest` (indexing, carries `ScanPosTol`) and
  `stages/_comp_params.comp_backend_paramstest` (refinement, does not). Refinement
  runs later and overwrites. **The file on disk is not the file the indexer read.**
  This is what made the above take a full session to find.
- **`RingNumber == 0` placeholder rows.** Failed transforms are written as
  all-zero rows rather than dropped — **235 334 of 1 170 954 (20.1 %)** on this
  campaign. Counting them fabricated a "20.2 % collapsed on merge" against a real
  **0.09 %**.
- **`argv[4]` is ignored in PF mode** (`IndexerUnified.c:3200` prints "argv ignored
  for PF"); `nVoxels = numScans²`. A voxel-limited test run silently processes the
  whole layer. Use `blockNr`/`nBlocks`.
- **LF/far-field peakfit percolation.** At PF thresholds on a line-focus frame the
  peak finder produced **4 384 588 regions capped at 400 peaks each** and ran 17 h
  without finishing. `midas-ring-thresh` criterion C (peak resolvability) is the
  binding one there, not SNR.

## 8.3 Method findings

- **The ω-shuffle null** (permute ω within `(ring, scan)`) is the validation this
  doc set lacked. Per-scan is load-bearing: shuffling across scans changes the
  beam-gate statistics and confounds the test. Full procedure in phase 7.
- **Determinism, verified on real dense PF data.** 30 vs 60 threads, and 1 vs 13
  shards, produced **byte-identical** `Output/*.bin`. Worth knowing independently
  of the investigation that produced it.
- **IA separates where completeness saturates** — real 0.1393 vs null 0.3669
  (2.6×). But it did **not** separate in merged-FF (null was *better*, 0.2896 vs
  0.3243), so it is a discriminator because the gate keeps the search sparse, not
  intrinsically.
- **Spatial coherence is a threshold-free screen.** distinct winners ÷ voxels:
  real 0.41, null 0.97. Across all 32 banked layers: 0.035–0.490, median 0.305 —
  none approached the null.
- **Uncapped orientation clustering matters.** A 15 000-solution subsample cap
  turned 30 601 distinct orientations into "≥ 4 604" and the cap binds invisibly —
  the completeness-cut rows barely move, which reads as stability.

## 8.4 Scientific findings — the chance ceiling

Both arms re-run fresh with one binary, production `per_voxel_cluster` winners:

| layer | spots | real vox / null vox | real best-comp median | **ceiling (null max)** | null p99 | vox at/below ceiling |
|---|---|---|---|---|---|---|
| s4/L9 | 22 848 | 49 / **0** | 0.6481 | **none — null found NOTHING** | — | **0 %** |
| s4/L4 | 418 783 | 80 / 10 | 0.9423 | **0.5333** | 0.5328 | **0.0 %** |
| s5/L5 | 935 620 | 143 / 73 | 0.9231 | **0.6957** | — | 24.5 % |
| s1/L3 | 1 294 542 | 354 / 139 | 0.8943 | **0.8333** | 0.7731 | 39.8 % |
| s2/L5 | 1 391 165 | 169 / 115 | 0.9600 | **0.7500** | 0.7406 | 14.2 % |

All five with both arms at the correct 0.75 µm gate, banked real arm against a
re-run null. At the shipped gate 0.50 the null/real retention ratio ran 0.00
(s4/L9) to 0.68 (s2/L5).

⚠ **The ceiling is NOT predictable from density.** A four-layer version of this
table looked cleanly monotonic in spot count and an explicit prediction was made
that s2/L5 — the densest, and the highest spots-per-voxel — would top it. It came
in *below* s1/L3 on both the ceiling and the contaminated fraction. Neither total
spots nor spots-per-voxel orders the table. Density is the mechanism (a sparse
enough list has no chance floor at all) but **not a substitute for measuring**.

⚠ **An empty null is a real outcome.** s4/L9's shuffled list produced zero
accepted solutions in any voxel. The first version of these scripts crashed on
the empty array instead of reporting it.

**Both arms of the sparse row ran at the same 0.75 µm gate** — the banked real arm
(which the pipeline had already run correctly) against a null re-run with
`ScanPosTol 0.750000;`. The first pass had both arms at the 0.80 µm fallback and
gave 0.7083; tightening to the true gate moved the ceiling **down** to 0.6957 and
the null's solution count from 365 067 to 208 594, exactly as a narrower gate
should. The dense row has not yet been re-run.

**The dense layers are the more contaminated ones** — the opposite of the
intuition that more spots means a better map.

## 8.5 Scientific findings — merged-FF and line-focus FF

merged-FF, 935 k-spot merged list, 10 000 matched seeds:

| | seeds w/ solution | completeness med | IA med | distinct @1° |
|---|---|---|---|---|
| real | 97.1 % | 1.0000 | 0.3243 | 6 086 |
| **null** | **97.5 %** | **1.0000** | **0.2896** | **7 652** |

Mechanism, and it is structural: merged-FF writes a **1-row `positions.csv`**, so
`nScans_ == 1`, `doScanFilter` is 0 (`IndexerUnified.c:1005`) and the beam gate is
off **in the matching loop**. Collapsing the scans deletes the `scannrobs` column
that makes the search well-posed.

Raising thresholds to LF-equivalent restores information (79.5 % vs 3.2 %; 3 741
vs 564 distinct) — but discards 92 % of the spots. **Thresholding is lossy; the
scan gate is information-preserving.** That is the argument for scanning, stated
precisely.

Line-focus far-field on the same layer, same null: **786 solutions real, 0 null.**

## 8.6 Scientific findings — the LF/PF detection budget

Raw pixels, same ω, same layer: **the LF frame is the sum of the 13 PF frames
× 0.164**, the same constant across total counts (0.1651 / 0.1638), whole-frame
median (0.1641), max pixel (0.163–0.179) and every ring-band background
(0.155–0.168), at two ω. A constant ratio across signal *and* background is
superposition; the line focus delivers **1/6.1 the flux density**.

Detection budget: signal ×0.164, background ×2.1 versus a single PF frame ⇒
**signal-to-background 12.8× worse**. A peak must reach `LF_threshold/0.164` in
the PF frame — 457 counts on a ring where PF used 50. In-band pixels above that:
**6.0 % survive, 94 % lost**. Independent corroboration: LF found 31 690 spots
against PF's 935 620 (3.4 %).

Exposure, threshold setting and filter transmission were **identical** in both
(read from the TIFF `ImageDescription`), so the counts are directly comparable.

## 8.7 Claims RETRACTED

- **"midas_index 0.7.9 changed indexing results vs 0.7.8."** FALSE. Build-and-
  bisect from canonical: v0.7.8, `77a82403`, `0dddb4c9` and v0.7.9 all produced
  **byte-identical** output (`1025d15ffe513a7d`). A full 169-voxel rebuild of
  v0.7.8 also matched 0.7.9. The real cause was the missing `ScanPosTol` in my own
  hand-run (§8.2). Both commits' "bit-identical at default" claims are **upheld**.
- **"The banked maps are version-dependent; 10.5 % of voxels get a different
  grain."** FALSE as stated. That comparison was correct-vs-wrong-gate, not
  version-vs-version. **The banked campaign results are the correct ones.**
- **"LF spots are ~2.4× brighter than their PF counterparts."** Withdrawn — the
  spot-matching route had its chance-match null sitting on top of the signal at
  every tolerance tried, and the figure contradicts the raw-pixel 0.164 by ~15×.
- **"20.2 % of spots collapsed on merge."** Withdrawn — that compared total rows
  against ring-filtered spots. The real collapse is **0.09 %**.
- **"A 26 px LF ring mismatch means LF needs its own calibration."** Withdrawn.
  LF and PF share the detector and distance; the mismatch was a stale `hkls.csv`
  built from the wrong sample's cell.

## 8.8 Hedges that must NOT be upgraded

- **The DENSE ceiling (0.8333) is still the 0.80 µm-gate figure** and is an
  **over-estimate**; a gate set from it is conservative rather than wrong. The
  sparse one has been re-measured at the correct 0.75 µm gate and moved
  0.7083 → **0.6957**; the dense re-run was still going when this was written.
  Always state which gate a ceiling was measured at.
- **The ceiling is per layer.** Two layers is not a law. It rose with density on
  both; do not extrapolate the trend.
- **Grain counts from these maps are not a census.** `OneSolPerVox` maps the
  *largest* grains. One edge layer gave 284 voxels → 10 distinct orientations
  (≈42 µm/grain) against a ~0.29 µm primary-particle size.
- **The spatial-coherence ratio is a screen, not a verdict** — a genuinely
  coarse-grained layer also sits low.
- **The binary that produced the 32 banked layers no longer exists** (both envs
  rebuilt), and the run logs record the paramstest path but never the binary's
  version or hash. The layers are reproducible *in principle* — nothing about them
  is now known to be wrong — but that provenance was not captured.

## 8.9 Measurement ledger

| quantity | value | file + command |
|---|---|---|
| sparse layer, real (banked, 0.75 gate) | 143/169 vox, 1 322 548 sol, best-comp median 0.9231 | `pf_arms_headline.py <banked layer> null_tol` |
| sparse layer, null (0.75 gate) | 73 vox, 208 594 sol, **max best-comp 0.6957** | same |
| sparse layer, both arms at the 0.80 fallback | real 144 vox / 1 516 937 sol; null 77 vox / 365 067 sol, max 0.7083 | first pass, superseded — kept to show the gate's size of effect |
| dense layer, real | 354/361 vox, 3 905 100 sol, median 0.8943 | same, s1/L3 |
| dense layer, null | 148 vox, 1 945 401 sol, **max 0.8333** | same |
| retention at gate 0.50 | 144/77 (sparse), 354/148 (dense) | `pf_chance_ceiling.py` |
| IA, real vs null | 0.1393 vs 0.3669 | `compare_pf_arms.py`, production `per_voxel_cluster` |
| distinct winners ÷ vox | 0.41 real, 0.97 null; 32 layers 0.035–0.490 | `layer_coherence_qc.py` |
| merged-FF null | 97.5 % vs 97.1 % seeds; 7 652 vs 6 086 distinct | `mff_null_shuffle.py` + `compare_null_real.py` |
| LF far-field null | 786 solutions real, **0** null | same, 3 518 matched seeds |
| LF = ΣPF × | 0.164 (5 independent quantities, 2 ω) | `superposition_raw_frames.py` |
| PF signal surviving at LF sensitivity | 6.0 % (per-ring 2.0–7.7 %) | `lf_equivalent_loss.py` |
| `ScanPosTol` effect | 30 216 (0.75) vs 36 441 (fallback 0.80) over voxels 0–12 | direct binary, one variable |
| bisect 0.7.8→0.7.9 | 4 refs, all `1025d15ffe513a7d` | `bisect_indexer2.sh` out of canonical |
| determinism | 30 vs 60 threads, 1 vs 13 shards: byte-identical | md5 of all four `Output/*.bin` |

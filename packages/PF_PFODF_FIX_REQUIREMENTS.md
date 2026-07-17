# PF + FF pipeline / pf-odf / grain-odf fix requirements (3 campaigns, 2026-07-14→16)

Requirements inventory from three concurrent campaigns: **SOH wt316** σ/γ pf-HEDM
(2 loads × 2 phases, copland recon + alleppey pf-odf), **Ni AM Layer-3**
(sangid_nov25, toro/s20hedm, 135M spots), and **emerson_oct25 Nate_crack1 FF**
(copland/alleppey, scan 3580 error diagnosis). Items marked
**[APPLIED-uncommitted]** are implemented in the laptop repo and deployed to the
named workspace; everything else is to-do. Each item: symptom → requirement →
anchors.

⚠️ **FF blast radius**: several PF fixes touch code shared with FF
(paramstest writer, fit_setup, bin_data, InputAllExtra schema). The emerson
section at the bottom carries a per-item FF-safety review — read it before
implementing any P0/N item. Rule of thumb enforced throughout: **rename-only
header fixes are safe; column-count or column-order changes are not** (all
in-tree consumers are positional; C binaries mmap fixed strides).

Companion context: `midas_pf_odf/dev/PF_HEDM_NEW_DATASET_HANDOFF.md` (needs the
§Corrections below), memory notes `project_pfodf_rawframe_geometry_bugs`,
`project_pipeline_completeness_half_legacy`,
`project_emerson_crack_recon_investigation`, `project_emerson_crack_followup`.

---

## P0 — silent-corruption bugs (fix before anyone else runs this)

### 1. paramstest writer drops `tx` (and truncates distortion + omega)
- **Symptom:** promoted `paramstest.txt` carries `LsdFit/tyFit/tzFit` but **no `txFit`**;
  distortion truncated to `p0..p3` (of 11+); no usable `OmegaStep`. Any raw-frame
  consumer (`midas_pf_odf`, `midas_grain_odf`) builds geometry with `tx=0` →
  ~0.27° in-plane rotation → **~3–4×10³ µε fake strain** (measured: −6 px tangential
  offset on all rings; GN calibration recovered tx≈+0.27 ≈ |master tx=−0.2737|).
- **Requirement:** the paramstest writer must emit `txFit`, **all** distortion
  coefficients (`p0..p14` or v2 names), and `OmegaStart`/`OmegaStep`.
- **Anchors:** `midas_transforms/midas_transforms/params.py`
  - writer: `:326-331` (writes LsdFit/YBCFit/ZBCFit/tyFit/tzFit/p0 — no tx, p0 only)
  - source of truncation: `:566-580` ("FitSetupZarr writes p0..p3 only"; `pt.tyFit/tzFit`
    set, no `pt.txFit`)
  - field list: `:136-137`

### 2. PF pipeline silently no-ops when `LayerNr_N/positions.csv` is missing
- **Symptom:** every early stage (zip_convert/hkl/peakfit/transforms) logs
  `scan discovery failed (…missing positions.csv…); skip.` and the run **exits 0**
  having done nothing. `ScanGeometry.pf_uniform` computes positions in-memory but
  nothing persists them before the stages run (only `binning`/`merge_scans` write it,
  which is too late). Handoff doc §6 claim "run regenerates positions.csv" is false
  for midas-pipeline 0.5.1.
- **Requirement:** (a) materialize `positions.csv` into the layer dir at layer setup
  from `ScanGeometry` (file order = acquisition order, sign per `--scan-step`);
  (b) in PF mode, missing positions must be a **hard error**, not a soft skip.
  **FF guard:** scope the hard error to `scan_mode == "pf"` — FF legitimately
  starts without positions.csv (transforms dump writes the 1-row `0.000000`
  itself, `midas_transforms/pipeline.py:214-217`); an unconditional check would
  brick every FF run.
- **Anchors:** `midas_pipeline/midas_pipeline/`
  - `_pf_scans.py:133-150` (`_positions_for_layer` raises; callers soft-skip)
  - `pipeline.py:196-199` (layer dir creation — insertion point for materialization)
  - `config.py:133-158` (`ScanGeometry.pf_uniform` — the in-memory positions)
  - `stages/merge_scans.py:528-530`, `stages/binning.py:173,194` (late writers)

### 3. omega step mis-derived from multi-`OmegaRange` paramstest
- **Symptom:** `build_model_from_paramstest` inferred `omega_step=0.0514` (≈74/1440)
  instead of 0.25 because paramstest lacks `OmegaStep` and the inference used
  shadow-gapped `OmegaRange` spans (3 ranges: −180/−106, −76/74, 105/180). All
  forward anchors landed nowhere (0 patches with signal) until overridden.
- **Requirement:** (a) covered upstream by P0-1 (emit OmegaStart/OmegaStep);
  (b) defensively, `geometry_from_paramstest` must not infer step from range spans
  when multiple `OmegaRange` lines exist — prefer explicit keys, else error loudly.
- **Anchors:** `midas_pf_odf/midas_pf_odf/io.py` (`geometry_from_paramstest` /
  `build_model_from_paramstest` `:159-222`); correct value for this data:
  omega_start=−180, omega_step=+0.25, n_frames=1440.

---

## P1 — pf-odf usability (works out-of-the-box instead of driver surgery)

### 4. Plumb detector distortion into `build_model_from_paramstest`
- `midas_diffract.forward` has the gated ideal→raw distortion **designed for pf_odf**
  (`HEDMGeometry.apply_distortion/p_distortion/rho_d`, forward.py:108-110, 360-380,
  1249-1280) but `midas_pf_odf.io` never populates it (zero distortion references).
- **Requirement:** read p-coeffs + RhoD from params, map via
  `midas_distortion.v2_coeffs_from_named` (**paramstest p0..p14 are v1-ordered**:
  p3 is `phi4` (a phase, degrees), NOT an amplitude — naive v2-order mapping explodes
  to meter-scale shifts; core.py:126-180 has the authoritative map), set
  `apply_distortion=True` (opt-in kwarg). Measured effect here: 4–166 µm ring shifts,
  ~100 µε on Stage-1 strain.

### 5. Ring-set override for `build_model_from_paramstest`
- Model hardwired to paramstest `RingNumbers` (io.py:186-188); hkls.csv already
  contains all rings. **Requirement:** `ring_numbers: Sequence[int] | None = None`
  kwarg (None = paramstest behaviour). Needed because indexing rings ≠ optimal
  strain rings (SOH matrix: indexing used 3,4; rings 1,2 are 3–4× brighter and
  fully on-detector; 5,7,8 are corner-clipped).

### 6. Productize the raw-frame geometry mini-calibration
- The empirical-Jacobian damped Gauss-Newton calibration (bump y_BC/z_BC/tx/ty/tz/
  Lsd/ω₀ in the actual model → convention-proof derivatives → robust LSQ, 3 iters)
  collapsed residuals **7.06 → 1.4 px RMS** and cross-validated between loads
  (tx/BC/tz agree; ty/Lsd/ω₀ scatter with 2-ring data). Any raw-frame consumer
  needs this. **Requirement:** port `mini_calib.py` into `midas_pf_odf` as e.g.
  `calibrate_raw_frame_geometry(ds, cache, params…) -> calibrated HEDMGeometry`,
  with tests. Working script: alleppey `/home/s20a/sharma_analysis/mini_calib.py`;
  calib JSONs: `/scratch/s1iduser/soh_pfodf_out/geom_calib_{140N,255N}_matrix.json`.

---

## P2 — research-design changes (need design sign-off, not just code)

### 7. Saturation-aware peakshape loss  ← current pf-odf blocker
- SOH Varex: `UpperBoundThreshold 64000`; ring 1: **96%**, ring 2: **98%**, ring 3:
  64%, ring 4: 50% of patches saturated (≥60 k). Flat-top blobs vs narrow Gaussian
  splats → structural MSE floor (loss stuck ~6.9e6, −24% over 400 steps) → **runaway
  strain** (ε grows 4237→5959 µε with more steps; any reported ε is an optimizer
  artifact). **Requirement:** per-pixel saturation mask (weight 0 where measured ≥
  threshold) carried through `assemble_grain_patch_data` → the loss; likely also
  per-voxel intensity weights. Design call = author's.

### 8. Sparse-grain smoothness regulariser
- `lambda_smooth>0` crashes on real grains: dense `grid_shape` reshape fails because
  grain voxels are a sparse subset (`shape [41,41,3] invalid for size 3936`) — the
  borbely driver comments this and sets 0. **Requirement:** scatter-based neighbour
  smoothness on `grid_ij` (no dense reshape).

### 9. Stage-2→Stage-3 spread hand-off shape mismatch
- `GrainPeakFitResult.spread_fit` returns per-voxel (G,) but `spread_init` demands
  per-region (`n_spread`) → `reshape('[16]') invalid for size 298`
  (`inversion.py:492`). **Requirement:** accept per-voxel spread_init and pool by
  `spread_region_map`, or also return per-region values.

### 10. Completeness convention ≈ 0.5× legacy — root-cause
- Same data/param: new-pipeline mean completeness 0.45 vs legacy 0.81; orientations
  identical (0.30° median, 97.7%<5° incl. the "missing" voxels). A `Completeness 0.4`
  gate silently drops ~42% of voxels legacy keeps (cosmetic "dead wedge").
  Suspect factor-2 in nExpected (Friedel counting?). **Requirement:** root-cause in
  `midas_index`/find_grains; either fix the convention or document that
  new-pipeline 0.2 ≈ legacy 0.4 and rescale param guidance.

---

## Doc corrections — `midas_pf_odf/dev/PF_HEDM_NEW_DATASET_HANDOFF.md`

1. **§6 positions.csv**: NOT regenerated by the run path in midas-pipeline 0.5.1 —
   must be pre-seeded at `<result>/LayerNr_1/positions.csv` (+ root), else silent
   all-stage skip (see P0-2).
2. **§7 `--refine-backend c-omp`** contradicts §9/§10: pf-odf reads only
   `Results/Result_OrientPos_voxel_<v>.csv` (Python refiner naming); c-omp writes
   `FitBest_<vox>_<sp>.csv`. For pf-odf runs use `--indexer-backend c-omp
   --refine-backend python` (same 43-col schema; a rename adapter is the alternative).
3. **§7 `--only` allowlist is incomplete**: real 0.5.1 PF order inserts
   merge_overlaps, calc_radius, cross_det_merge, global_powder, merge_scans, seeding,
   refinement (before find_grains). Use `--skip` on the 9 tomo/vmap tail stages
   (voxel_cleanup…refine_vmap) instead of an allowlist.
4. **§1 activate path** is host-specific (`/APSshare/miniconda/x86_64/...` absent on
   copland; use the user-owned miniconda's `etc/profile.d/conda.sh`).
5. **§5 zip ratio**: observed ~0.40× raw (23.9 GB h5 → 9.5 GB zip), not "~half".
6. **§11**: on shared hosts scope liveness checks `pgrep -u <user> -f …` (matches
   other users' watchers otherwise).
7. Add: legacy `microstructure_pf.h5` recon is **not** pf-odf input (format mismatch)
   — a new-pipeline recon is required.
8. Add: completeness ≈ 0.5× legacy note (P2-10).

---

## Evidence / ablation record (140N_matrix grain 0, Stage-1 mean |ε|)

| config | mean \|ε\| | note |
|---|---|---|
| rings 3,4 · ideal model | 4022 µε | baseline artifact |
| rings 1,2,3,4,6 · ideal | 4890 µε | more rings amplified systematic |
| + distortion (v1→v2 mapped) | 4767 µε | distortion ≈ 100 µε |
| + GN geometry calibration | 4237 µε | geometry fixed (residuals 7→1.4 px) |
| σ_yz=4 width-matched splat | 5235 µε | width not the driver |
| centroid-only fit | diverged (42 546) | unusable as configured |
| 400 steps (convergence probe) | 5959 µε, loss floor ~6.9e6 | **runaway vs misfit floor → P2-7** |

Cross-checks: R_fit moves (0.32° median — aa-at-zero bug not triggered);
corr(|ε|, misorientation)=+0.02; independent 255N calibration agrees on tx/BC/tz.

## Assets (for whoever picks this up)

- Recon (validated): copland `/home/s20a/sharma_analysis/{255N,140N}_{matrix,sigma}/LayerNr_1/`
- pf-odf driver/diagnostics: `/home/s20a/sharma_analysis/{pfodf_driver,mini_calib,ring_check,residual_diag,width_test,strain_maps}.py`
- Patch caches + calib JSONs + fit npz: alleppey `/scratch/s1iduser/soh_pfodf_out/`
- Env: `soh_pfodf` (`/home/beams/S20IDUSER/opt/miniconda3/envs`), CUDA torch
  2.13.0+cu126; workspace `/home/beams/S20IDUSER/opt/MIDAS_pfodf` @ ce53ef1 +
  local-only midas_pf_odf/midas_grain_odf.
- Deliverable figures: laptop `~/Desktop/soh_pf_teams/` + copland
  `/home/s20a/sharma_analysis/validation/`.
- Lattice-strain result (pre-pf-odf, shareable): `strain_maps.py` output — matrix
  banding at 140N, σ a/c stress dipole, hydro −116→−491 µε with load.

---
---

# Additions — Ni AM Layer-3 run (sangid_nov25, toro/s20hedm, 2026-07-14→16)

Same format. Stress case: 259 scans × ~515k spots/scan ≈ **135M spots** (~270×
a typical layer) — every memory/throughput assumption broke. ⚠️ Unlike the SOH
list above, items marked **[APPLIED-uncommitted]** are already implemented:
present in the laptop repo (uncommitted) AND deployed to the s20hedm workspace
`/home/beams/S20HEDM/opt/MIDAS_pfodf` (the live Ni run uses them). Validation:
22 bin_data tests (incl. parity-vs-SaveBinDataScanning C fixtures) at default +
forced multi-chunk on CPU + MPS, plus a 55M-pair direct chunked-vs-unchunked
bit-equality check. Revert = restore the three files below.

## P0 additions — silent-corruption class

### N1. `INPUTALL_EXTRA_HEADER` mislabels cols 13–17 **[APPLIED-uncommitted]**
- **Symptom:** header names cols 13–17 `IntegratedIntensity RawSumIntensity
  FitRMSE maskTouched FitErrCode`, but `fit_setup/core.py:396-410` writes
  col 13 = **det-corrected Omega**, 14 = IntegratedIntensity, 15 = RawSumIntensity,
  16 = maskTouched, 17 = FitRMSE. Any name-based reader is silently one column
  off — on this run it produced a complete false diagnosis ("all strong-spot
  fits failing"; the "intensities" being read were omega angles in ±180°).
  Data layout itself is legacy-positional and self-consistent (all pipeline
  consumers are positional).
- **Requirement:** header text must match the data (done); longer-term, name-based
  accessors in one place instead of magic column indices.
- **Anchors:** `midas_transforms/midas_transforms/io/csv.py:120-131` (header);
  `fit_setup/core.py:396-410` (assembly, the ground truth).
- **Emerson corroboration + EXTENSIONS (see E2 below):** hit independently on FF
  (cost half a day: "IntInt" reads were omegas). Three gaps the applied fix does
  NOT yet cover: (a) cols 11/12 are still named `YOrigNoWedge/ZOrigNoWedge` but
  hold **raw pixel** YCen/ZCen (verified == Result_StartNr YCen/ZCen exactly);
  (b) the **deployed wheels still write the old header** — copland s20iduser
  midas-transforms 0.7.2 emits a 19-name variant ending
  `…FitRMSE maskTouched FitErrCode DetID`, misaligned from col 13, and emerson
  data on disk has 19 cols (incl. a real DetID); the repo header has 18 names —
  reconcile the DetID column against `fit_setup/core.py` before bumping, then
  redeploy everywhere; (c) `midas_pipeline/stages/transforms.py:204-205` looks up
  `"YOrig(NoWedgeCorr)"` — a name **no header variant ever contained** → the PF
  per-scan y-offset on that column is a silent no-op (must be fixed in the SAME
  commit as any header rename, and fail loud on missing columns).

### N2. Per-peak `returnCode` dropped at merge
- **Symptom:** peakfit's 29-col rows carry `returnCode` (col 18,
  `midas_peakfit/postfit.py:85`) but the 17-col merge output drops it — fit
  success/failure is unrecoverable downstream (related: N1's phantom
  `FitErrCode` column name).
- **Requirement:** propagate `returnCode` through `merge/core.py` → fit_setup →
  csv (and then the `FitErrCode` header name becomes honest).
- **Anchors:** `midas_transforms/merge/core.py:75-80` (17-col schema comment).
- **⚠️ Coordinate with E4 (OrigSpotID) as ONE schema change:** both add columns to
  the same files. Append-at-END only, never insert; bump `INPUTALL_EXTRA_NCOLS`
  and every reader with a col-count assert in the same commit; leave
  `Spots.bin`/`ExtraInfo.bin` binary strides untouched (C consumers mmap them).

### N3. Binning pair expansion is unbounded — OOM at scale **[APPLIED-uncommitted]**
- **Symptom:** `_bin_assignment` materializes ~10 pair-length **int64** tensors
  at once (dtype flag irrelevant). At ~8e9 (spot × eta × ome-bin) pairs: single
  60.4 GiB tensor → CUDA OOM (A6000 48 GB); CPU float64 AND float32 both
  **kernel OOM-killed** at 250 GB (peak observed 232 GB even after filtering to
  3.2e9 pairs).
- **Requirement (implemented, three layers, bit-identical):**
  (a) spot-chunked expansion in `_bin_assignment` (`MIDAS_BIN_PAIR_CHUNK` env,
  default 2^28 pairs; boundaries computed on CPU; MPS forced single-chunk —
  multi-chunk int64 ops segfault on MPS only);
  (b) memory hygiene in `_bin_to_data_ndata_scanning` (callee takes a list and
  clears it = sole ownership so `del`s work; in-place composite sort key with
  divmod recovery; all-true-mask fast path) → ~5 live pair tensors, not ~10;
  (c) **per-ring pack + concat** at the scanning call site (Data.bin is
  ring-major ⇒ bit-identical by construction; peak ÷ n_active_rings; ring
  selected by zeroing other radii, reusing the keep filter).
  Remaining: expose the chunk budget as a real param; root-cause the MPS
  segfault if Mac matters.
- **Anchors:** `midas_transforms/bin_data/core.py::_bin_assignment` (~:223-280),
  `bin_data/voxel_binner.py::_bin_to_data_ndata_scanning` (:151-) and the
  scanning call site (~:444-490).
- **⚠️ FF blast radius:** FF uses the SAME scanning writer path
  (`midas_transforms/pipeline.py:203-212`, "FF uses scan_nr=0") — this patch is
  live in the FF code path, and the existing parity fixtures are
  scanning-flavoured. **Before committing, add/execute an FF-mode bit-parity
  fixture** (Data.bin/nData.bin byte-equality on an FF reference layer); FF
  binning has regressed silently before (pipeline 0.2.0 wrote zero-byte
  Data.bin). Note: emerson recon_3580_003 ran on the *unpatched* 0.7.2 wheel,
  so it does not validate this patch on FF.

### N4. `--only` allowlist silently produces a broken recon (code guard, not just docs)
- Doc-correction 3 above covers usage; the **package requirement** is a
  dependency check: error (or loud warning) when an `--only` set omits stages
  the requested ones need (merge_scans/seeding/refinement/…). Both campaigns
  independently hit this from the same handoff doc.
- **Anchors:** `midas_pipeline/pipeline.py:239-260` (stage loop / skip logic).

## P1 additions — throughput & operability (dense data = every stage serial-bound)

### N5. peakfit stage: sequential per-scan; `shard_gpus` unused by it
- 59 min/scan CPU (1 core effective) vs **120 s/scan** on one A6000; second GPU
  idle; `_run_pf` is a plain loop (docstring: parsl fan-out "a follow-up").
  External 2-worker helper (1/GPU) gave a true 2× — but two independent runners
  race (both pick lowest-undone scan; outputs identical so benign, still wrong).
- **Requirement:** per-scan fan-out across GPUs/processes with per-scan claim
  (lockfile/rename) inside the stage. Anchors: `stages/peakfit.py:66-120`.

### N6. transforms + zip_convert stages: sequential per-scan
- transforms ~49 s/scan (3.5 h here), zip_convert ~2–3 min/scan (~13 h);
  scans independent. Same fan-out requirement (I/O-aware worker count for
  zip_convert). Anchors: `stages/transforms.py` loop; `stages/zip_convert.py:57-90`.

### N7. binning has no per-stage device/dtype override
- Binning inherits global `--device`; on GPU it OOMs long before anything else
  (N3) — this run had to move the whole tail to `--device cpu --dtype float32`.
- **Requirement:** per-stage device fallback (binning is a natural CPU stage) or
  a `--binning-device` knob. Anchors: `stages/binning.py:87,170`.

### N8. Intensity spot-filter as a first-class parameter
- fit_setup filters are MinEta/OmegaRanges/BoxSizes/RingToIndex only — nothing
  intensity-based. On noise-dominated data (60% of 515k spots/scan at
  IntegratedIntensity < 200 counts) the run had to post-filter layer CSVs by
  hand (awk `$15>=200`, originals kept as `*.unfiltered`) — unrecorded in
  paramstest, invisible to reruns.
- **Requirement:** `MinIntegratedIntensity` (global or per-ring) applied in
  fit_setup and written to paramstest. Verified safe: binning renumbers SpotIDs
  1..N itself and writes its own SpotsToIndex.csv, so filtered CSVs stay
  self-consistent. Anchors: `fit_setup/core.py` docstring §6 (filter list).
  **FF note:** fit_setup is shared FF/PF — default must be 0/off (no behaviour
  change for FF), key written to paramstest only when set, and added to the
  known-keys ignore lists (see E7) so downstream parsers don't warn.

### N9. `midas_zipper` missing `tqdm` dependency
- Fresh env → `ff_zip.py:488 from tqdm import tqdm` → every scan exits 1;
  zip_convert marches on WARNING-only ("0 built + N failed") and the run "succeeds".
- **Requirement:** add tqdm to midas_zipper pyproject deps; arguably also make
  zip_convert fail hard when *all* scans fail. Anchor: `midas_zipper/ff_zip.py:488`.

### N10. peakfit orchestrator: thread oversubscription
- Without caller-set `OMP_NUM_THREADS=1`, `num_procs` workers × OMP/BLAS threads
  → load 28/64, frame rate collapse (22 → 4–5 f/s). Orchestrator should cap
  worker threads itself. Anchor: `midas_peakfit/orchestrator.py` (pool setup).

### N11. Separate "raw read dir" from "scan work dir" in `_pf_scans`
- `scan_dir = RawFolder/<n>` ⇒ zip_convert `mkdir`s inside RawFolder — fails on
  read-only collaborator data (`PermissionError`). Workaround: writable symlink
  farm as RawFolder (works, but every run against shared beamline data needs it).
- **Requirement:** a `WorkFolder`/`--scan-work-dir` distinct from the raw h5
  location. Anchors: `_pf_scans.py:218-236` (scan_dir candidates),
  `stages/zip_convert.py:67`.

## P2 addition

### N12. merge_overlaps is a no-op stub — cross-frame merge quality unquantified
- Stage returns `stub_run` ("cross-frame merge handled inside peakfit"); the
  legacy `MergeOverlappingPeaks` port is pending. On dense data any
  under-merging of frame-spanning spots multiplies the spot table and every
  downstream cost (and biases omega centroids). **Requirement:** port the
  bounding-box merge, or quantify peakfit's omega-tail merge against legacy on
  a reference layer. Anchor: `stages/merge_overlaps.py` (whole file).

## Corroborations of the SOH items from this run
- **P0-2 (positions.csv silent skip): confirmed independently** — and note the
  extra wrinkle that `_pf_scans._positions_for_layer` sorts ascending while
  `stages/indexing.py:147` reads the same file unsorted/file-order; unify or
  document the convention when fixing.
- Doc-corrections 2 & 3 (c-omp refine schema; `--only`): both bit this run too
  (caught before refinement stage ran; relaunched with `--refine-backend python`
  and skip-list instead of allowlist).

## Ni-run assets
- Recon (in progress): toro `/storage/Purdue_8T01/s20hedm_ni_recon/` (layer CSVs
  filtered ≥200 with `*.unfiltered` backups; `launch_cpu32_ds.sh`; per-ring
  binning deployed). Env `ni_pfodf` (`/home/beams/S20HEDM/.conda/envs`), CUDA
  torch 2.6.0+cu124; workspace `/home/beams/S20HEDM/opt/MIDAS_pfodf`.
- Patched files (uncommitted, laptop = source of truth):
  `midas_transforms/{bin_data/core.py,bin_data/voxel_binner.py,io/csv.py}`.
- Original geometry validated byte-exact vs owner's `ps_Layer3_pf_FullVolume_Rec1.txt`
  + `positions.csv` (129→0@line130→−129, step −1 µm; `--scan-step -1.0`).

---
---

# Additions — emerson_oct25 FF crack-sample investigation (copland/alleppey, 2026-07-15→16)

FF campaign: Nate_crack1 (304 SS, cracked), scan 3580, 7,912 grains, "very high
errors" diagnosis. Verdict (adversarially proven, 6 controls): DiffPos ≈ 886 µm
is **irreducible intragranular orientation spread** (~0.36° per-spot internal
angle; grain-odf peak-shape fits confirm 0.4–0.8°/grain independently) plus a
**−850 ppm reference-lattice offset** absorbed as +850 µε fake hydrostatic
strain. Full evidence + scripts:
`/gdata/dm/20ID/2025/emerson_oct25/analysis/nate_analysis/reconstructions/`
`recon_3580_001/diagnostics_investigation/` (FINDINGS.md + 13 py + grain_odf/).
Re-run `recon_3580_003` (recalibrated a₀ 3.596759 + MinMatchesToAcceptFrac 0.5):
7,889 grains, radial bias −160 → +1.5 µm, ring ppm −850 → ±110, mean hydrostatic
+850.5 → **+7.6 µε**, DiffPos unchanged (physical) — recalibration validated e2e.

## E0. midas-process-grains 0.6.0 — signed residual decomposition **[APPLIED-uncommitted]**
- **What:** `compute/residual_decomposition.py` (new): per-spot signed dY, dZ,
  radial, tangential, wrapped dOme + internal angle, collected inside the
  existing FitBest pass in `pipeline.py::_build_spot_matrix_rows` (now returns a
  tuple — single caller updated); aggregates (per-grain median/MAD, per-ring
  dR/R ppm, 30° eta bins, global scalars) + gzip float32 per-spot table written
  to `processgrains_diagnostics.h5:/residuals` (`io/consolidated.py`); log
  summary warns when |median dR/R| > 200 ppm (reference-lattice signature).
- **Validation:** 7 new tests (`tests/test_residual_decomposition.py`), full
  suite 290 pass; ran in production on recon_3580_003 (1.66M-row spot table).
- **Deployed:** copland s20iduser midas_env (pip `--no-build-isolation
  --no-deps` — that account has **no PyPI**; plain pip install bricks the env
  mid-uninstall). Laptop repo = source of truth, **uncommitted** (commit drafts
  in session log).
- **Remaining:** manuals not yet updated (`manuals/` — document `/residuals`
  schema + that DiffPos/DiffOme are now decomposable); `mode=="legacy"` path
  (no FitBest) emits empty residuals by design.

## P0 additions — FF

### E1. `RMSErrorStrain` in Grains.csv is hardwired to 0
- **Symptom:** every grain reports 0 — users read it as "perfect strain fit".
  `pipeline.py:806-808`: "We don't yet capture the per-grain solver residual in
  the strain loop; emit zeros for now."
- **Requirement:** capture the strain solver's `residual_norm`
  (`compute/strain.py:98-108` — `PerSpotStrainResult.residual_norm`, already
  computed at `:264`) per grain in the strain loop and write it into the column
  (µε, matching legacy C convention). Add a test asserting non-zero on a
  strained synthetic.
- **Anchors:** `midas_process_grains/midas_process_grains/pipeline.py:793-810`.

### E2. InputAllExtra cols 11/12 mislabeled + deployment/DetID reconcile
- Extension of N1 (details inline there). Requirements: (a) rename cols 11/12 to
  what they hold (**raw detector pixels** — suggest `YRawPx ZRawPx`); (b)
  reconcile the 18-name repo header vs the 19-col (…`DetID`) data written by
  deployed 0.7.2 against `fit_setup/core.py`; (c) version-bump midas-transforms
  and **redeploy to copland s20iduser** (offline pip flags above) — the live FF
  env still writes the misaligned header; (d) fix
  `midas_pipeline/stages/transforms.py:204-205` name lookup in the same commit +
  fail loud on missing columns (PF y-offset silent no-op).
- **FF safety:** rename-only; all in-tree consumers positional. Do NOT reorder.

### E3. SpotID spaces: Result_StartNr ≠ InputAll/SpotMatrix/FitBest
- **Symptom:** peaksearch-merge IDs and fit_setup IDs are different spaces
  (fit_setup re-sorts/renumbers); nothing documents it. Any SpotID join between
  the two silently pairs random spots — it invalidated two analyses this
  session until caught (a correct bridge exists: InputAllExtra col 14
  IntegratedIntensity == Result col 1, exact float, + Omega; 96.6% coverage).
- **Requirement:** carry the merge-space SpotID through fit_setup as an
  **appended** `OrigSpotID` column in InputAll/InputAllExtra (mapping exists at
  write time) — coordinate with N2 as ONE schema change (append-only, bump
  ncols + readers, binaries untouched). Plus loud docstrings on both writers.
- **Anchors:** `midas_transforms/fit_setup/core.py` (renumbering),
  `io/csv.py` (writers).

## P1 additions

### E4. grain-odf σ_θ/σ_ε spread DOFs diverge at default LRs on real data
- **Symptom:** with `refine_orientation_spread/refine_strain_spread=True` at
  defaults (`lr_orientation_spread=50`, `lr_strain_spread=5e-4`,
  `inversion.py:373,405`) on emerson patches: both pin at their ceilings
  (σ_θ→20 px, σ_ε→5e-2) and the correlation loss **worsens** (−25.4 → −4.0).
  DOFs postdate the park22 validation; synthetic-only tuning.
- **Requirement:** (a) warn/flag when a spread parameter sits at its clamp for
  N consecutive steps (fit is invalid); (b) LR auto-scale from the initial
  gradient magnitude, or per-dataset LR guidance; (c) report the
  **weighted-median** particle spread (+ weight-within-X°) alongside wRMS —
  wRMS scales with `theta_max` (prior-dominated tails; measured 1.75× at
  2× trust region, weighted median only 1.24×). Reference runner:
  `…/diagnostics_investigation/grain_odf/fit_odf_emerson.py`.
- **Anchors:** `midas_grain_odf/midas_grain_odf/inversion.py:337-620`,
  `odf.py::ParticleODF.sample`.

### E5. midas_zipper writes `omegaStep = 0.0` into zarr analysis_parameters
- **Symptom:** `-omegaStep` CLI default 0.0 (`ff_zip.py:879`) lands verbatim in
  `analysis/process/analysis_parameters/omegaStep` (emerson zarr says 0.0;
  actual 0.25). Pipeline ignores it, but any metadata reader (and P0-3-style
  inference) gets a wrong scan description.
- **Requirement:** derive from the data/param file when the flag is absent, or
  write NaN/omit rather than a false 0.0. Converges with P0-1/P0-3 (make
  OmegaStart/OmegaStep authoritative and consistent in paramstest AND zarr).
- **Anchors:** `midas_zipper/midas_zipper/ff_zip.py:879` (+ its attrs writer).

### E6. Known-keys lists must absorb the new paramstest keys (pairs with P0-1/N8)
- When P0-1 lands (`txFit`, full `p0..p14`, `OmegaStart`, `OmegaStep`) and N8
  (`MinIntegratedIntensity`), downstream parsers that warn on unknown keys will
  spam or confuse users. Extend the ignore/known lists:
  `midas_fit_grain/midas_fit_grain/config.py:317-327` (currently knows only
  tx/ty/tz + p0..p5), and the analogous list in `midas_process_grains/params.py`
  if present. C parsers keyword-match and skip unknowns (expected safe) — add a
  smoke test running FitUnified + midas_indexer on a paramstest carrying all
  new keys.

## FF blast-radius review of the SOH/Ni items above (implement-time checklist)

| Item | FF surface | Verdict / guard |
|---|---|---|
| P0-1 paramstest tx/p/omega | FF paramstest confirmed same truncation (emerson: no tx, p0..p3) — but FF **C chain is unaffected** (tilts baked into YLab/ZLab by DetCor upstream). | Safe & beneficial for FF raw-frame consumers. Do E6 with it. |
| P0-2 positions.csv | FF writes its own 1-row file late. | Hard-error **only** in `scan_mode=="pf"`. |
| P0-3 omega inference | pf_odf-only code. | No FF surface. |
| P1-4/5/6 pf-odf | separate package. | No FF surface (emerson corroborates the distortion scale: raw↔DetCor tens of px at p3=35.5). |
| N1/E2 header | FF consumers positional; C reads binaries. | Rename-only. Never reorder/insert. Redeploy wheels. |
| N2+E3 schema | shared CSVs. | ONE append-only change; binaries untouched; bump ncols+readers together. |
| N3 binning chunking | **FF uses the same scanning writer** (scan_nr=0). | FF bit-parity fixture REQUIRED before commit. |
| N4 --only guard | FF has a different stage list. | Per-mode dependency graph, not a shared allowlist. |
| N5/N6 fan-out | `_run_pf` loops only; `_run_ff` is a single call. | Keep shared helpers signature-stable. |
| N7 binning device knob | shared stage. | Default = inherit (no FF behaviour change). |
| N8 intensity filter | fit_setup shared. | Default off; paramstest key only when set; E6. |
| N9 tqdm dep | shared zip_convert. | Pure win for FF; also fail hard when ALL scans fail. |
| N10/N11 | PF orchestration paths. | Keep FF single-dir path untouched. |
| N12 merge_overlaps | PF stage; FF merges inside midas_transforms. | No FF surface. |

## P2 addition

### E7. Auto reference-lattice (d0) advisory in process_grains
- The new `/residuals` diagnostics already flag >200 ppm ring offsets; the fix
  is one call away (`midas_stress.equilibrium.recover_d0_cubic_free_standing` —
  emerson: eps_iso +850.6 µε, recovered a₀ 3.596759, validated e2e in
  recon_3580_003). **Requirement (design sign-off needed):** when the flag
  trips, print the recovered a₀ and the exact `LatticeConstant` line to paste —
  advisory only, never auto-apply (free-standing assumption is the user's call;
  loaded samples need `recover_d0` + stiffness + applied stress).

## Doc corrections (emerson)

1. `manuals/`: document `processgrains_diagnostics.h5:/residuals` (schema =
   `SPOT_RESIDUAL_COLS` in `compute/residual_decomposition.py`) and that
   Grains.csv `DiffPos`/`DiffOme` decompose there; note `RMSErrorStrain`
   semantics once E1 lands.
2. Document the two ID spaces (E3) prominently in both writers + manuals until
   the `OrigSpotID` column ships.
3. Convention notes for raw-frame consumers (grain_odf/pf_odf extractors), all
   verified on emerson Varex: zarr frame layout `frame[row = ZCen_px,
   col = (nPxY−1) − YCen_px]`; ideal-µm→px map `y_px = y_BC − y_µm/px`,
   `z_px = z_BC + z_µm/px`; raw vs DetCor differ by **tens of px** when
   distortion is large (p3≈35.5 here vs −0.64 on park22) — never anchor
   forward-model matching on raw px.
4. InputAllExtra empirical column map (until E2 ships), verified by exact-value
   bridge: 0 YLab(µm) 1 ZLab 2 Omega 3 GrainRadius 4 SpotID 5 Ring 6 Eta
   7 Ttheta 8 OmegaIni 9 Ycorr(µm) 10 Zcorr(µm) 11 **YrawPx** 12 **ZrawPx**
   13 **Omega(dup)** 14 **IntInt** 15 RawSum 16 maskTouched 17 FitRMSE 18 DetID.

## Evidence record (emerson)

| check | result |
|---|---|
| recalib: ring dR/R | −707…−1016 ppm → **±110 ppm** (003) |
| recalib: mean hydrostatic (vol-wtd) | +850.5 µε → **+7.6 µε** (003) |
| MinMatchesToAcceptFrac 0.2→0.5 | 7,912 → 7,889 grains (conf floor 0.63; threshold non-binding) |
| DiffPos after recalib | 886 → 889 µm (unchanged ⇒ physical) |
| internal angle vs intensity (correct join) | 0.375° weak → 0.323° top-10% (not noise) |
| internal angle vs SigmaEta (correct join) | flat 857/809/821/783 µm (not broadening) |
| Wahba optimal rotation | 0.381° → 0.369° (orientation fit already optimal) |
| grain-odf weighted-median spread | 0.4–0.8°/grain, 12 grains, loss improves 2–3× |

## Assets (emerson)

- Investigation bundle: copland `…/recon_3580_001/diagnostics_investigation/`
  (FINDINGS.md, 13 analysis scripts, patches h5 ×2, ODF JSONs, runner).
- Recons: `recon_3580_001` (original), `recon_3580_003` (recalibrated,
  validated); params `parameter_files/ps_304SS_recal_tight.txt`
  (LatticeConstant 3.596759 + MinMatchesToAcceptFrac 0.5); driver
  `run_ff_pipeline_003.py`.
- alleppey: `~s1iduser/emerson_odf/` (patches, runner, logs,
  `pkgs/` PYTHONPATH overlay of 6 midas packages — the overlay pattern for
  hosts where the env can't be upgraded).
- Deployment matrix (a new chat MUST reconcile before committing):
  laptop repo = source of truth with **uncommitted** changes in
  `midas_process_grains` (E0, at 0.6.0) and `midas_transforms`
  (N1/N3, version un-bumped); copland s20iduser runs process-grains 0.6.0
  (new) but transforms **0.7.2 wheel (old header)**; s20hedm/toro workspace
  carries the patched transforms; alleppey s1iduser uses the PYTHONPATH
  overlay, env otherwise stale.

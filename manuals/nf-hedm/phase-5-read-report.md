# Phase 5 — Read the result, and report

> Part of the **NF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 9. STEP 7 — Read the result

### 9a. The text `.mic`

Header: **three `%` lines plus one `%` column-name line** ⇒ `skip_header=4`
(`consolidate.py:47-49`). Column names are the literal string at `parse_mic.py:145-147`;
the *meanings* come from the writer's `MicRecord` (`fitorientation/output.py:37-56`).

| col | header name | what it actually holds | units |
|---|---|---|---|
| 0 | `OrientationRowNr` | row index of the winning candidate in the seed list (`best_row_nr`) | index |
| 1 | `OrientationID` | **number of phase-1 winners** carried into phase 2 (`n_winners`) — **not an ID** | count |
| 2 | `RunTime` | per-voxel fit wall time | s |
| 3 | `X` | voxel centre, sample frame | µm |
| 4 | `Y` | voxel centre, sample frame | µm |
| 5 | `TriEdgeSize` | voxel triangle size | µm |
| 6 | `UpDown` | `+1` if grid col0 ≤ col1 else `-1` (`fitorientation/io.py:355-365`, `:248-269`) | ±1 |
| 7–9 | `Eul1..3` | Bunge ZXZ | **radians** |
| 10 | `Confidence` | FracOverlap of the best solution | 0–1 |
| 11 | `PhaseNr` | copied from the `PhaseNr` key | int |

Radians confirmed by the bundled reference row `… 4.860570 0.768787 2.147169 0.114583 1`
in `NF_HEDM/Example/Au_txt_Reconstructed.mic`.

**Two things that will bite you:**

- **Rows with `Confidence == 0` are silently dropped** from the text file
  (`parse_mic.py:150-151`). The text `.mic` is *shorter* than the voxel grid. Only
  `MicFileBinary` is one-record-per-voxel. **Row *i* of the `.mic` is not voxel *i*.**
- **`%TriEdgeSize` is read off row 0 of the binary** (`parse_mic.py:140`), which may be an
  *invalid* voxel. The bundled Au reference has header `%TriEdgeSize 0.000000` while every
  data row carries `1.000000`. Downstream: `mic2grains` falls back from spatial to global
  merging when `TriEdgeSize <= 1e-6` (`mic2grains.py:365-373`) and the grain radius
  collapses to zero (`mic2grains.py:294`). **Read column 5 of a data row.**

### 9b. The companion binaries

Little-endian float64, **4-double header `[xSize, ySize, minX, minY]`**
(`parse_mic.py:1-22`).

| file | payload | units |
|---|---|---|
| `<MicFileText>.map` | 7 planes of `xSize*ySize`: Confidence, Eul1, Eul2, Eul3, OrientationRowNr, PhaseNr, distance-to-voxel-centre (`parse_mic.py:290-304`) | mixed; Eulers rad |
| `.map.kam` | 1 plane: KAM over the assigned 8-neighbours | **radians** |
| `.map.grainId` | 1 plane: connected-component grain label, BFS with edge threshold `GBAngle` | int, 1-based |
| `.map.grod` | 1 plane: misorientation to the highest-confidence pixel of the same grain | **radians** |

Unassigned pixels: `-15` in `.map` (`parse_mic.py:306`), `0` in the single-plane files.

**`.map.grainId` is the only real grain segmentation in the output set.**

### 9c. The consolidated HDF5 — and its four mislabelled datasets

Written by `generate_consolidated_hdf5` (`consolidate.py:185`) on `PipelineH5`
(`state.py:71`); arrays gzip-4 (`state.py:33`).

```
/provenance                 attrs: created, last_opened, parameter_file (full text),
                                   one attr per MIDAS package version
/pipeline_state             attrs: workflow_type, command_line_args (JSON), start,
                                   last_update, current_stage
    completed_stages/<i>    stage-name strings, in completion order
    timestamps/per_stage    attrs: stage -> ISO timestamp
/parameters/…               SpaceGroupNr, LatticeConstant, GridSize, GlobalPosition,
                            GBAngle, NumPhases, PhaseNr, nSaves (whichever were found)
/voxels/position            (N,2) X, Y um
/voxels/euler_angles        (N,3) radians
/voxels/confidence          (N,)  0-1
/voxels/{orientation_row_nr, orientation_id, tri_edge_size, up_down}   <- ALL FOUR WRONG
/voxels/phase_nr            (N,)
/grains/{grain_id, mean_euler_angles, mean_position, mean_confidence, num_voxels}  <- NOT GRAINS
/grains/strain              empty group, attrs["status"] = "reserved"
/maps/orientation           (ySize, xSize, 7)
/maps/extent                [minX, minX+xSize, minY, minY+ySize]
/maps/{kam, grod, grain_id} (ySize, xSize)
/all_matches/data           the .AllMatches text, parsed
/grid/{points, num_points}
/multi_resolution/<label>/  attrs: grid_size, pass_type; then voxels/… maps/…
```

Resolution labels: `loop_0_unseeded`, `loop_<k>_{seeded,unseeded,merged}`
(`workflows.py:348-352, 424-428, 509-514, 534-540`). `/grains/`, `/all_matches/`, `/grid/`
are written **only for the root pass** (`consolidate.py:253, 276, 283`).

**`/raw_data_ref/` does not exist here.** `packages/midas_nf_pipeline/USAGE.md:197-204`
advertises it; grepping `packages/` finds it only in
`midas_ff_pipeline/midas_ff_pipeline/stages/consolidation.py:508`.

**The four mislabelled datasets** (`consolidate.py:238-250`, repeated at `:330-340`):

| dataset name | column written | what that column actually is |
|---|---|---|
| `tri_edge_size` | 0 | `OrientationRowNr` |
| `up_down` | 1 | `OrientationID` / winner count |
| `orientation_row_nr` | 2 | `RunTime` |
| `orientation_id` | 6 | `UpDown` |
| `run_time` | 12 | column 12 does not exist (text `.mic` has 0–11) — the write is skipped |

`position` (3:5), `euler_angles` (7:10), `confidence` (10), `phase_nr` (11) are correct.
The four above are **name-shifted, not value-corrupted**: `nf_qt.py:1651-1662` reads each
dataset back into the column index it came from, so the viewer round-trips
self-consistently. Any other consumer that trusts the names reads the wrong quantity.

**`/grains/` is not grains — re-verified 2026-07-29.** `aggregate_grains`
(`consolidate.py:153-176`):

```python
valid = mic_data[:, 10] > 0          # consolidate.py:157   confidence filter
data = mic_data[valid]
gids = np.unique(data[:, 6])         # consolidate.py:161   <-- column 6
gids = gids[gids >= 0]               # consolidate.py:162
...
    mask = data[:, 6] == gid         # consolidate.py:172
```

Column 6 of the text `.mic` is `UpDown` (`parse_mic.py:145-146`: `%OrientationRowNr
OrientationID RunTime X Y TriEdgeSize **UpDown** Eul1 Eul2 Eul3 Confidence PhaseNr`),
which takes values ±1 (`fitorientation/io.py:355-365`). After the `gids >= 0` filter, `/grains/`
gets **one "grain" per distinct non-negative `UpDown` value — in practice a single row
covering every upward-pointing voxel.** `/grains/mean_euler_angles` is then the mean Euler
angle of half the map. Established by reading; **not executed against a real H5** — check
the row count before quoting this. **Use `/maps/grain_id`, or run `mic2grains`.**

### 9d. Viewer — `gui/nf_qt.py`

```bash
export MIDAS=<your MIDAS checkout>   # beamline host: ~s1iduser/opt/MIDAS_canonical
cd <DATA_FOLDER>                     # required for BeamPos auto-detect (§3e)
python "$MIDAS/gui/nf_qt.py" &       # --dark for dark theme (nf_qt.py:2169)
```

What it is for, in priority order for an agent:

- **Confirm the frame layout.** *First File* populates folder, stem, pad width and start
  frame from one chosen file (`nf_qt.py:1199-1220`). Its frame formula is flatter than the
  pipeline's: `fnr = start_frame + frame + dist * n_files_per_dist` (`nf_qt.py:1248`).
- **Tell distances apart.** Step *Distance* with *Frame* fixed; spot spread changes
  visibly. This is the practical discriminator when the log is missing.
- **Beam-edge measurement for calibration.** *Box H* / *Box V* — click two opposite
  corners; the right panel shows the integrated profile with the two threshold crossings,
  their centre and width (`nf_qt.py:1337-1361, 1490-1514`). The walkthrough is
  `manuals/NF_Calibration.md:82-100`.
- **Lab axes** (`A`) — overlays the MIDAS lab frame at the current distance's beam centre
  and warns if BC is still `(0,0)` (`nf_qt.py:1834-1861`). Convention, from tooltip/help
  (`nf_qt.py:386-391, 649-656`): `X_Lab` (= `Y_MIDAS`, red) display-left, `Y_Lab`
  (= `Z_MIDAS`, green) up, `Z_Lab` (= `X_MIDAS`, blue) into the page; η = 0 toward
  `Y_Lab`/`Z_MIDAS`; NF display origin bottom-right, FF bottom-left.
- **Overlay predicted spots.** *Load Grain* → *Make Spots* → *Select Point*
  (`nf_qt.py:2045-2116`, `1931-2015`). Shells out to the C `GetHKLList`,
  `GenSeedOrientationsFF2NFHEDM`, `SimulateDiffractionSpots` — **these must be built**, or
  it silently prints a failure and returns. Radius is scaled by `this_lsd / sim_lsd` with
  `this_lsd = Lsd + dist * dist_diff` (`nf_qt.py:1986-1992`) — hence the *distance
  difference* field.
- **Calc Median** shells out to the C `MedianImageLibTiff` (`$MIDAS_NF_BIN_DIR` or
  `~/opt/MIDAS/NF_HEDM/bin/`), one thread group per distance (`nf_qt.py:1865-1914`).
  *Max/Frames* and *Sum/Frames* read the `.bin` sidecars from §3f, not the TIFFs
  (`nf_qt.py:1256-1281`).
- **Load Mic / Load H5.** `.mic` uses `skip_header=4` and scatters; `.map` reads the
  4-double header then planes (`nf_qt.py:1518-1541`). *Load H5* enumerates
  `multi_resolution/*` into a *Resolution* combo, appending `⚠ slow` where a resolution has
  no rasterised map, defaulting to the highest `_seeded` loop that has maps
  (`nf_qt.py:1543-1610`).

**Two labelling traps in the viewer**, both inherited from §9a/§9c:

- Colour mode **`GrainID`** paints `.mic` column 0 / `.map` plane 4, which is
  `OrientationRowNr` — **not a grain label** (`nf_qt.py:1681-1682`).
- The mode that shows real grains is **`GrainMap`**, reading `maps/grain_id` or the
  `.map.grainId` sidecar (`nf_qt.py:1787-1799`). `KAM` and `GROD` come from `.map.kam` /
  `.map.grod` and are in **radians**.

Shortcuts: `←`/`→` frame, `L` log scale, `A` lab axes, `Q` quit, `Ctrl+scroll` frame
(`nf_qt.py:665-670`).

---


## 11. Validation status — put every number you report in one of these buckets

"Byte-parity port" in a docstring is an *intent*. The test file is the evidence.

### Has a real parity test against a C reference

| Component | Test | Gate |
|---|---|---|
| `parse_mic` — text `.mic` | `midas_nf_pipeline/tests/test_parse_mic.py:98-122` | header lines byte-equal; data tokens `abs=1e-6` vs `Au_txt_Reconstructed.mic` |
| `parse_mic` — `.map` | `test_parse_mic.py:137-151` | planes 0–5 **exactly equal**; plane 6 `rtol=atol=1e-12` |
| `parse_mic` — `.map.grainId` | `test_parse_mic.py:154-164` | **exactly equal** |
| `parse_mic` — `.map.kam`, `.map.grod` | `test_parse_mic.py:167-185` | `atol=rtol=1e-10` (radians) |
| `mic2grains` | `tests/test_mic2grains.py` — invokes the **live C `Mic2GrainsList`** at test time | grain **count** equal; header lines equal; each Python grain matches some C grain to **< 0.1°**. Position and radius parity **explicitly not asserted** (`test_mic2grains.py:152-156`) |
| `fit_orientation` | `midas_nf_fitorientation/tests/integration/test_vs_c_fit_orientation.py` — runs C `simulateNF` + `nf_MIDAS.py` + `FitOrientationOMP`, then re-fits in Python | **median misorientation < 0.5°** and **≥ 90 % of voxels < 0.5°**, on a **30-voxel stratified sample** of the bundled Au example (`:66-68`, `:396-409`) |

The `parse_mic` reference input is a **frozen** copy,
`NF_HEDM/Example/sim/Au_bin_Reconstructed.mic.c_ref`, because the live binary is
overwritten by every Python fit (`test_parse_mic.py:39-48`). Both integration tests **skip**
unless `MIDAS_RUN_INTEGRATION=1`; `test_mic2grains` also skips if the C binary is not built.

### Does NOT have a parity test against C

- **`midas_nf_fitorientation` end to end.** Its own README: *"the forward path is validated
  against `midas-diffract` (pixel-exact vs. the C simulators); the fit drivers have
  unit-test coverage at the module level. **End-to-end agreement against the C
  `MicFileBinary` on a real reconstruction dataset is the next milestone.**"*
  (`packages/midas_nf_fitorientation/README.md:137-142`). The 30-voxel stratified test is a
  *sample* on synthetic Au, not a dataset-level result. No C-parity tests among the 13
  module-level test files.
- **Everything in `midas_nf_preprocess`.** There is **no** `tests/integration/` directory in
  that package; tests are unit tests over synthetic data
  (`packages/midas_nf_preprocess/notebooks/README.md`: *"All notebooks run on CPU with
  synthetic data"*). So `hex_grid`, `diffr_spots`, `process_images`, `seed_orientations` and
  `tomo_filter` — **including `SpotsInfo.bin` itself** — carry **no byte-parity evidence
  against the C**. Their docstrings cite C line numbers: that is provenance, not
  verification.
- **`midas_nf_pipeline` end to end.**
  `tests/integration/test_au_end_to_end.py` is gated on `MIDAS_RUN_INTEGRATION=1` and asserts
  only that the consolidated H5 *contains* `voxels/position` and either `grains/grain_id` or
  `multi_resolution/loop_0_unseeded` (`:61-66`). It is a smoke test: **no numerical
  comparison against anything.** (It is no longer blocked by §8a — the orchestrator
  defects are fixed — but passing it says only that the run produced an H5 with the right
  keys in it.)

### Deliberate, documented departures from the C — do not report these as parity failures

(`packages/midas_nf_fitorientation/README.md:21-57`)

- Orientation optimiser: NLopt Nelder-Mead → vectorised PyTorch NM over all
  `(voxel × winner)` problems at once.
- Calibration optimiser: NLopt NM → L-BFGS over a **soft Gaussian-splat surrogate** with
  tanh-boxed bounds. This optimises a slightly smoothed objective, which is the stated
  reason C-vs-Python misorientation sits near 0.12° rather than at zero
  (`test_vs_c_fit_orientation.py:20-32`).
- Multi-start replaces the C's NM→CRS2→NM ladder; *"CRS2's true global behaviour is lost"*.
- `mic2grains` uses a stable sort where the C used `qsort`, so equal-confidence voxels can
  seed a grain differently — which is why the test asserts orientation but not position
  (`mic2grains.py:348-353`).
- `parse_mic` deliberately **reproduces a C macro bug** so the pixel→voxel assignment stays
  byte-identical: `CalcNorm2` is unparenthesised in the C, so
  `CalcNorm2(X, intX+j, Y, intY+k)` expands to `sqrt((X-intX+j)² + (Y-intY+k)²)`
  (`parse_mic.py:230-242`). "Fixing" it breaks the `.map` parity test by design.
- `midas_hkls`' NF writer is functionally identical but has **deterministic intra-ring row
  order** where the C's `qsort` was unstable (`midas_hkls/nf_hkls.py:16-21`).
- Dropped outright (`packages/midas_nf_pipeline/README.md:151-162`): Parsl multi-node
  dispatch, per-machine config modules, the `MMapImageInfo` fallback, the `FitOrientationGPU`
  C binary.

### Could not verify — do not upgrade these

1. **Rotation-stage names other than `aero`.** Only `aero` was observed (441397/441397 rows
   in `bt_1id_jun25`). What any other value implies for the ω sign is **unknown**. Stop
   and ask.
2. **`*_nf_*` folder naming** is convention; nothing in this repo parses it.
3. **RESOLVED — the §8a orchestrator defects.** They were originally established by
   reading, not execution. They are now fixed (`b95c38c0`, `d231fdf3`) and the three NF
   suites run green in this tree: `midas_nf_pipeline` 39 passed / 2 skipped,
   `midas_nf_preprocess` 442 / 1, `midas_nf_fitorientation` 102 / 24 (2026-08-07,
   `midas_env`, CPU). `run` is the supported route (hard rule 5).
4. **`/grains/` being a single row** was established by reading `aggregate_grains`, not
   against a real H5. Check the row count before quoting it.
5. **The `.mic` header `%TriEdgeSize 0.000000` degradation path** was traced in code
   (`mic2grains.py:365-373, 294`); the downstream effect was not measured on a real run.
6. **Whether the C `ProcessImagesCombined` behaved differently when invoked per distance**
   — the C was not read. The `--all-layers` rule (hard rule 6) is established from the
   Python (`process_images/pipeline.py:229-243`, `cli.py:57-60`) only.
7. **`FileCount.txt` fields 13, 14 and 18–23** were not identified. f13 = 0.02, f14 = 735.42,
   f18 ≈ image count offset, f22/f23 = 721 for Au4. Only f10, f11, f12, f15, f16, f17 are
   established.
8. **`fastsweep_Emon.txt` fields other than f10** were not identified.
9. **`NF.par` fields other than f6–f11, f17 and f29** were not identified.
10. **Absolute `Lsd` for `bt_1id_jun25`** is not established in this document. (§6 is no
    longer a placeholder — it carries the full DetZBeamPos procedure and reference
    numbers in §6h — but those are `bt_1id_jul26`. Nothing here pins jun25.)
11. **Which of geometry A / B is physically correct** (§7e/§7f). A was adopted on operator
    judgement after B scored slightly worse on the map; the two orientations agree to
    0.04°, so this is a **preference, not a measurement**. Do not report A as "the
    verified geometry".
12. **Why `LsdRelativeTol 5` stalled at confidence 0.27** while `LsdRelativeTol 1`
    succeeded on the same data (§7b). Observed once, not diagnosed.
13. **Whether dropping the redundant 3-vertex axis in `screen()`'s centroid branch is
    safe** with non-zero tilts. Affine commutation breaks; the error bound was never
    measured. Do not implement on the assumption that it is safe.
14. **Whether the `.mic` row shortfall is exactly the `Confidence == 0` rows.** A 5046-voxel
    grid produced 5012 text rows; the drop rule is documented (§9a) but the specific
    count was not reconciled against the writer.
15. **The `EdgeLength` ≪ `GridSize` consequences in §10e** were read out of
    `mic2grains.py:198-222, 294` and `fit_orientation.py:524-527`, **not measured on a
    run.** That `EdgeLength` cannot move the voxel count or positions *is* solid — it
    follows directly from `hex_grid/grid.py:97-153`, where the lattice terms contain
    `grid_size` only. The specific claim that a 2 µm merge threshold connects nothing on a
    10 µm lattice has not been executed.

### Verified in this tree — safe to rely on

1. **`BoxSize` semantics and effect** (§7d): 0.949153 → 1.000000 on one Au voxel, matching
   the C reference exactly; Triton fused kernel agrees with eager in both states.
2. **`screen()` dtype rework is answer-preserving** (§8g): every field bit-identical across
   a 5046-voxel grid, `RunTime` the only difference.
3. **`screen()` results are independent of `MIDAS_NF_SCREEN_VOXEL_CHUNK`** (§8h), checked at
   a fixed forced chunk size.
4. **The three calibration negatives** (§7b) — plateau, multipoint, iteration ratchet — were
   each observed directly on `bt_1id_jul26` Au5, not inferred.

### Bottom line

- **Trustworthy, with a C reference behind it:** `.mic` → `.map`/`.kam`/`.grainId`/`.grod`
  rasterisation; the voxel→grain clustering **count**.
- **Trustworthy to ~0.1–0.5° on synthetic Au, sampled:** per-voxel orientations.
- **Unverified against C:** the image-reduction chain that produces `SpotsInfo.bin`, the
  candidate-spot simulation, the grid generation, the pipeline orchestration.
- **Known wrong:** the `/grains/` group and four dataset names in the consolidated H5
  (§9c); grain **radii** from `mic2grains` whenever `EdgeLength ≪ GridSize` (§8a, §10e).
  The orchestrator call sites are **fixed** — see §11 could-not-verify item 3.

**Say which bucket each number you report falls into.** Every quantitative claim must name
the file and the command that produced it.

---

## 12. Report — and what "done" means

### 12a. What to hand back

- The §1 install-gate output, verbatim. Every claim below is conditional on it, and the
  gate has caught a real mixed install (`SumFrames` raw-vs-post-sum) more than once.
- The measured scan definition per scan — `nDistances`, `NrFilesPerDistance`, `StartNr`,
  `OmegaStep` — **re-derived from the per-frame log, not inherited from the calibrant**
  (§3g), with the three consistency checks of §3d.
- The geometry: `BC` per distance from `DetZBeamPos` (§6), `Lsd` from spots (§6i), the
  five acceptance gates of §6g, and **which of them failed**.
- The calibration: the paramfile used, `NumIterations` inside one invocation, and the
  §7b negatives you did *not* re-run.
- The reduction settings, in σ: `BlanketSigma`, `SumFrames` (with the ω-width measurement
  that chose it, §8j), NLM on/off. **At 20-ID, the `PixelScale` you set and the
  `np.unique` output that chose it** — a threshold in ADU means nothing without it (§5d).
- **At 20-ID, where the numbers the files do not carry came from.** The HDF5 has no
  energy, no distance and no pixel size. State the beam energy and say whether it was
  *confirmed by the beamline* (63.314 keV is, for `nfdev_jul26` and `bt_20id_jul26b`) or
  *inferred*; give the scan definition as read from the `nfscan(...)` call in the
  ipython log; and say that only ΔD was supplied, so any δ against a motor scale is not
  anchored (§3h, `RUNBOOK.md` §R2c).
- The `.mic` result with the §9a caveats, and grain **counts** — not `mic2grains` radii
  unless `EdgeLength == GridSize` (§8a).
- Every place this document said *stop and ask* and you proceeded anyway.

### 12b. Say which bucket each number falls in

§11 has five: **has a real parity test against C**, **does not**, **deliberate departure
from C**, **could not verify**, **verified in this tree**. Put every number in one.

**In particular:** the entire image-reduction chain that produces `SpotsInfo.bin`, the
candidate-spot simulation and the grid generation carry **no byte-parity evidence against
the C** (§11). Their docstrings cite C line numbers — that is provenance, not
verification. A reconstruction is not wrong because of this, but a report that does not
say it is overclaiming.

### 12c. Done means

- [ ] §1 install gate run, output pasted, **no package below the strictest floor**
- [ ] ω sign established — 1-ID: par field 9, or **stopped and asked** if not `aero`.
      20-ID: §2a, `aero` and negated, so `OmegaStart 180` / `OmegaStep -0.25`. Any other
      beamline: **stopped and asked**
- [ ] **20-ID only:** `PixelScale` set from `np.unique` on a frame, **not inherited** from
      another scan — the encoding is per scan (§5d, §3h)
- [ ] **20-ID only:** `NrFilesPerDistance` taken from the ω **range**, not from the length
      of `exchange/data` — the sweep can exceed 360° (1442 frames for 1440 real ones)
- [ ] scan definition re-derived per scan (§3g); §3d's three checks all pass
- [ ] `StartNr` = the first image; the GE skip-first-frame rule **not** carried over (§3g)
- [ ] raw frames looked at before anything was built (§5), on the temporal-median + LoG
      path — **never a max-projection** (hard rule 4)
- [ ] BC measured this campaign, not inherited (§6d); β measured this beamtime (§6f)
- [ ] all five §6g acceptance gates checked, pass or fail recorded
- [ ] calibration accepted on something other than confidence (hard rule 14); `BoxSize`
      checked first (§7d)
- [ ] reduction tuned before geometry on weak signal (hard rule 18), threshold set in σ
- [ ] orientation field tested by **neighbour vs random misorientation**, not by maxC or
      median — those are blind to the plateau failure (trap table, lab notebook §8h)
- [ ] every number bucketed per §12b, with its provenance

**If a box cannot be ticked, say so in the report rather than leaving it blank.** An
unticked box is a known limit; a silently skipped one becomes a false claim.

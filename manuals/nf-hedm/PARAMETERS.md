# Parameter-file reference

> Part of the **NF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 10. Parameter-file reference

One whitespace-delimited `Key Value [Value…]` per line; `#` comments; blanks skipped.

### 10a. Parser behaviour that differs between packages — know this before debugging

- **Repeated keys.** `midas_nf_pipeline.parse_parameters` keeps the **last** occurrence
  (`midas_nf_pipeline/params.py:51-100`); `collect_multiline()` gets *all*
  (`:103-127`). `midas_nf_fitorientation` and `diffr_spots` instead **accumulate** `Lsd`,
  `BC`, `OmegaRange`, `BoxSize`, `RingsToUse` into per-distance lists
  (`fitorientation/params.py:221-226`, `diffr_spots/params.py:75-109`). The pipeline's HKL
  stage deliberately uses the **last** `Lsd` line (`stages.py:30-40`).
- **Multi-value keys** get a fixed float count and raise if short: `LatticeParameter`(6),
  `GridMask`(4), `BC`(2), `OmegaRange`(2), `BoxSize`(4), `BCTol`(2), `GridPoints`(12),
  `GridRefactor`(3) (`midas_nf_pipeline/params.py:32-41`).
- **Everything else is stored as the first token only, as a string**
  (`midas_nf_pipeline/params.py:98-99`). `midas_nf_fitorientation` **silently skips
  malformed lines** (`fitorientation/params.py:343-346`).
- **Cheapest sanity check:** the fitorientation parser asserts `len(Lsd) == nDistances` and
  `len(BC) == nDistances` and raises otherwise (`fitorientation/params.py:352-361`).

Annotated reference file: `NF_HEDM/Example/ps_au.txt` (2 distances, `Lsd 8289.154576` /
`10290.724494`, `BC 985.415831 17.510494` / `985.161497 24.511210`, `px 1.48`,
`NrPixels 2048`, `OmegaStart 180`, `OmegaStep -0.25`, `StartNr 0`, `EndNr 1439`,
`NrFilesPerDistance 1440`).

### 10b. Material / crystallography

| Key | Values / units | Read by |
|---|---|---|
| `LatticeParameter` | `a b c α β γ` — Å, deg | pipeline (`params.py:33`); HKL gen (`stages.py:175-178`); fitorientation (`params.py:265`); diffr-spots (`params.py:91`); mic2grains (`mic2grains.py:65-68`) |
| `LatticeConstant` | alias | **only** fitorientation, diffr-spots and the H5 consolidator (`consolidate.py:130-134`). **Use `LatticeParameter`.** The pipeline's multi-value list contains `LatticeParameter` only (`params.py:33`), so writing `LatticeConstant` leaves `p["LatticeParameter"]` absent and the HKL stage raises `KeyError` (`stages.py:123`). Cost of using `LatticeParameter`: the consolidator greps for `LatticeConstant` only, so `/parameters/` carries no lattice. Cosmetic. |
| `Wavelength` | Å | HKL gen, diffr-spots, fitorientation |
| `SpaceGroup` | 1–230 | HKL gen, seed cache, `ParseMic`, mic2grains, diffr-spots, fitorientation |
| `SGNr` | fallback alias | pipeline stages only (`stages.py:175, 264, 638`) |

### 10c. Detector geometry

| Key | Values / units | Read by |
|---|---|---|
| `nDistances` | count | pipeline, image processing loop, fitorientation (`midas_nf_pipeline/params.py:43`) |
| `Lsd` | µm — **one line per distance** | fitorientation (list), diffr-spots (list), HKL gen (**last** line) |
| `BC` | `ybc zbc` px — one line per distance | fitorientation (`params.py:224-226`) |
| `tx` `ty` `tz` | deg — shared across distances | fitorientation |
| `Wedge` | deg | fitorientation |
| `px` | µm, square | diffr-spots, fitorientation |
| `NrPixels` | px — sets both Y and Z | image processing, fitorientation |
| `NrPixelsY` `NrPixelsZ` | px — override `NrPixels`; 0-fallback chain at `process_images/params.py:60-68` | image processing, fitorientation |
| `MaxRingRad` | µm | diffr-spots, fitorientation |
| `RhoD` | µm — **preferred** alias of `MaxRingRad` for the HKL stage only (`stages.py:43-49`) | HKL gen |

### 10d. Scan

| Key | Values / units | Read by |
|---|---|---|
| `OmegaStart` | deg — ω of the first frame. **See §2 for the sign.** | fitorientation |
| `OmegaStep` | deg between **RAW** images (negative = CW). Not multiplied by `SumFrames` — the fit does that (§8j). **See §2.** | fitorientation |
| `OmegaRange` | `min max` deg — one line per distance | fitorientation (list), diffr-spots (list), pipeline |
| `StartNr` `EndNr` | frame numbers. **`EndNr` is OPTIONAL for NF** and derived as `StartNr + NrFilesPerDistance − 1`; the fitter takes frames/distance from `NrFilesPerDistance`, not from `EndNr−StartNr+1` (`fitorientation/params.py:204-221`, commit `60dcc94c`). Supply it by hand only if you want the inconsistency check. FF/PF still require it. | fitorientation |
| `NrFilesPerDistance` | **RAW** image count per distance — the one source of truth for frames/distance. Not divided by `SumFrames` (§8j). | image processing, pipeline, multi-layer offset, fitorientation |
| `WFImages` | wide-field frames per layer, **excluded** from `NrFilesPerDistance` (`process_images/io.py:31-33`) | image processing |
| `RawStartNr` | first raw file number; rewritten per sample layer | image processing, pipeline |

Arithmetic consistency check (derived from the example, **not enforced by code**):
`NrFilesPerDistance ≈ ω sweep / |OmegaStep|`. In `ps_au.txt` that is `1440 × 0.25 = 360°`.

### 10e. Sample geometry and I/O

| Key | Values / units | Read by |
|---|---|---|
| `Rsample` | µm — radius the hex grid must cover | hex grid (`hex_grid/params.py:19`) |
| `GridSize` | µm — the hex **triangle edge**, NOT the voxel spacing. Nearest-neighbour pitch is **`GridSize/√3`** (measured: `GridSize 10` → 5.7735 µm; `GridSize 20` → 11.547 µm). Treating it as the pitch overstates areas by 3× and diameters by **1.73×**. Get the cell area from the grid itself (`hull area / n_voxels`), never from `GridSize²`. **Overwritten on disk each multi-resolution loop** (`workflows.py:373-376`) — `EdgeLength` is NOT, and stays 1 (verified: `grid.txt` col 5 = 0.5000 exactly, `%TriEdgeSize 1.000000` in the loop `.mic`) | hex grid; fitorientation (multipoint only) |
| `EdgeLength` | µm — probe-triangle edge; `0`/absent ⇒ equals `GridSize` (`hex_grid/params.py:25-26`). **An independent knob — see below.** | hex grid |
| `GridFileName` | default `grid.txt` | hex grid, fitorientation |
| `GridMask` | 4 floats. The code filters grid columns 2 and 3, i.e. **x and y in µm** (`stages.py:345-370`). `ps_au.txt:89` labels them `ymin ymax zmin zmax`; **the code's meaning wins.** | pipeline `run_grid_mask` |
| `GlobalPosition` | µm — written into the `.mic` header | `ParseMic`, consolidator |
| `TomoImage` | path to a **square `uint8`** mask; side inferred from file size (`tomo_filter/filter.py:33-52`) | pipeline `run_tomo_filter` — **fixed**; the old path-vs-tensor defect is documented in `stages.py:327-331`. The `tomo-filter` CLI (§8b step 3) remains the way to re-run it standalone |
| `TomoPixelSize` | µm per tomo pixel | as above |

#### Building a SYNTHETIC tomo mask (no tomography required)

When the sample is a few small particles inside a large search area, a hand-built mask cuts
the voxel count enormously. On `nfdev_jul26` (two Au cubes, one on-axis and one 497 µm off)
three dilated discs kept **5.3 %** of an `Rsample 600` disc — **19× fewer voxels**.

Format and convention, from the loader and sampler (`tomo_filter/filter.py:33-52`,
`121-149`):

- raw **square `uint8`**; the byte count must be a **perfect square** (side = `isqrt(size)`)
- **any nonzero pixel keeps the voxel** (`mask = values != 0`)
- sampling is `xPos = int(x_um/px) + n//2`, `yPos = int(y_um/px) + n//2`, read as
  **`tomo[n - yPos, xPos]`** — Y flipped, C parity
- out-of-image ⇒ 0

```bash
midas-nf-preprocess tomo-filter grid.txt grid_filt.txt --tomo tomo.bin --px-tomo 2.0
#   then point GridFileName at grid_filt.txt.  Use the CLI, not the pipeline stage.
```

Three things that will bite:

1. **The index is `n - yPos`, not `n-1-yPos`.** A voxel at the extreme −y edge maps to row
   `n` and falls out of the array. **Make the mask larger than `Rsample`** (e.g. ±800 µm of
   mask for `Rsample 600`) so no voxel lands at the edge.
2. **Dilate.** Use a radius comfortably larger than the particle — position uncertainty,
   the grid pitch and any residual geometry error all eat margin. 80 µm radius for a
   ~50 µm cube is reasonable.
3. **Verify the mask by round-tripping it through `sample_tomo`**, not by reasoning about
   the flip. Probe a point you expect to keep and one you expect to reject and check the
   returned values. The Y-flip convention is easy to get backwards, and a mirrored mask
   silently deletes exactly the voxels you wanted.

**Tip — when a position is known only up to a convention**, put a dilated region at *each*
candidate rather than guessing. On `nfdev_jul26` the second cube's sample-frame position is
`±(406.4, 285.7) µm` — the magnitude is measured to ±0.7 µm but the sign pair depends on the
ω sign and the detector Y handedness. Masking **both** candidates costs one extra disc and
lets the reconstruction settle the handedness empirically.
| `DataDirectory` | path — raw TIFFs | everything |
| `OutputDirectory` | path — falls back to `DataDirectory` | everything |

> ## ⇒ ALWAYS SET `EdgeLength 1`. On every paramfile, at every `GridSize`.
>
> This is the single highest-value line in an NF paramfile and it is **absent by
> default**, in which case `EdgeLength` silently becomes `GridSize` and every voxel
> triangle grows to the grid pitch.
>
> `screen()` rasterises each voxel's triangle over its bounding box **in detector
> pixels** — a `(T, P, Q)` tensor with `P, Q ≈ EdgeLength / px` (`screen.py:229-230`).
> So per-voxel cost scales as **`EdgeLength²`**:
>
> | `EdgeLength` | triangle at px 0.548 | pixels per triangle |
> |---|---|---|
> | 1 µm | 1.8 px | **~4** |
> | 4 µm | 7.3 px | ~53 |
> | 16 µm | 29 px | **~850** |
>
> `EdgeLength 1` versus `EdgeLength 16` is a **~200× difference in screen cost per
> voxel.** With it omitted at `GridSize 16`, a 4202-voxel annulus scan on
> `nfdev_jul26` had not finished after **3.4 hours**; with `EdgeLength 1` the same
> region at `GridSize 4` — **66,864 voxels, 16× more** — runs in a fraction of that.
>
> **Corollary: with `EdgeLength` omitted, coarsening `GridSize` saves NOTHING.**
> Voxel count falls as `1/GridSize²` while per-voxel cost rises as `GridSize²`, so
> the product is flat. Measured, same data and geometry:
>
> | run | voxels | `GridSize` | `screen` |
> |---|---|---|---|
> | step4_std | **9038** | 4 (EdgeLength defaulted) | **7900 s** |
> | step4_std_g6 | **6676** | 6 (EdgeLength defaulted) | **8265 s** |
>
> Fewer voxels, MORE time. Choosing a coarse grid "for speed" is a null optimisation
> that costs resolution for nothing. **With `EdgeLength 1` pinned, `GridSize` controls
> cost as you would expect** — and a fine grid becomes affordable.
>
> The probe-vs-tile consequences below still hold (`mic2grains` areas describe the
> probe, `doNeighborSearch` will not connect) — that is the deliberate trade, and it
> is the right one for locating and orienting grains.

#### `EdgeLength` vs `GridSize` — independent, and the difference is deliberate

**Earlier revisions of this file said `EdgeLength` must equal `GridSize` and that
setting it smaller "breaks the grid". That is RETRACTED** (lab notebook R2).
`EdgeLength` is a supported, independent knob; only the *default* ties the two
together (`hex_grid/params.py:25-26`, `grid.py:88-89`).

In `make_hex_grid` the two quantities touch different things:

- the **lattice** — how many voxels and where they sit — comes from `grid_size`
  alone: `a_large = 2·Rsample/√3` and `nr_hex = ceil(a_large/grid_size)`
  (`hex_grid/grid.py:97-100`), `nr_row_elements = 2(2·nr_hex − |i|) + 1`
  (`:117`), `x = xstart + grid_size·j/2` (`:142`),
  `ht_triangle = √3·grid_size/2` (`:98`);
- `edge_length` sets only the **probe triangle**: `edge_half = edge_length/2`
  (`:153`, grid.txt column 5) and the sub-triangle offsets
  `xt1 = edge_length·√3/6`, `xt2 = 2·edge_length·√3/6` (`:105-106`, columns 1-2).

**Changing `EdgeLength` therefore never changes the voxel count or the voxel
positions.** Small probe triangles on a coarse lattice are an intentional mode —
a sparse point sampling of the volume. Removing an `EdgeLength 1` line from a
`GridSize 10` paramfile made the triangles 10 µm and cost a ~94 GiB-per-voxel
allocation, with the triangle *count* unchanged (lab notebook R2).

What it **does** change is everything downstream that reads `TriEdgeSize`, since
the fitter writes `2·edge_half` into `.mic` column 5
(`fit_orientation.py:524-526`):

| consumer | uses `TriEdgeSize` as | effect when `EdgeLength` ≪ `GridSize` |
|---|---|---|
| `mic2grains` grain radius | `area = TriEdgeSize²·√3/4` (`mic2grains.py:294`) | the area of the *probe*, not of the lattice cell — smaller by `(GridSize/EdgeLength)²` |
| `mic2grains` spatial merge (`doNeighborSearch 1`) | bins of side `1.01·TriEdgeSize`, edge added when `dist² < (2·TriEdgeSize)²` (`mic2grains.py:198-222`) | lattice neighbours are `GridSize/2`–`GridSize` apart, so `EdgeLength 1` on `GridSize 10` gives a 2 µm threshold that connects **nothing** — every voxel becomes its own grain. **Read from the code, not measured (§11).** |
| soft-overlap splat σ, per-voxel path | `auto_sigma_px(edge_half, px)` (`fit_orientation.py:527`) | none in practice — `auto_sigma_px` clamps at 1.0 px (`soft_overlap.py:425`) and NF values sit below the clamp either way |

So: a small `EdgeLength` is the right choice when you want a sparse probe, and
then `mic2grains` output describes the probe rather than the volume — read grain
areas accordingly, and do not expect `doNeighborSearch 1` to connect anything.
Omit the key when you want the triangles to tile the lattice and grain areas to
mean volume fractions.

**Multi-resolution:** `GridRefactor` rewrites `GridSize` every loop (10 → 5 →
2.5) but a hardcoded `EdgeLength` does *not* follow, so a fixed probe size is
held across all levels. That is a real effect either way — intended if you want a
constant probe, surprising if you assumed it tracked. Omitting the key makes the
edge track `GridSize` at every level.

**Inconsistency between the two fit paths**, worth knowing before debugging a σ:
the per-voxel path takes the splat σ from `grid.txt` column 5
(`fit_orientation.py:527`), whereas `fit_multipoint` takes it from the paramfile
`GridSize` and never reads `edge_half` at all
(`fit_multipoint.py:165`: `auto_sigma_px(p.grid_size_um/2.0, p.px, …)`). Set the
two keys differently and the paths disagree by construction; both clamp to 1.0 px
at typical NF values, so it has not bitten yet.

Check what you actually got rather than trusting the paramfile — column 5 of
`grid.txt` is `edge_half`:

```bash
head -2 grid.txt | tail -1 | awk '{print "edge_half =", $5}'
# key omitted, GridSize 10 -> 5.0      EdgeLength 1 -> 0.5
```

### 10f. Image processing

All read by `midas_nf_preprocess.process_images` (`process_images/params.py:83-105`).

| Key | Units | Meaning |
|---|---|---|
| `BlanketSubtraction` | counts (**float** since `4e90be80`; was int) | flat offset subtracted **after** the temporal median, then clamped at 0 (`process_images/pipeline.py:165-166`). An absolute count does not transfer between reductions — prefer `BlanketSigma` |
| `BlanketSigma` | multiples of σ | **the transferable threshold.** `threshold = BlanketSigma × σ_MAD` of the POST-denoise residual, measured **per layer**; overrides `BlanketSubtraction` when set (`4e90be80`, §8k). ~3.5σ was optimal across a 14-configuration catalog however it was reached |
| `MedFiltRadius` | px | spatial median radius: `0` = identity, `1` = 3×3, `2` = 5×5 (`process_images/params.py:225`) |
| `GaussFiltRadius` | px | maps to the LoG `sigma` field — the *name* is `GaussFiltRadius`, the field is `sigma` |
| `LoGMaskRadius` | px | LoG kernel half-width |
| `DoLoGFilter` | 0/1 | `0` labels connected components of `img > 0` directly (`pipeline.py:180-195`). **Not a simple "always 1"** — LoG can suppress genuine weak peaks, so weak-signal samples are run with `0` and tolerate the cosmics. See the decision table in §5b. Changing it requires regenerating `SpotsInfo.bin`. |
| `OrigFileName` / `ReducedFileName` | stem | input / reduced stems |
| `extOrig` / `extReduced` | e.g. `tif` / `bin` | extensions |
| `WriteFinImage` | 0/1 | forced to 1 when `Deblur != 0` (`process_images/params.py:229`) |
| `Deblur`, `WriteLegacyBin` | 0/1 | |
| `SoftTemperature` | float or `auto` | **Python extension, not in the C** — sigmoid temperature for the differentiable spot-probability surrogate (`params.py:14-18`) |
| `NLMDenoise` | 0/1 | NLM on the median-corrected residual, before `BlanketSubtraction` (§8f) |
| `NLMH` | × σ_MAD | filter strength as a **multiple of σ_MAD**. Useless when σ_MAD = 0 — see `NLMHAbsolute` |
| `NLMHAbsolute` | **counts** | absolute filter strength; overrides `NLMH · σ_MAD` when > 0. **Required on photon-starved detectors** where σ_MAD is exactly 0, otherwise NLM is skipped (now with a `RuntimeWarning`; it used to be silent) |
| `NLMPatchSize` / `NLMPatchDistance` | px | NLM patch geometry (5 / 6) |

### 10g. Orientation search

| Key | Values / units | Read by |
|---|---|---|
| `MinFracAccept` | 0–1 | phase-1 screen threshold; also a `MinConfidence` fallback in `mic2grains` (`mic2grains.py:80-83`). `ps_au.txt:124` suggests **0.1 seeded / 0.04 unseeded / 0.01 deformed** |
| `OrientTol` | deg | phase-2 search box per seed (`fit_orientation.py:466-470`). Default 1.0 |
| `ExcludePoleAngle` | deg | diffr-spots, fitorientation |
| `BoxSize` | 4 floats µm, relative to beam centre — one line per distance | diffr-spots (list), fitorientation (list) |
| `MinConfidence` | 0–1 | `mic2grains`; fitorientation; the multi-resolution bad-voxel filter `_filter_bad_voxels` (`workflows.py:145-170`) |
| `NrOrientations` | count | diffr-spots. **The pipeline overwrites it** from the seed-file line count (`stages.py:256-262`). Cubic-high cache = 243129 lines, matching `ps_au.txt:140` |
| `SeedOrientations` | path to the comma-separated `w,x,y,z` CSV (`seed_orientations/io.py:24-38`) | diffr-spots, pipeline |
| `SeedOrientationsAll` | path — the full unseeded library. **Required for multi-resolution** (`workflows.py:360-364`) | pipeline |
| `GrainsFile` | FF `Grains.csv`; rewritten per refinement loop | pipeline FF-seed stage |
| `SaveNSolutions` | count | fitorientation; `.AllMatches` record width; binary-merge record size (`workflows.py:165-166`) |
| `MinMisoNSaves` | deg | separation between saved solutions |
| `NearestMisorientation` | 0/1 | fitorientation |
| `RingsToUse` | ring number, **repeatable** | diffr-spots, fitorientation |
| `MaxAngle` | deg | `mic2grains` clustering tolerance; default 1.0 (`mic2grains.py:52`) |
| `GBAngle` | deg | `ParseMic` grain-boundary threshold for `.map.grainId`; default 5.0 (`parse_mic.py:56`) |

### 10h. Phase, output, calibration tolerances, multi-resolution, denoise

`NumPhases` (count, into the `.mic` header), `PhaseNr` (int, into `.mic` col 11 and `.map`
plane 5), `MicFileBinary` (filename), `MicFileText` (see below).

> **`ParseMic` does NOT append `.mic`.** The text microstructure is written to the
> `MicFileText` value **verbatim**; only the companions get suffixes
> (`.AllMatches`, `.map`, `.map.kam`, `.map.grainId`, `.map.grod`). So
> `MicFileText Microstructure` produces a text file literally named `Microstructure`, and
> anything looking for `Microstructure.mic` gets `FileNotFoundError`. **Put the `.mic` in
> the value yourself** — the bundled reference does exactly that
> (`Au_txt_Reconstructed.mic`). Verified on `nfdev_jul26`, where `ParseMic` reported
> `[Microstructure, Microstructure.AllMatches, Microstructure.map, …]`.

Calibration tolerances, all read by `midas_nf_fitorientation`
(`fitorientation/params.py:276-289, 328-340`). Each becomes an
`x = x0 + tol*tanh(u)` box so the refined value cannot leave the box
(`packages/midas_nf_fitorientation/README.md:40-42`):

| Key | Units | Default |
|---|---|---|
| `LsdTol` | µm | 1000.0 |
| `LsdRelativeTol` | µm (between distances) | 100.0 |
| `BCTol` | `a b` px | 1.0, 1.0 |
| `TiltsTol` | deg | 0.05 |
| `NumIterations` | multi-start trials in `fit_multipoint` | 1 |
| `WedgeTol` | deg — only if `RefineWedge 1` | 0.05 |
| `RefineWedge` | 0/1 — **new, not in the C** | 0 |
| `TikhonovCalibration` | λ; 0 disables — **new** | 0.0 |
| `TikhonovSigmaLsd` / `SigmaTilts` / `SigmaBC` / `SigmaWedge` | µm / deg / px / deg | 100.0 / 0.05 / 1.0 / 0.05 |
| `GaussianSplatSigmaPx` | px — override the auto soft-overlap σ — **new** | auto |
| `GridPoints` | 12 values — a raw `.mic` data row. The fitter reads `args[3,4,6,7,8,9]` (`.mic` columns X, Y, UpDown, Eul1-3) as `xc yc ud eul1 eul2 eul3` (`params.py:328-335`), where `args` is the line **after** the `GridPoints` key. **Earlier revisions of this table said 4,5,7,8,9,10 — those are the C's line-token indices, one too high for Python, and they are the off-by-one that scored 0.0026 against the C's 0.8515** (lab notebook defect 10). If you see `args[4]`/`args[10]` in a parser, it is the broken version. | — |

Without a `GridPoints` block, `fit_multipoint_run` derives its voxel set from the
reconstructed `MicFileText` `.mic` — highest-confidence voxels above `MinConfidence`
(`packages/midas_nf_fitorientation/notebooks/README.md:26-31`). Note `ps_au.txt:174-176`
uses tighter values than the defaults: `LsdTol 500`, `LsdRelativeTol 3`, `BCTol 2 0.2`.

`GridRefactor StartingGridSize ScalingFactor NumLoops` — µm, ×, count; absent ⇒ single
resolution (`workflows.py:253-261`).

Denoise (optional step 0, `stages.run_denoise`, `stages.py:56-111`) requires the separate
`MIDAS-NF-preProc` package. `DenoiseMethod n2v` **raises** without a CUDA GPU
(`stages.py:66-74`). On success it rewrites `DataDirectory` in memory **and on disk, by
appending a line to your parameter file** (`stages.py:108-111`). Keys: `Denoise` (0),
`DenoiseMethod` (`nlm`|`n2v`, default `nlm`), `DenoisedDirectory`
(`<DataDirectory>/denoised`), `DenoiseConfigFile`, `DenoiseCheckpoint`, `DenoisePattern`
(`*.tif`), `DenoiseTrainJointly` (0), `DenoiseFinetune` (0), `DenoiseMaskThreshold`
(unset ⇒ `None`), `DenoiseNoMedian` (0 — 1 disables the temporal median).

### 10i. Keys in `ps_au.txt` that no Python NF module reads

Verified by grepping every `.py` under `packages/midas_nf_*` and `packages/midas_hkls`:
**`OnlySpotsInfo`, `WriteImage`, `LayerThickness`, `GlobalPositionFirstLayer`** are never
read. `PrecomputedSpotsInfo` (added by the fitorientation integration test's patched
paramfile, `tests/integration/test_vs_c_fit_orientation.py:119`) is likewise unread.
`Ice9Input` is explicitly **deprecated and silently ignored**
(`fitorientation/params.py:319-322`). Leaving them in is harmless; expecting them to do
anything is not.

---

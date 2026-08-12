# Phase 4 — Run the reconstruction

> Part of the **NF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 8. STEP 6 — Run the reconstruction

### 8a. `midas-nf-pipeline run` — now the supported route

**This section used to say "do not use it".** Ten call-site defects made the orchestrator
unusable; all ten are fixed (commits `b95c38c0`, `d231fdf3`), and the multi-resolution
ladder now runs end to end. The defect list, with what each one did, is in the lab
notebook §2 — it is history, not instructions.

```bash
midas-nf-pipeline run params.txt \
    --n-cpus 64 --device cuda \
    --fit-gpus 0,1 \            # shard the FIT across GPUs, one process each
    --no-image-processing        # only if SpotsInfo.bin already exists
```

`--fit-gpus` splits the voxel range into one disjoint block per GPU in **every** loop.
Without it the fitter uses a single GPU while the rest idle. The outputs are pre-allocated
once by the parent and workers open them `r+`; do not hand-roll this, `MicWriter`'s default
zeroes the whole file on open (lab notebook §3c).

**Three things that bite on first invocation** (all hit on `bt_20id_jul26b`, 2026-08-01):

1. **`export MIDAS_NF_SEED_DIR=<…>/NF_HEDM/seedOrientations`.** The seed stage re-derives
   the cache path from the *install* directory (`from_cache.py:106`), which in a conda env
   resolves to `…/lib/python3.11/../NF_HEDM/seedOrientations` and does not exist. It dies
   with `SeedCacheNotFound` **after** writing `hkls.csv`, so the run looks like it started
   fine. Passing `--seed-dir` to the standalone `seed-orientations` command does *not* help
   — the orchestrator calls the loader itself.
2. **The driver works inside `<OutputDirectory>/LayerNr_<N>/`, not `OutputDirectory`.**
   With `--no-image-processing` you must put `SpotsInfo.bin` (or a symlink) *there*, not in
   the parent. `rf=…/LayerNr_1` appears in the first few log lines — read it.
3. **`RawStartNr`, `OrigFileName`, `ReducedFileName` are demanded by the validator even
   with `--no-image-processing`.** They are 3 hard errors; the run continues anyway unless
   `--strictValidation`. Add stubs so real errors are not lost in the noise.

**Reading the loop outputs.** `MicrostructureBinary.mic` is **pre-allocated and zeroed for
the NEXT loop** as soon as a loop finishes, so reading it mid-run returns all zeros and an
apparently dead reconstruction. The per-loop result is `<MicFileText>.<k>.mic`. A
consolidation line reading `Wrote 1 grains to /grains/` likewise does not mean one grain —
check `Grains.csv.<k>` (`%NumGrains`) instead.

**Grain RADII from `mic2grains` are invalid whenever `EdgeLength ≪ GridSize`** — which is
always, since `EdgeLength 1` is mandatory (§10e). Verified on `bt_20id_jul26b` loop 0: 93
grains with radii 0.37–11.08 µm, median 1.78 µm, against an indexed area of ~499,000 µm²;
93·π·1.78² ≈ 925 µm², a **500× discrepancy**. The merge threshold is `2·TriEdgeSize` = 2 µm
while neighbours sit `GridSize/√3` = 5.77 µm apart, so the radii describe the **probe
triangle, not the cell**. Orientations and grain *count* are fine (they seed the next loop
correctly); sizes are not. For real grain sizes, segment by **neighbour misorientation**
(link neighbours below ~5° with `midas_stress.misorientation`, cubic, then take connected
components) and multiply voxel counts by the **measured** cell area — see the pitch warning
in §10e.

Two things the orchestrator still will not do:

* **A phase that indexes nothing** stops cleanly after loop 0 with
  "no grains above MinConfidence" rather than refining — that is a valid result, not an
  error (lab notebook §4c).
* **`refine-params --multi-point`** defaults to `--objective hard`, which optimises the
  same FracOverlap the C does. The `soft` surrogate is still selectable but is **not**
  equivalent — see lab notebook §3b before using it.

The nine-command route in §8b remains valid and is still the right thing when you need to
re-run one stage in isolation.

### 8b. The route that works — nine commands

Every step below has an `argparse`-defined signature, so the flags are the flags. **Run all
of them from inside `OutputDirectory`** — the fitter resolves all inputs relative to it
(`fit_orientation.py:238-251`).

```bash
cd <OutputDirectory>

# 0. hkls.csv — no console script exists for the NF variant. Use the stage helper.
/home/beams12/S1IDUSER/opt/envs/midas/bin/python - <<'PY'
from midas_nf_pipeline.params import parse_parameters
from midas_nf_pipeline import stages
p = parse_parameters('params.txt'); p['resultFolder'] = '.'
stages.run_get_hkls(p, 'params.txt')
PY

# 1. seed orientations
midas-nf-preprocess seed-orientations --method cache --space-group 225 \
    --output seedOrientations.csv
#    FF-seeded instead (writes the 11-column layout; diffr-spots reads the first
#    4 columns, hkls.py:107-113, so it is accepted directly):
# midas-nf-preprocess seed-orientations --method from-grains \
#     --grains-file Grains.csv --output seedOrientations.csv
#    then set  SeedOrientations seedOrientations.csv
#         and  NrOrientations <wc -l of that file>   in params.txt

# 2. voxel grid
midas-nf-preprocess hex-grid params.txt                 # -> grid.txt

# 3. optional grid mask
midas-nf-preprocess tomo-filter grid.txt grid_filt.txt --tomo tomo.bin --px-tomo 1.5
# midas-nf-preprocess tomo-filter grid.txt grid_filt.txt --bbox -500 500 -500 500

# 4. forward-simulate candidate spots
midas-nf-preprocess diffr-spots params.txt              # -> Key.bin OrientMat.bin DiffractionSpots.bin

# 5. raw TIFFs -> SpotsInfo.bin.  --all-layers IS MANDATORY.
#    The positional "1" is a DETECTOR DISTANCE index and is ignored with --all-layers.
midas-nf-preprocess process-images params.txt 1 --all-layers --device cuda --dtype fp32

# 6. fit orientations (positionals: paramfile blockNr nBlocks nCPUs)
midas-nf-fit-orientation params.txt 0 1 8 --device cuda --fp32

# 7. binary -> text .mic + .map/.map.kam/.map.grainId/.map.grod
midas-nf-pipeline parse-mic params.txt

# 8. cluster voxels into grains (doNeighborSearch=1 -> spatial BFS)
midas-nf-pipeline mic2grains params.txt out.mic Grains.csv 1 8

# 9. bundle into one readable HDF5
midas-nf-pipeline consolidate out.mic --paramFN params.txt --output out_consolidated.h5
```

Seed cache: `NF_HEDM/seedOrientations/`, overridable with `$MIDAS_NF_SEED_DIR` or
`--seed-dir` (`from_cache.py:36-49`). Fully populated in this checkout:
`seed_cubic_high.csv` 243129 rows, `seed_hexagonal_high.csv` 486755 rows. The
`orientations_master.bin` + `lookup_<type>.bin` fallback is present too
(`from_cache.py:64-82`).

Fit-orientation flags (`midas_nf_fitorientation/cli.py:33-63`): `--device {auto,cpu,cuda}`,
`--fp32`, `--screen-only`, `--verbose`, `--lbfgs-max-outer N` (20), `--lbfgs-max-iter N`
(20), `--refine {nm-batched,nm-serial,lbfgs+nm,lbfgs}` (default `nm-batched`),
`--nm-max-iter N` (200; the C used 5000), `--nm-batch-size N` (4096 — the GPU-memory
knob).

`nm-triton` is **not** a CLI choice (`midas_nf_fitorientation/cli.py:47-48`). It is
auto-selected when `--refine nm-batched` **and** device is CUDA **and** Triton is
importable **and** the obs volume is bit-packed (`fit_orientation.py:370-377`).

Manual sharding: block *b* of *nBlocks* covers voxels
`[ceil(N/nBlocks)*b, min(ceil(N/nBlocks)*(b+1), N-1)]` (`io.py:236-245`). **Multi-process
sharding is deliberately not wired** — `MicFileBinary` writes need a `pwrite`-safety audit
first (`packages/midas_nf_pipeline/USAGE.md:244-256`).

### 8c. Multi-layer by hand

`run_multi_layer` (`workflows.py:683`, wired from `cli.py:123-129`) **works** — the
"broken `run` path" this section used to describe was fixed in `b95c38c0`/`d231fdf3`
(§8a). Use it. The manual reproduction below is still the right move when you need to
re-run one layer in isolation: for each sample layer *n*, make
`<result-folder>/LayerNr_<n>/`, copy the paramfile in, and rewrite two keys:

```
OutputDirectory  <result-folder>/LayerNr_<n>
RawStartNr       RawStartNr0 + (n-1) * nDistances * NrFilesPerDistance
```

then run §8b in that directory. Per-layer grain list afterwards:
`midas-nf-pipeline mic2grains ... > <result-folder>/GrainsLayer<n>.csv`.

**Known inconsistency in the built-in version** (read, not executed): the per-layer grain
list is built from `<base>.<NumLoops>.mic` — the *seeded* pass of the last loop
(`workflows.py:608-612`) — not from `<base>_merged.<NumLoops>.mic`, which the loop itself
designated final (`workflows.py:532`). If you build grain lists yourself, use the merged
file.

**Multi-resolution**, if you want it: `GridRefactor StartingGridSize ScalingFactor NumLoops`;
absent ⇒ single resolution, `NumLoops = 0`, only loop 0 runs — there is no separate code
path (`workflows.py:253-261`). Loop *k* runs at `StartingGridSize / ScalingFactor**k`
(`workflows.py:373`). Each loop *k* ≥ 1 (`workflows.py:369-542`): rewrite `GridSize` into
the paramfile → `Mic2GrainsList` on the previous `.mic` → seeded pass → **bad-voxel
filter** (voxels with `Confidence < MinConfidence` collected, `grid.txt` *overwritten*
with only those lines, `workflows.py:437-470`; short-circuits if none) → unseeded pass on
just those voxels from `SeedOrientationsAll` → binary merge by `pwrite` overlay at
full-grid offsets, then `ParseMic` (`workflows.py:600-660`). `doImageProcessing` is forced
to 0 from loop 1 on (`workflows.py:371`).

Stage labels, in resume order (`workflows.py:54-68`): `loop_0_initial`, then
`loop_<k>_seeded`, `loop_<k>_unseeded`, `loop_<k>_merge`.

### 8d. Where files land

Everything is **flat** in `OutputDirectory` (fallback `DataDirectory`, then cwd —
`workflows.py:239-244`); the driver `chdir`s into it (`workflows.py:44-52`).

| File | Written by | Contents |
|---|---|---|
| `midas_log/` | driver (`workflows.py:242-243`) | log dir |
| `hkls.csv` | HKL gen | 11 cols `h k l d RingNr g1 g2 g3 θ 2θ R` (`midas_hkls/nf_hkls.py:1-21`) |
| `<SeedOrientations>` | seed stage | comma-separated `w,x,y,z` |
| `grid.txt` | hex grid | count line, then `dx dy x y edge_half` per voxel (`hex_grid/io.py:1-9`); the fitter reads the same 5 columns as `y1 y2 xs ys gs` (`fitorientation/io.py:273-305`) |
| `grid_unfilt.txt`, `grid_old.txt` | tomo / mask filter | pre-filter copies |
| `DiffractionSpots.bin`, `OrientMat.bin`, `Key.bin` | diffr-spots | `T×3` f64, `N×9` f64, `N×2` int32 (`fitorientation/io.py:56-86`) |
| `SpotsInfo.bin` | image processing | bit-packed int32 spot mask (`process_images/spots_io.py:90-96`: sized `nDistances * NrFilesPerDistance * NrPixelsY * NrPixelsZ` bits) |
| `<MicFileBinary>` | fitting | 11 f64/voxel at offset `voxel_idx*88` (`fitorientation/output.py:32-56`) |
| `<MicFileBinary>.AllMatches` | fitting | `7 + 4*SaveNSolutions` f64/voxel (`output.py:92`, `parse_mic.py:585`) |
| `screen_cpu.csv` | fitting with `--screen-only` | phase-1 dump (`fit_orientation.py:310`) |
| `<MicFileText>` (**no suffix added**) + `<MicFileText>.AllMatches` `.map` `.map.kam` `.map.grainId` `.map.grod` | `ParseMic` | §9 |
| `<base>_pipeline.h5` | `PipelineH5` | provenance + completed stages |
| `<base>_consolidated.h5` | consolidator | §9c |

Backups the multi-resolution driver leaves: `DiffractionSpots.bin_unseeded_backup` and
friends (`workflows.py:81-97`), `<MicFileBinary>.seeded_backup`, `.unseeded_backup`,
`<SeedOrientationsAll>_Backup`.

### 8e. Multi-phase samples — reduce ONCE, fit once per phase

**The NF path fits one phase per run.** `NumPhases` and `PhaseNr` are forwarded
only to `parse_mic` (`stages.py:618-626`); `diffr-spots` and `fit-orientation`
each read a single `LatticeParameter`/`SpaceGroup`. A two-phase sample therefore
needs two paramfiles and two runs.

**But do not re-run image processing for the second phase.** `SpotsInfo.bin` is
**phase-independent** — verified by reading `ProcessParams`
(`process_images/params.py:29-58`), which parses only I/O, frame indexing and
reduction keys (`BlanketSubtraction`, `DoLoGFilter`, `MedFiltRadius`,
`LoGMaskRadius`, `GaussFiltRadius`, `RawStartNr`, `NrFilesPerDistance`,
`nDistances`). It reads **no** lattice parameter, space group, wavelength, `Lsd`,
`BC`, tilt or `MaxRingRad`. Nothing in the reduction depends on the crystal.

So:

```bash
# phase 1 — does the reduction
midas-nf-pipeline run params_phase1.txt --n-cpus 32 --device cuda

# phase 2 — reuse the SAME SpotsInfo.bin, skip the 1440-TIFF reduction entirely
ln -s <phase1>/LayerNr_1/SpotsInfo.bin <phase2>/LayerNr_1/SpotsInfo.bin
midas-nf-pipeline run params_phase2.txt --n-cpus 32 --device cuda \
    --no-image-processing
```

The two phase runs are otherwise independent and can go on separate GPUs
concurrently.

**What DOES invalidate a shared `SpotsInfo.bin`:** changing any reduction key
above. In particular `BlanketSubtraction` and `DoLoGFilter` are baked in at
reduction time, so changing either forces a regeneration for *both* phases.

**Wanted (not built): a proper multi-phase driver** that reduces once and then
loops the fit over N phases, merging the per-phase `.mic` by confidence into one
multi-phase map. Today that orchestration is manual, and `PhaseNr` in the `.mic`
is only whatever the paramfile declared — it is *not* evidence that a phase
assignment was fitted.

### 8f. Denoise the residual before thresholding — the biggest single lever

On a weak-signal sample this is worth far more than any geometry work. Measured on
`nf_sampleB_htB_s2`, 10 µm loop 0, identical geometry/grid/rings, only the reduction
differing: **voxels at C ≥ 0.9 went 1,424 → 5,186 (3.6×)**, median confidence 0.359 →
0.562, `max C` unchanged at 1.0000 (lab notebook §4d).

```
NLMDenoise 1
NLMH 1.0                # h = NLMH * sigma_MAD
NLMPatchSize 5
NLMPatchDistance 6
BlanketSubtraction 2    # ~0.7 sigma, NOT the ~3 sigma you need without NLM
```

> **`NLMH` is a MULTIPLE OF σ_MAD, and σ_MAD can be exactly 0.** On photon-starved data
> the median-corrected residual is almost entirely exact zeros (20-ID `nfdev_jul26`:
> **99.73 %**), so `σ_MAD = 0`, `h = NLMH · σ_MAD = 0`, and NLM is skipped — historically
> **silently**, so `NLMDenoise 1` became a no-op that nothing in the output revealed.
> It now warns, and you can set an absolute strength in **counts**:
>
> ```
> NLMHAbsolute 1.0        # overrides NLMH * sigma_MAD when > 0
> ```
>
> Verified on synthetic photon-starved data: with `NLMHAbsolute 1.0` a real spot peak is
> preserved exactly (6.00 → 6.00) while an isolated single count is suppressed 167×
> (1.000 → 0.006) — which is also why NLM plus an *absolute* threshold does not
> manufacture spots, while NLM plus a σ-derived one does (§5d).
>
> **Check `σ_MAD` before trusting any σ-scaled setting** (§5d has the one-liner).

NLM is applied to the **median-corrected residual, before** the blanket subtraction and
the clamp. That ordering is the whole point: the fixed-pattern background is already gone,
so what remains is noise plus spots, and the threshold can drop to well under 1 σ.

**This is NOT the `Denoise` key.** That is a separate stage which denoises **raw** frames
*before* median subtraction and needs `MIDAS-NF-preProc` (absent from the beamline env).

Two operational notes:

* Set `BlanketSubtraction` **down** when you enable NLM. Leaving it at 5 throws away most
  of what the denoiser just recovered.
* Changing any reduction key invalidates `SpotsInfo.bin` for **every** phase sharing it
  (§8e). Regenerate once, then re-use.

Sanity check before trusting a new reduction: `max C` must not fall. Denoising before
peak-finding can in principle smear a centroid or merge neighbouring spots, and that shows
up as degradation at the ceiling, not as a lower average.

### 8g. Comparing two runs — never by checksum

`MicFileBinary` is **11 float64 per voxel** (`output.py:32-56`), in this order:

```
0 OrientRowNr  1 OrientID  2 RunTime  3 X  4 Y  5 TriEdge
6 UpDown  7 Eul1  8 Eul2  9 Eul3  10 Confidence
```

Column 2 is a **per-voxel wall-clock time**. It differs on every run, so `md5sum`
of two physically identical reconstructions never matches. Compare field by field:

```python
import numpy as np
NAMES = ["OrientRowNr","OrientID","RunTime","X","Y","TriEdge",
         "UpDown","Eul1","Eul2","Eul3","Confidence"]
a = np.fromfile("run_a.bin", dtype=np.float64).reshape(-1, 11)
b = np.fromfile("run_b.bin", dtype=np.float64).reshape(-1, 11)
for c, n in enumerate(NAMES):
    d = np.abs(a[:, c] - b[:, c])
    print(f"{n:12s} ndiff {int((d>0).sum()):6d}  maxabs {d.max():.6g}")
```

**Read it as float64, not float32.** A float32 view splits each field across two
columns and makes one changed `RunTime` look like two changed physical
quantities — including something that reads convincingly as a confidence shift
of 0.35. That misread cost real time; do not repeat it.

Reference: the `screen()` dtype rework (float intermediates → `bool`/`int32`) was
validated exactly this way on a 5046-voxel grid — every field bit-identical,
`RunTime` the only difference.

### 8h. `screen()` memory — chunking and its knob

The vectorised path builds `(V, T, 3)` tensors, where `T` is the **total**
simulated-spot count over every candidate orientation (~3×10⁷ for a cubic seed
list). One voxel already costs `T*3*itemsize`; a full grid is terabytes
(5046 × 3.02×10⁷ × 3 × 4 B = **1704 GiB**, the allocation that actually failed).
It only ever worked before because calibration runs have `V = 1`.

Voxels are therefore processed in chunks sized from free device memory. Override
with:

```bash
MIDAS_NF_SCREEN_VOXEL_CHUNK=<n>     # voxels per chunk; omit to auto-size
```

Auto-sizing is the right default. Forcing it **too large OOMs**: on a 47 GiB
A6000, `MIDAS_NF_SCREEN_VOXEL_CHUNK=64` died trying to allocate a further
21.62 GiB. Results are independent of the chunk size — verified by comparing runs
at a fixed forced chunk (identical in every field but `RunTime`).

> **COARSENING `GridSize` DOES NOT SPEED UP `screen()`.** This is counter-intuitive
> and it wastes real hours. `screen()` builds a `(T, P, Q)` tensor where `P`, `Q` are
> each voxel triangle's bounding box **in detector pixels** (`screen.py:229-230`), and
> the triangle side is `EdgeLength / px` with `EdgeLength` defaulting to `GridSize`.
> So per-voxel cost grows as `GridSize²` while the voxel count falls as `1/GridSize²`
> — **the product is roughly constant.**
>
> Measured on `nfdev_jul26`, same data, same geometry:
>
> | run | voxels | `GridSize` | `screen` |
> |---|---|---|---|
> | step4_std | **9038** | 4 | **7900 s** |
> | step4_std_g6 | **6676** | 6 | **8265 s** |
>
> **Fewer voxels, MORE time.** Choosing a coarse grid "to make it faster" is a null
> optimisation and costs resolution for nothing.
>
> **What actually controls cost is the MASK AREA.** For a fixed masked region the
> `screen` cost is ~independent of pitch, so **use a fine grid and shrink the mask**
> (§10e) — resolution is close to free, empty space is not.

Performance is *not* the problem and never was: 5046 voxels × 4 distances took
`screen=697.22s nm_batched=8.03s writeback=23.45s` on one A6000, against ~748 s
for the C reference scaled to the same grid. Memory was the only defect.

### 8j. Omega binning — `SumFrames`, and measure the spot width first

A spot spans a finite ω range. Sampling finer than that splits its photons across
several frames, each carrying the FULL background, so each frame is harder to threshold
than the spot really is. `SumFrames N` sums N consecutive RAW frames before the
reduction.

**Measure the ω width before choosing N** (`recon/omega_width.py`): on
`nf_sampleC_htA_s2` at a 0.1° step the profile is 1.00 / 0.69 / 0.69 at 0, ±1 frame —
**FWHM 0.30°**, so N = 3. Summing beyond the spot width adds background to a fixed
signal and SNR *drops* as √N.

Expected gain is **not** √N. The profile is peaked, so summing 3 gathers 2.38× the
peak-frame signal against 1.50× the noise (measured σ_MAD 2.965 → 4.448) ≈ **1.6×**.

**`SumFrames` is INTERNAL — every other key stays in RAW units.** The parameter file
describes the experiment as performed: `NrFilesPerDistance` is the **raw** image count
per distance, `OmegaStep` is the rotation between **raw** images, `EndNr` is optional.
Changing `SumFrames` is a one-line edit and nothing else moves.

```
SumFrames 3
NrFilesPerDistance 1800         # RAW count — NOT divided by SumFrames
OmegaStep -0.1                  # RAW step  — NOT multiplied by SumFrames
                                # EndNr: omit it; the pipeline derives and logs it
```

The code derives the rest at the single place that needs it: the fit uses
`omega_step_raw × SumFrames` and `NrFilesPerDistance // SumFrames`
(`midas_nf_fitorientation/params.py:186-221`), and the reduction reads
`NrFilesPerDistance` raw files per distance on a stride independent of `SumFrames`
(`process_images/io.py:29-35`, `process_images/params.py:152-166`). `SumFrames` must
divide `NrFilesPerDistance`, enforced with a named error (`params.py:167-175`).

> **This convention INVERTED on 2026-08-04** (`a7c50926`, `60dcc94c`). It used to be the
> other way round — you restated all three keys in post-sum units and the pipeline
> rewrote the file. **Writing post-sum values into the current code is silently wrong.**
> With `SumFrames 3` and `NrFilesPerDistance 600`, the reduction reads the first 600 of
> the 1800 raw images — the first 60° of the sweep — and sizes its output at 600//3 = 200
> frames; the fit independently derives the same 200 and an ω step of −0.3 × 3 = −0.9°,
> so it believes those 200 frames span 180°. The two stages **agree**, no size error is
> raised, and every spot is assigned the wrong ω. Guarded by
> `midas_nf_preprocess/tests/process_images/test_sum_frames_internal.py`.

`process-images` measures the ω width itself and logs the `SumFrames` it implies
(`process_images/pipeline.py:471-506`) — read that line rather than guessing N.

### 8k. How low can `BlanketSubtraction` go — measure, do not guess

`BlanketSubtraction` is applied to the **NLM-denoised** residual, so what matters is
σ of the *denoised* frame, not the raw one. Measured on `nf_sampleC_htA_s2`:
raw σ_MAD **2.965 → 0.282 after NLM** (10.5×). So `BlanketSubtraction 2` sits at
**7.1σ** — far more conservative than it looks.

Ladder (`recon/threshold_floor.py`, 5 frames), using isolated single pixels as the noise
indicator — a real spot cannot be 1 px:

| thr | × σ_nlm | spot-like ≥3 px | isolated singles |
|---|---|---|---|
| 2 (typical) | 7.1σ | 3,602 | 2,603 |
| 1 | 3.5σ | 5,359 (+49 %) | 5,370 (+106 %) |
| 0.5 | 1.8σ | 28,989 | 54,282 |

Singles stay flat from 4 down to 1.5 then explode — the floor is between 1.0 and 0.5.

> **Use `BlanketSigma`, not `BlanketSubtraction`.** This whole section reasons in σ while
> naming a key that could not express it: `BlanketSubtraction` was parsed as an **int**,
> so on NLM-denoised data — where σ_MAD of the residual is ~0.27 counts — the smallest
> legal value (1) is already 3.7σ and **nothing below that could be written at all**.
> Fixed in `4e90be80`: `BlanketSubtraction` is now a float, and
>
> ```
> BlanketSigma 3.5        # threshold = BlanketSigma × σ_MAD of the POST-denoise residual
> ```
>
> is the transferable form — measured **per layer**, and it overrides `BlanketSubtraction`
> when set. An absolute count does not carry between reductions: a 14-configuration
> catalog on Ce-5%Y found every good reduction at **~3.5σ however it got there**, while one
> fixed `BlanketSubtraction 2` was 7.5σ unsummed and 3.6σ summed on the same sample. The
> numbers in the table above are that key's σ equivalents — read them as σ, set them as σ.

`process-images` also logs an ω-persistence diagnostic that advises a `SumFrames`
**direction** (§8j). Its floor is calibrated on one dataset and is explicitly *not*
threshold-independent — at 7.5σ it recovers the measured-best `SumFrames 3`, at 3.5σ the
same data gives 5. The log says so in the code itself; **do not read that number as
calibrated.**

**Do not raise `NLMH` to compensate.** Tested 1.5 and 2.0: σ barely moves
(0.282 → 0.270 → 0.269) while spot-like components drop 35–42 %. 1.0 is the operating
point.

**Lowering the threshold inflates confidence mechanically** — thr 1 lights 2.35× the
pixels, so a simulated spot is likelier to hit a lit pixel by luck. Judge the change by
the number of orientation-coherent grains, never by confidence.

### 8l. Structure factors — stop counting reflections that cannot exist

Space-group extinction rules do not see **basis-dependent** extinctions. Declare the
unit-cell basis and the generator computes |F|² per reflection:

```
PhaseAtom <El> 0.0 0.0 0.0
PhaseAtom <El> 0.3333333333 0.6666666667 0.25   # a DHCP cell, hP4: sites 2a + 2c
DropForbiddenReflections 1
ConfidenceMetric filtered                       # raw | filtered | weighted
```

(`<El>` is the element symbol; the worked example below is a DHCP/FCC polytype pair.)

Measured on `nf_sampleB_htB_s2`: the DHCP polytype has **126 of 736 reflections with
|F|² = 0**, capping FracOverlap at **0.829**; the FCC parent has none, cap 1.000 — which is
why this never showed up on a single-atom cell. Dropping them took max confidence
**0.4938 → 0.5962**
(predicted 0.596) and voxels above `MinConfidence 0.5` from **0 → 213**.

Affects any phase with a non-trivial basis: HCP, DHCP, intermetallics, oxides.
**Maps made before and after are not numerically comparable for those phases.**

- Omit `PhaseAtom` → output byte-identical to before.
- `DropForbiddenReflections` filters `hkls.csv` itself, so it fixes the SEARCH too —
  every downstream stage reads that one file.
- `ConfidenceMetric weighted` changes the *scale* of confidence: on the DHCP map it
  moved the median 0.397 → 0.727 and lifted the sham null 0.097 → 0.148. Re-tune
  `MinConfidence` before adopting it; `filtered` is the safe driver.
- **No Lorentz factor**, deliberately — see lab notebook §11c.

### 8m. DHCP-scale phases need a big GPU

A hexagonal cell generates far more of everything than a cubic one. For the DHCP polytype at
`MaxRingRad 1400`: **736 reflections and 486,755 seeds**, against fcc's 228 and 243,129
— roughly 8.7× the forward-model footprint. That **OOMs a 47 GB A6000** in
`calc_bragg_geometry` before fitting a single voxel. Run it on an H200 (143 GB), or cut
`MaxRingRad`. An OOM here looks exactly like "the phase is absent" — see lab notebook §10a.

### 8i. Obs-volume memory — check which paths are packed BEFORE planning a fit

`ObsVolume.from_spotsinfo` has a `packed` flag, and **the four fit entry points do not
agree on it**. Dense costs `nDistances · nFrames · NrPixelsY · NrPixelsZ · 4` bytes;
packed costs one **bit** per pixel, i.e. 32× less.

| entry point | `packed` | source |
|---|---|---|
| `midas-nf-fit-orientation` | **True** (v0.4 default) | `fit_orientation.py:284-292` |
| `midas-nf-fit-parameters` | `False` — dense | `fit_parameters.py:85-92` |
| `midas-nf-fit-multipoint` (**always** soft) | `False` — dense | `fit_multipoint.py:138-145` |
| `midas-nf-pipeline refine-params --multi-point --objective hard` | **True**, uint8 | `fit_multipoint.py:519-525` |

> **`midas-nf-fit-multipoint` has NO `--objective` flag.** Its CLI takes only
> `params.txt [nCPUs]` and unconditionally calls `fit_multipoint_run`, the **soft, dense**
> path (`cli.py:159-186`). The hard/packed path is reachable **only** through the pipeline:
> `midas-nf-pipeline refine-params --multi-point --objective hard`
> (`midas_nf_pipeline/cli.py:207-217`, where `hard` is the default). Reaching for the
> obvious-looking console script gets you the one that cannot run on a large detector.

At 1-ID (2 × 1440 × 2048²) dense is ~56 GiB — painful but survivable on a big node, which
is why this went unnoticed. On the 20-ID Oryx (3 × 1440 × 5320 × 4600) dense is **423 GB**
and packed is **13.2 GB**:

- `fit-orientation` — fine.
- **`fit-parameters` cannot run at all.** Do single-voxel parameter optimisation with
  `midas-nf-pipeline refine-params --multi-point --objective hard` and a **single**
  `GridPoints` row: it optimises the same hard FracOverlap with the same parameter layout,
  over the packed volume. "Multi-point" with one point is the supported route to a
  single-voxel refinement on a big detector.
- The **soft** multipoint objective is likewise unusable on a large detector. It is not
  equivalent to the hard one anyway (lab notebook §3b).

Rule of thumb: `dense_GB = nDistances · nFrames · NrPixelsY · NrPixelsZ · 4 / 1e9`.
Compute it before choosing an entry point.

---

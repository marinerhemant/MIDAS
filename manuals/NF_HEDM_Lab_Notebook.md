# NF-HEDM Lab Notebook — 1-ID `pokharel_jul26` + 20-ID `nfdev_jul26`

**Companion to `NF_HEDM_Handbook.md`.** The handbook says what to do; this records what
was actually found, how it was measured, and what turned out to be wrong. They are kept
apart on purpose: the handbook has to stay short enough to follow, and this has to stay
honest enough to stop a refuted idea coming back.

**§1-§6 are the 1-ID campaign. §7 is the 20-ID HT-HEDM campaign** (`nfdev_jul26`), a
different beamline, detector, acquisition stack and file format — read §7 before assuming
any 1-ID number or convention carries over.

Datasets throughout: APS 1-ID, NF detector 2048², px 1.48 µm, 95.0000 keV.
`Au5_cubes_nf_96keV` (gold calibrant, 4 distances, ω step 0.25°) and
`nf_Ce_ht525_s2` (cerium, 2 distances DetZ 7/9, ω step 0.1°, double exposure).
`§n` without a qualifier means a section of *this* file; handbook sections are named
as such.

**Read §5 before re-opening any question here** — four attractive claims are recorded
there as retracted, each with the measurement that killed it.

---

## 1. What this campaign established

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | The Python multi-resolution path had never run end to end | FIXED (10 defects) | §2 |
| 2 | `GridPoints` parser was off by one on every field | FIXED | §2, defect 10 |
| 3 | `label_components` was 112× slower than scipy for identical output | FIXED | §3a |
| 4 | Multipoint's soft surrogate does not track the hard fraction | FIXED (hard path added) | §3b |
| 5 | Geometry A is at a local optimum: refinement moves it 0.17 µm | ESTABLISHED | §4a |
| 6 | Confidence 1.0 is a plateau, not a unique geometry | VERIFIED NEGATIVE | §4b |
| 7 | `-multiGridPoints` cannot break the degeneracy on a single-crystal calibrant | VERIFIED NEGATIVE | §4b |
| 8 | Re-seeding a refinement ratchets the tilts ~1°/pass | VERIFIED NEGATIVE | §4b |
| 9 | ~~No detectable β-Ce (DHCP)~~ | **RETRACTED — the run CRASHED** | §10a |
| 10 | NLM on the median-corrected residual gives 3.6× the voxels ≥0.9 | ESTABLISHED | §4d |
| 11 | The reconstruction is spot-detection-limited, not geometry-limited | ESTABLISHED | §4a vs §4d |
| 12 | Spots span 0.30° FWHM but were sampled at 0.1° — `SumFrames` recovers ~1.6× | ESTABLISHED | §9a |
| 13 | `BlanketSubtraction 2` is 7.1σ of the DENOISED residual, not 0.7σ of the raw | ESTABLISHED | §9b |
| 14 | Raising `NLMH` above 1.0 destroys signal for no noise gain | VERIFIED NEGATIVE | §9c |
| 15 | Ce5Y is fine-grained; refining the pitch FINDS grains where the control MERGES them | ESTABLISHED | §9d |
| 16 | β-Ce IS present, at the polytype orientation of its parent γ grain | PROVISIONAL | §10 |
| 17 | 126 of 736 dhcp reflections are forbidden and capped confidence at 0.829 | ESTABLISHED | §11 |
| 18 | The rotation Lorentz factor CANCELS for per-frame threshold detection | ESTABLISHED | §11c |

Finding 11 is the through-line. A full geometry refinement was worth **+0.005**
FracOverlap; changing the *reduction* was worth **3.6× the high-confidence voxels**.
Effort spent on the geometry after §4a was effort spent in the wrong place.

---

## 2. Defects fixed

All ten produced wrong numbers or a crash, none produced a useful error message
until it was made to.

| # | Component | Defect | Commit |
|---|---|---|---|
| 1 | `run_diffr_spots` | Namespace missing `hkls_csv`/`seeds`; and they must be passed EXPLICITLY — `DiffrSpotsParams` resolves defaults against `data_directory` while hkls.csv is written to the per-layer `resultFolder` | b95c38c0 |
| 2 | `run_image_processing` | Namespace missing `all_layers`/`layer_nr`/`output`, AND looped over distances — each call writes the whole `SpotsInfo.bin`, so only the last distance survived | b95c38c0 |
| 3 | `run_tomo_filter` | Passed a path where an `(N,5)` tensor was wanted, wrong kwarg name, treated a `(kept, mask)` tuple as an output path | b95c38c0 |
| 4 | both seed stages | `write_seeds_csv(path, quats)` — arguments SWAPPED; FF path also passed `list[GrainOrientation]` where an `(N,4)` tensor was wanted | b95c38c0 |
| 5 | `cmd_refine_params` | `row_nr=` against a positional `voxel_idx` | b95c38c0 |
| 6 | multi-res loop | `SeedOrientationsAll` checked as a KEY then `copy2`'d as a FILE that nothing creates | b95c38c0 |
| 7 | multi-res loop | Loop 0 wrote `<base>.mic` while downstream expected `<base>.0.mic` — `MicFileText` updated on disk but `run_parse_mic` reads the in-memory dict | b95c38c0 |
| 8 | multi-res loop | A phase that indexes NOTHING crashed instead of stopping cleanly | d231fdf3 |
| 9 | `TanhBox.perturb` | CPU `torch.Generator` against a CUDA `randn` — multi-start had never run on GPU | d231fdf3 |
| 10 | `GridPoints` parser | **Off by one on every field** — see below | d231fdf3 |

### Defect 10 — the one that mattered

A `GridPoints` line is the key followed by a raw 12-column `.mic` row. The C
(`FitOrientationParametersMultiPoint.c:696-698`) parses it with

```c
sscanf(aline, "%s %s %s %s %lf %lf %s %lf %lf %lf %lf %s %s", ...)
```

whose **first `%s` consumes the word `GridPoints` itself**, so its `xc` is line-token 4.
Python's `args` list starts *after* the key, so `args[i]` is line-token `i+1` — the C's
indices are one too high there. The parser used them verbatim, under a comment claiming
it "matches the C sscanf format exactly". It read:

```
xc     <- Y
yc     <- TriEdgeSize
eul1-3 <- (Eul2, Eul3, Confidence)     # a confidence used as a Euler angle
```

Mean hard FracOverlap at the seed geometry, same 12 voxels:

| | value |
|---|---|
| C reference | 0.8515392717 |
| Python, before fix | **0.0026065837** |
| Python, after fix | 0.8729672829 |

**Method note.** This was only findable because the C was run as a reference. Without
`Original val: 0.85154` to compare against, 0.0026 looks like a pessimistic score rather
than a broken one. Any future "the Python disagrees with the C" question should start the
same way.

**The shared beamline env still has the bug (checked 2026-08-01).**
`/home/beams12/S1IDUSER/opt/envs/midas/.../midas_nf_fitorientation/params.py` reads
`args[4,5,7,8,9,10]`; the fixed tree reads `args[3,4,6,7,8,9]`. Those are ordinary
`site-packages` copies at 0.3.2, not editable installs, and `MIDAS_canonical` there is on
an unrelated HEAD without `b95c38c0`/`d231fdf3`. Any multipoint refinement run in that env
is invalid until it is reinstalled — handbook §1. Handbook §10h also documented the broken
indices as if they were correct; fixed 2026-08-01.

---

## 3. Performance findings

### 3a. `label_components` — 112× for identical output

It was a hand-written torch label-propagation: seed every foreground pixel with its flat
index, then 3×3 min-filter until stable, with an `(new != old).any()` convergence test
each pass. Two compounding problems — it needs ~component-diameter passes, and on CUDA
that `.any()` is a **device→host sync every iteration**. Each frame became hundreds of
tiny kernels each followed by a stall, which is why the GPU sat at 0% while one core spun.

Labels are integers and the caller is already inside `torch.no_grad()`, so there was never
an autograd reason for it to be in torch.

| | per 2048² frame |
|---|---|
| torch propagation | 1319.9 ms |
| `scipy.ndimage.label` | **11.8 ms** |

Verified equivalent, not merely similar: same component count, same foreground mask, and
an **identical partition of pixels into components**. 8-connectivity preserved with an
explicit 3×3 structure (scipy defaults to 4-way, which would silently merge fewer blobs).

Image reduction of a 3600-frame scan: ~80 min → 20 min. The remaining cost is GPFS I/O
plus the frame loop, not labelling.

### 3b. The soft overlap surrogate — two bugs, and the wrong design

Multipoint optimised a differentiable Gaussian-splat surrogate with L-BFGS. The C
optimises the **hard FracOverlap** directly with derivative-free NLopt
(`NELDERMEAD`/`CRS2_LM`), so the objective never needed to be differentiable at all.

Two genuine bugs in the surrogate (both verified on synthetic volumes, both fixed):

* The blur kernel was **sum-normalised**, i.e. mass-conserving. On a sparse binary volume
  that collapses a lit pixel's value to ~1/(2πσ²): a spot sitting exactly on a lit pixel
  scored **0.0050** at σ=1.5 instead of ~1.0. Now peak-normalised.
* The `sigma_px <= 1.0` branch used **bare trilinear** sampling on the docstring's claim
  that bilinear is "roughly Gaussian-with-σ-1". It is not — bilinear is a tent reaching
  zero one pixel out, so a spot in the *correct* pixel scored **0.0625** at half-a-pixel
  offset while `hard_fraction` was 1.0. `auto_sigma_px` clamps at exactly 1.0, so typical
  NF configs took this branch every time.

Blurring is now done **in place, chunked**: the out-of-place version held the original and
the result at once — 112 GiB on a 56 GiB volume, OOM even on a 140 GiB H200.

`fit_multipoint_hard_run` (`--objective hard`, now the default) optimises the same
quantity the C does, with the same parameter layout, and uses the **packed** obs volume
(~1.9 GB vs 56 GiB dense) — which removes the memory wall entirely.

### 3c. Multi-GPU sharding

The fitter always supported block splitting (`grid.slice_block`), but
`midas-nf-pipeline run` never used it — one process, one GPU, the rest idle. `--fit-gpus
0,1` fans disjoint voxel blocks out, one process per GPU, in every loop.

This required making `MicWriter` concurrency-safe: it opened `mode="w+"` and then zeroed
the whole file, so every worker would have wiped the others' records. The parent now
pre-allocates once and workers open `r+`.

### 3d. Threaded frame loop

The per-frame reduction work is embarrassingly parallel but was serial, and `n_cpus` only
reached `torch.set_num_threads` on CPU — so on `device=cuda` the cores sat idle. Threads
suffice because skimage's fast NLM and scipy's labeller both release the GIL (measured
**3.96× on 4 threads**); no 16 MB frame is ever pickled.

**Device trap:** threading multiplies the per-frame GPU temporaries by the worker count.
`spatial_median`'s im2col alone OOM'd a 47 GB card at 64 workers. With NLM on the loop
runs on CPU (NLM already forces a host round-trip); with NLM off concurrency is capped.

---

## 4. Scientific findings

### 4a. Geometry A is already at an optimum

Multipoint refinement against the hard FracOverlap, 12 voxels, converged
(142,977 evaluations, 2.2 h):

| | seed | refined | Δ |
|---|---|---|---|
| Lsd₀ | 7228.5849 | 7228.4187 | **−0.17 µm** |
| Lsd₁ | 9229.7096 | 9228.0814 | −1.63 µm |
| BC₀ | (996.7168, 37.9415) | (996.5724, 38.0212) | −0.14, +0.08 px |
| tilts | 0.7882 / 0.6834 / 0.0827 | 0.7866 / 0.6921 / 0.0802 | <0.01° |
| FracOverlap | 0.8730 | 0.8776 | +0.0046 |

**Nothing here is worth re-reconstructing for.** The value of this run is the negative:
it says stop working on the geometry.

### 4b. Three verified calibration negatives

Established directly on Au5, not inferred:

1. **Confidence 1.0 is a plateau, not a unique solution.** Single-voxel refinements seeded
   at `ty` = 0.559, 1.507 and 2.622° all converge to confidence **exactly 1.0000**.
2. **`-multiGridPoints` does not break the degeneracy on a single-crystal calibrant.**
   Seeded from `ty`=0.559 it converged to 0.683 (mean 0.9562); from `ty`=2.622 to 2.985
   (mean 0.9753) — **2.3° apart in `ty`, ~48 µm in `Lsd`**, both looking excellent. All
   voxels of a calibrant cube share one orientation, so N voxels give one grain's
   constraint. It *does* work on a polycrystal (§4a used 12 voxels across 6 grains).
3. **Never re-seed a refinement with its own output.** `TiltsTol` is interpreted relative
   to the current seed, so feeding the output back ratchets the tilts ~1°/iteration;
   `ty` walked to 4.6° with confidence high throughout.

### 4c. ~~No detectable β-Ce — a controlled null~~ — SUPERSEDED, see §10

**Do not use this section.** Its conclusion is refuted (§10a) and its numbers below have
**no surviving provenance**: no `.mic`, no run directory, nothing re-derivable. The only
DHCP run left on disk crashed before fitting a voxel. Kept only so the reasoning that
led here can be audited.


Both phases were run against the **same** `SpotsInfo.bin` (verified phase-independent:
`ProcessParams` reads no lattice, space group, wavelength, `Lsd`, `BC`, tilt or
`MaxRingRad`), same geometry, same ω, same grid.

| | max C | median C | voxels ≥0.5 |
|---|---|---|---|
| γ-Ce FCC (a=5.1596, SG 225) | **1.000** | 0.359 | 7,338 |
| β-Ce DHCP (a=3.6671, c=11.805, SG 194) | **0.191** | 0.052 | **0** |

Identical ceiling of 0.191 at Rsample 200 *and* at the full Rsample 550, so it is not an
artifact of sampling only the centre. The FCC control rules out geometry, mirrored ω,
ring cap and reduction as explanations.

**Caveat that must stay attached:** this is at a **10 µm voxel pitch**. β-Ce in grains well
below that is under the detection limit. "Not detected at 10 µm" ≠ "not present". The
operator's hypothesis that the DHCP grains may simply be very small is untested.

### 4d. NLM on the median-corrected residual — the biggest single win

Denoise the residual *before* thresholding, so the threshold can drop to ~0.7σ instead of
~3σ. Distinct from MIDAS's existing `Denoise` stage, which denoises **raw** frames before
median subtraction.

10 µm loop 0, identical geometry/grid/rings, **only the reduction differs**:

| | baseline (thr 5) | NLM + thr 2 |
|---|---|---|
| median C | 0.359 | 0.562 |
| voxels ≥0.5 | 7,338 | 9,340 |
| voxels ≥0.7 | 4,648 | 7,724 |
| **voxels ≥0.9** | 1,424 | **5,186 (3.6×)** |
| max C | 1.0000 | 1.0000 |

Two checks that make this a real gain rather than a bigger number:

* **max C stayed exactly 1.0000.** The plausible failure mode of denoising before
  peak-finding is smearing a centroid or merging neighbours, which would degrade the
  ceiling. It didn't.
* **Grain topology unchanged** — same grains, same places, filled in rather than invented.

Frame-level, per the cosmic-ray audit (handbook §5b: real spots are blobs ≥30 px, noise is
1-px specks):

| | area in ≥30 px blobs | 1-px specks |
|---|---|---|
| raw @ thr 10 | 8,127 | 2,023 |
| raw @ thr 5 | 19,065 | 91,457 |
| **NLM @ thr 2** | **42,740** | **1,167** |

5.3× the real spot area of the old production threshold, with *fewer* specks.

---

## 5. Open questions, and claims that were RETRACTED

### Retracted

**R1. "Raising `MaxRingRad` 1000→1400 caused the confidence regression."** It did not. The
regression (max C 1.000 → 0.804, voxels ≥0.5 6668 → 1875) was measured on the *failed*
low-signal 0p5deg dataset. The **same settings** on the 0.1° double-exposure data give max
C 1.000 with 1,424 voxels ≥0.9. The cause was signal, not ring count. The argument that
extra reflections enter the confidence denominator unmatched is plausible and still
unproven — it was never isolated, because two things were changed at once.

**R2. "`EdgeLength` must equal `GridSize`; setting it smaller breaks the grid."** Wrong.
`MakeHexGrid.c:23-58` shows `EdgeLength` is an independent, deliberately supported knob:
positions come from `GridSize`, `EdgeLength` sets only the triangle. `EdgeLength 1` on a
coarse grid means small probe triangles, which is intentional. Removing it (making the
triangles 10 µm) is what caused a 94 GiB-per-voxel OOM. **The triangle count never
changed.**

Confirmed in the maintained Python (2026-08-01, by reading `hex_grid/grid.py:97-153`):
every lattice term — `nr_hex`, `nr_row_elements`, `x`, `ht_triangle` — contains
`grid_size` only, while `edge_length` appears solely in `edge_half` and the sub-triangle
offsets `xt1`/`xt2`. So `EdgeLength` provably cannot change the voxel count or positions.
What it *does* propagate into is `TriEdgeSize` (`.mic` col 5), hence `mic2grains` areas and
its neighbour-merge threshold — the operational consequences are tabulated in handbook
§10e. Handbook §10e used to carry the retracted claim verbatim while the traps table
pointed at it as the retraction; fixed 2026-08-01.

**R3. "`screen()` results depend on the voxel chunk size."** Two runs at the *same* forced
chunk gave different md5s, which looked like chunk-dependence. It was the per-voxel
`RunTime` field: read as float64, every physical field is bit-identical. The float32 read
that suggested otherwise split one field across two columns.

**R4. "The GPU sitting at 0% means the job has stalled."** Twice this was a live CPU-bound
stage (`label_components`, then `load_tiff_stack`). py-spy distinguishes them in one call;
GPU utilisation does not.

### Open

**O1. Python multipoint objective reads ~2.5% high** — 0.8730 vs the C's 0.8515 on
identical inputs. Suspected cause: the C passes the voxel **triangle vertices**
(`XGrain[3]`/`YGrain[3]`, built from `gSze` via `ysmall = gSze/(2√3)`) and rasterises
them; the Python path uses the centroid only. Falsifiable prediction: `screen` scores a
spot as the *fraction of its triangle's pixels that are lit*, so adding the triangle
should pull the Python value **down** toward the C.

**O2. Python multipoint optimiser is weaker** — +0.0046 against the C's +0.0119. The C
runs NM → CRS2 → NM → CRS2 → NM per iteration; the Python now runs a local→global→local
ladder with `differential_evolution`, which closed part but not all of the gap. Likely
throughput-bound (see O3), and low priority given §4a.

**O3. Multipoint objective is not batched.** `forward_batched_grains` was tried and
returned a **negative** mean fraction: it hands back `valid` as `(B,K,M)` but
`y_pixel`/`z_pixel` as `(D,B,K,M)`, and `hard_fraction` does not compose with that
pairing. Needs `hard_fraction` taught the batched layout **with a parity check against the
per-voxel loop** — that check is what caught the bug.

**O4. Defect 8's sibling.** A phase that indexes nothing now stops cleanly, but the DHCP
re-run on the good 0.1° dataset has not been done. The null in §4c rests on the failed
dataset for the *phase* question and on the 0.1° data only for FCC.

**O5. Reduction is now I/O-bound.** With labelling 112× faster and the loop threaded, the
remaining ~20 min for 3600 frames is GPFS reads plus the temporal median. Not yet attacked.

---

## 6. Measurement ledger

| Claim | How it was measured | Status |
|---|---|---|
| GridPoints off-by-one | C `Original val` 0.8515 vs Python 0.0026 → 0.8730 after fix | VERIFIED |
| scipy labeller equivalent | identical component count, foreground, and pixel partition | VERIFIED |
| scipy labeller 112× | timed on a 2048² frame with 33k lit px | VERIFIED |
| threading equivalent | 3600-frame reduction BYTE-IDENTICAL to serial (`cmp` clean, 92,196,375 bits) | VERIFIED |
| NLM gain | 10 µm loop 0, identical geometry/grid/rings, only reduction differs | VERIFIED |
| NLM preserves spots | synthetic: noise std 3.063 → 0.686, spot peak 46.1 → 46.1 unchanged | VERIFIED |
| spot-axis tiling | identical results at tile sizes 1/2/3/7 and voxel chunk 1 | VERIFIED |
| `screen()` dtype rework | every field bit-identical on a 5046-voxel grid, `RunTime` excepted | VERIFIED |
| `BoxSize` gate | 0.949153 → 1.000000, matching C exactly; Triton == eager in both states | VERIFIED |
| concurrent MicWriter | two workers writing disjoint rows both survive; old path zeroes everything | VERIFIED |
| shared tilts | gradient 4.0 on each of 3 leaves for 4 distances | VERIFIED |
| geometry A optimal | 142,977-evaluation converged refinement moves it 0.17 µm | VERIFIED |
| β-Ce absent at 10 µm | FCC control on the same SpotsInfo.bin reaches 1.000 | CONTROLLED NULL |
| geometry A vs B | full maps; A chosen on operator judgement, orientations agree to 0.04° | **PREFERENCE, not a measurement** |

---

# 7. 20-ID HT-HEDM — `nfdev_jul26` / `Au_cube` (opened 2026-08-01)

A **different beamline**: Bluesky/ophyd acquisition, tomography-style fly scans, a FLIR
optical camera, and HDF5 in DXchange layout. Almost nothing about the 1-ID file handling
carries over. Data
`/gdata/dm/20ID/HT_HEDM/2026-2/nfdev_jul26/data/or1/Au_cube`, reachable as `s1iduser` on
chutoro. Working dir `~/Desktop/analysis/nfdev_jul26_20id/` (has its own `CHECKPOINT.md`).

## 7a. What the data is

| | |
|---|---|
| Detector | FLIR Oryx ORX-10G-245S8M, **5320 (Y) × 4600 (Z)**, 0.548 µm/px ⇒ FOV 2915 × 2521 µm |
| Container | `exchange/data` (1442, 4600, 5320) uint16, chunks (1,1500,1960), **uncompressed** |
| **Bit depth** | **10-bit stored ×64.** Values are multiples of 64; saturation 65472 = 1023×64. Divide by 64 for ADU |
| ω | `exchange/theta`, −180 → +180.25 step **+0.25°**, monotonic, no duplicates |
| Frames | 1442 = 360.25°; frames 1440-1441 duplicate 0-1 ⇒ **`NrFilesPerDistance` 1440** |
| Energy | **63.314 keV** (Lu K-edge), λ 0.195824 Å — from the `foilA.about` foil-wheel table in the Bluesky log |
| Scan | `nfscan(0.2, nfz_start=7, nfz_end=11, ndz=3, y0=8.04, dy=0.01, y_nlayers=2)` ⇒ 3 distances × 2 layers = 6 files |
| δ | 2000 µm (operator) ⇒ `Lsd` 9000 / 11000 / 13000 µm |

**Chunk padding wastes 44 %** — chunks are (1,1500,1960) on a 4600×5320 frame, so 4×3
chunks cover 6000×5880. File 101.77 GB vs 70.58 GB of payload.

**`ndz` counts distances, not steps.** Established from the log's own API change: at 15:45
the same scan was issued as `nfz0=7, dz=2, ndz=2` and then, two minutes later under a
renamed API, as `nfz_start=7, nfz_end=9, ndz=2`. So `ndz=3` over 7→11 is
`linspace(7,11,3)`.

**Metadata is NOT self-contained.** ω is in the HDF5; **detector distance, energy and pixel
size are not** — the HDF5 root carries only areaDetector camera attributes. They come from
`data/metadata/nfdev_jul26/.logs/ipython_logger.log`. There is no `~/new_data` equivalent.

**All darks and flats are empty.** In-file `data_dark` / `data_white` / `data_white_post`
are all-zero placeholders, *and* the separate `_dark_before` / `_dark_after` files are
all-zero too (max = 0 over 24.5 M px). No dark and no flat-field exists for this dataset.
Not fatal — the frame mean is 0.0125 ADU — but there is nothing to flat-field with.

`NF_Au_cube_000594.h5` is an **aborted first attempt**: no `theta`, different `y0`/`dy`.
Do not mix it with the `NF_Au_cube2_0005xx` series.

## 7b. The reduction regime is photon-starved — "0.7 σ" does not apply

Measured on the median-corrected residual (a 400×5320 strip of file 595):

| | |
|---|---|
| fraction exactly zero | **99.734 %** |
| **MAD** | **exactly 0** |
| nonzero pixels at 1 / 2 / 3 / 4 / ≥5 ADU | 4954 / 138 / 18 / 2 / 41 |

So this is near-counting data, and **a MAD-based σ is degenerate**. Handbook §8f's
`BlanketSubtraction ≈ 0.7 σ` cannot be applied as written: `0.7 × σ` collapses onto the
code's σ floor and admits the entire single-count floor as "signal". The threshold must be
**absolute**. Operator's production value: **2 counts, after median subtraction and NLM.**

Second-order trap: at a sub-ADU threshold NLM actively *manufactures* false spots, because
it smears an isolated single count over a patch and the result clears a 4-px minimum-area
cut. NLM plus an absolute threshold is fine; NLM plus a σ-derived threshold is not.

## 7c. Direct beam and beam centre — measurable here, unlike 1-ID

The direct beam **is on the detector** (no DetZBeamPos scan needed, and none exists):
a stationary horizontal stripe, **peak row 4478**, centroid 4479.875, **FWHM 11 rows =
6.0 µm** (that is the layer thickness). Illuminated band ~1670 px ≈ 915 µm wide, which is
slit-defined — **its centre is NOT `ybc`** (handbook §6e).

`ybc` came instead from the sample shadow, two independent ways that agree to 3.5 px
(1.9 µm):

| method | ybc (raw array column) |
|---|---|
| stationary on-axis cube's shadow centroid, file 595 | 2621.5 |
| constant term of the moving cube's sinusoid, files 598/599/600 | **2625.0 ± 5.9** |

**The beam is parallel to 0.14 %** — the moving shadow's amplitude is 906.6 ± 1.3 px across
three files that differ only in detector distance. A divergent beam would have magnified it
by tens of percent over Lsd 9→13 mm. Consequence: **shadows cannot give the detector
distance**; only diffraction can.

`tx` from the stripe slope is **weak**: slope 1.302e-3 px/px ⇒ 0.075°, but the fit rms is
2.55 px against only 3.65 px of linear signal, and the centroid wanders 13.5 px
non-linearly. Treat as a bound, |tx| ≲ 0.15°, not a seed.

## 7d. TWO gold cubes, not one

The operator raised this; the data supports it. Two absorbers are present:

| | on-axis | off-axis |
|---|---|---|
| evidence | **stationary** shadow — only possible for an object on the rotation axis | sinusoid, fit rms 6.2 px |
| position | col 2621-2625 (= the axis) | **496.8 ± 0.7 µm** from the axis |
| shadow width | 40.6 µm | 44.0 µm |
| T | 0.622 | ~0.61 |
| **implied Au path** | **62.4 µm** | **65.0 µm** |

Au at 63.314 keV: µ/ρ = 3.944 cm²/g, µ = 0.007606 µm⁻¹ (`midas_hkls.absorption`, which
takes **wavelength in Å, not energy**). Both absorbers give path/width ≈ 1.5, which sits in
the `s` → `s√2` = 1.41`s` range a cube spans between face-on and corner-on. Two Au cubes of
the same size (~45-65 µm), one centred, one offset by ~0.5 mm.

**The on-axis object is confirmed convex, centred and ~4-fold.** Fourier amplitudes of
−ln T versus ω, normalised to the mean (file 595):

| A1 | A2 | A3 | **A4** | A6 | A8 |
|---|---|---|---|---|---|
| 0.001 | 0.079 | 0.003 | **0.100** | 0.081 | 0.074 |

Three independent readings: (i) a *static detector artifact* is excluded, because a defect
would give all A_n ≈ 0 and these do not; (ii) the odd harmonics vanish (A1, A3 are 30-100×
below A4), which is required for a convex body rotating about an axis **through itself**;
(iii) A4 is the largest term, as a square section demands. **Caveat: A4/A2 is only 1.27**,
so "square" is supported but not strongly — a slightly rectangular section or a small tilt
reproduces this too.

The **moving** object's shape is **untested**: it swings out of the ~1670 px beam band at
both extremes, and that clipping injects spurious low harmonics (its A1 = 0.270,
A2 = 0.452 are artifacts, not shape).

**Still open: confirmation by diffraction.** Absorption establishes "two gold-sized
absorbers"; only indexing establishes "two gold single *crystals*".

**Why this matters for calibration:** a single-crystal calibrant fit that silently averages
two cubes returns a wrong geometry, and a 497 µm offset is far outside anything `Rsample`
would normally bound.

## 7e. File → (layer, distance) grouping

The six `Au_cube2` files split **3/3** on every shadow observable, with tight agreement
inside each group. The argument that fixes the meaning of that split:

> The shadow depends on the sample `y` (which slice the 6 µm beam cuts) and **not** on
> detector distance, because the beam is parallel (§7c). Observed: a 3/3 split with
> within-group agreement of 1.3 px in amplitude. Therefore the split is **by layer**, and
> distance varies *within* each group.

⇒ **595, 596, 597 = layer 1** (y = 8.04); **598, 599, 600 = layer 2** (y = 8.05),
distance-minor. Corroboration: `ybc` drifts monotonically inside group 2 — 2617.55 →
2625.47 → 2631.91 — which is the per-distance beam-tilt signature, **β_y/p ≈ 14.36 px /
4000 µm = 0.0036 px/µm** (1-ID measured 0.0078 px/µm, same order).

Layer 2's shadows are shallower and smeared (on-axis 15.3 µm of Au, moving 29.3 µm across a
90 µm width — path *shorter* than width), i.e. that layer's beam grazes the cubes near an
edge rather than cutting through their middle.

**The absolute distance order within a group is NOT yet measured** — it is assumed to
follow the `linspace(7,11,3)` loop order. The spot radial-scaling test is what settles it.

## 7g. First reconstruction — the Y handedness, and why one voxel cannot calibrate here

**Y handedness settled empirically (2026-08-01).** The detector's horizontal convention was
unknown, so the reduction wrote **two** `SpotsInfo.bin` from one expensive pass — one with
the labels as-is (the writer applies the 1-ID flip `y → NrPixelsY-1-y` internally) and one
with the labels Y-reversed, which is exactly the opposite convention. Identical data,
identical geometry, one bit different:

| variant | maxC | medianC | voxels ≥0.5 | `nm_batched` |
|---|---|---|---|---|
| **std** (1-ID flip) | **0.6957** | 0.3043 | **634** | 7.09 s |
| yflip | **0.000000** | 0.0000 | **0** | **0.00 s** |

yflip does not merely score worse — **nothing survives screening at all**. So the 1-ID
convention `ybc = NrPixelsY-1-col_raw`, `zbc = NrPixelsZ-1-row_raw` holds on the 20-ID Oryx
too. Note this was *established*, not assumed: handbook §3h still says do not inherit the
constant, and the right move when it is unknown is to build both masks from one reduction
and let the calibrant decide.

### The single-voxel refinement is QUANTISATION-LIMITED — §7c's recipe does not port

Step 3 (`refine-params --multi-point --objective hard`, one `GridPoints` row,
`NumIterations 3`) returned:

```
Original val: 0.6956521869   ->  Final: 0.7173913121   (+0.0217, 31088 evals, 160 s)
rounds 2 and 3 identical to round 1
Lsd 6139.7 -> 6162.3 (+22.6 um, same shift at all 3 distances)
BC essentially unmoved;  Tilts: tx=ty=tz=0.0000  -- NEVER MOVED
```

The tilts *are* in the parameter vector (`fit_multipoint.py`: `x0[0:3] = tx,ty,tz`, bounds
from `tilts_tol`), so this is not a wiring bug. The cause is in the numbers:

**0.695652 = 32/46 and 0.717391 = 33/46.** That voxel has only **46 predicted spots**, so
the hard FracOverlap is a step function quantised at **1/46**, and the entire refinement
gained exactly **one spot**. A derivative-free simplex on a piecewise-constant objective has
no direction in which to move **12** parameters (3 tilts + 3 `Lsd` + 6 `BC`); it lands on a
plateau tread and stops, which is why rounds 2-3 were bit-identical.

**Diagnostic:** if the reported objective values are ratios of small integers, you are
quantisation-limited, not converged. Print `Original val` and `Final value` to full
precision and look for the common denominator.

**Why only 46 spots here:** the beam sits **121 px from the detector's bottom edge**, so
only the upper part of each diffraction cone is captured — roughly half the azimuthal
coverage 1-ID gets from a centred beam. Spot count per voxel is a property of the
*beamline geometry*, so §7c's "single voxel, one invocation" recipe is not portable; it
worked at 1-ID because that geometry yields enough spots.

**Fix:** more voxels. With V voxels the objective resolution is `1/(46·V)`, so V = 12 gives
~1/552 and the simplex regains traction. This makes multi-point **load-bearing rather than a
polish step** — and it is precisely where the two cubes matter, since §7b(2)'s negative was
that N voxels of ONE grain give one grain's worth of constraint no matter how many.

## 7h. The confidence ceiling was a BEAMSTOP, and a row profile cannot see one

`maxC` sat at **33/46 = 0.7174** and would not move. Per-distance breakdown of the same
voxel (`missing_spots.py`, which reproduces the reported confidence exactly):

| distance | Lsd (µm) | matched |
|---|---|---|
| 0 | 6162 | **33/46 = 71.7 %** |
| 1 | 8162 | 42/46 = 91.3 % |
| 2 | 10162 | **46/46 = 100 %** |

`hard_fraction` does `hits_d.prod(dim=0)` (`obs_volume.py:395`) — a spot counts only if seen
at **all** distances — so the aggregate equals the worst distance and gives no hint which.
**Always break confidence down per distance before theorising.**

### The cause: a circular beamstop, R ≈ 1100-1240 px = 600-680 µm

The rings vanish by **radius**, not by index, and the same radius works at every distance:

| ring | r at d0 | r at d1 | r at d2 |
|---|---|---|---|
| {111} | 938 **blocked** | 1242 weak | 1546 **strong** |
| {200} | 1084 **blocked** | 1435 present | 1787 **strong** |
| {220} | 1538 strong | 2037 strong | 2536 strong |

{200} blocked at d0 ⇒ R > 1084; {111} weakly visible at d1 ⇒ R ≲ 1242. The bound holds in
**pixels** at all three distances, so the stop is fixed in the DETECTOR PLANE, not
subtending a fixed angle from the sample.

Assigning the 13 distance-0 misses to rings by radius: **6 × {111}, 3 × {200}** — all inside
the stop — plus **4 × {220}**, which are outside it and remain **unexplained**. The beamstop
accounts for 9 of 13, not all of them.

**This is why the STRONGEST reflections were the missing ones.** {111} and {200} dominate an
FCC pattern; at the near distance they are the only ones landing inside the stop. An
inner-ring deficit that grows as `Lsd` shrinks is the signature to look for.

### Method failure: five wrong hypotheses, and why the right test was late

Refuted in order: beam halo inflating the median (median = 0.0000 ADU at the miss
positions); a geometry shift at d0 (nearby lit pixels occur at exactly the chance rate —
see below); an opaque block (no dead rows; spots to 1000 ADU at every row); an attenuating
block (raw row profiles flat, no step); ω misalignment of file 595 (best frame shifts
scatter ±6.3).

**The error: I tested for a horizontal BAND when the object is a DISC.** Row profiles and
row occupancy are structurally blind to a stop centred on the beam, because every row keeps
unobstructed columns far from BC. **A radial profile about BC found it immediately.**
Rule: *for anything centred on the beam, profile in RADIUS, not in rows.*

**Chance-rate trap.** "Is there a lit pixel near the prediction?" is meaningless without the
null. With ~3800 lit px per 24.5 Mpx frame the density is 1.55e-4/px, so a ±400 px search
circle contains **~78 lit pixels by chance** and a ±60 px window ~1.75 — which is exactly
what was "found". A first version of the script even labelled this SYSTEMATIC using a
standard-error test. **Compute the chance expectation before interpreting a search radius.**

### A SECOND, separate problem at the near distance — ~10-50× intensity suppression

The beamstop explains 9 of the 13 distance-0 misses. The other 4 are **not** a new
mechanism — they are the tail of a broader effect. Comparing the peak ADU of the SAME
reflection at d0 and d2 for the 30 predicted spots outside the stop:

| | n | d0 peak, median |
|---|---|---|
| hits | 26 | **111.5 ADU** |
| misses | 4 | **3.5 ADU** (threshold is 2) |

and the per-spot `d0/d2` ratio splits into two clean populations:

* **normal**, ratio 0.9-1.6 — 15 spots
* **suppressed**, ratio **0.013-0.08** — 14 spots, i.e. **12-55× weaker at d0 than the
  identical reflection at d2**

Many suppressed spots still count as hits because they clear the 2-count threshold (e.g.
16 ADU at d0 against 428 at d2). The 4 misses are simply the ones pushed under it — 3 sit
at 3-4 ADU; the fourth has 163 ADU in the window but not at the exact predicted pixel
(`hard_fraction` is a single-pixel test).

**Mechanism NOT identified.** Refuted: a beamstop support arm (miss azimuths span 139°,
sd 57.8° vs 47.9° for hits — no exclusion sector); and the acquisition's
`oryx_hdf1 did not capture the correct number of images` warnings, which occur **identically
for every file** (595-600) and are a routine writer-flush artifact, not specific to d0.

This is a **data-quality property of the near-distance acquisition**, not a geometry or
analysis defect. It is why dropping d0 lifts the median confidence from 0.30 to 0.87 —
far more than removing the beamstop-blocked rings does.

### Consequence for the reconstruction

Blocked reflections still enter the confidence **denominator** as misses: `MaxRingRad` sets
only an OUTER limit and MIDAS has no inner radial exclusion. So the ceiling is not a
calibration defect and **no amount of geometry refinement removes it** — confirmed by
multi-point over 12 voxels (objective resolution 1/552) failing to improve on 0.6884 at all.

Fix: `RingsToUse 3 4 5`, dropping {111} and {200}, so the denominator counts only rings
outside the stop at every distance. Preferable to dropping distance 0, which would discard
a third of the geometry leverage.

**Wanted (not built): an inner radial mask.** A `MinRingRad` / beamstop-radius parameter
would let the forward model exclude blocked reflections properly, per distance.

## 7f. Two tests that FAILED — do not repeat them

**F1. Median spot radius does not resolve the detector distances.** Expected ratios
1.000 / 1.286 / 1.571; measured 1.000-1.198 with no three-group structure. Cause: at larger
`Lsd` the high-angle spots leave the detector, so the surviving population is biased toward
small radii and the ratio compresses. **A pooled median over *different* reflections cannot
measure a radial scale.** This was *not* evidence against 7/9/11. Its spot counts were also
invalid — thresholded on RAW frames with no temporal median and no NLM.

**F2. Pairwise nearest-neighbour distance between sparse spot sets has no discriminating
power.** The idea was that two files at the same distance would show near-identical spot
patterns. Measured matrix: every off-diagonal 287-490 px against a different-ω control of
417 px. Diagnosis is arithmetic: for N points placed at random in a `4600×5320` frame the
median nearest-neighbour distance is `0.4699/√(N/A)` = **393 px at N = 35**, mean 418 px.
At 25-49 spots/frame the entire matrix **is** the chance-proximity expectation. Any
NN-matching test at this spot density is measuring nothing.

Method lesson common to both: **compute the random-coincidence expectation of a matching
statistic before running it.** Both failures were predictable on paper.

**F0. The work was done in the WRONG ORDER, and that is the biggest failure here.**
Hard rule 13 / §6a: *BC comes from the direct beam, Lsd comes from spots, in that order.*
What happened instead: an operator-supplied "delta = 2000 µm" was written straight into an
`Lsd 9000/11000/13000` paramfile, and a 100-minute reduction plus a reconstruction were
launched — §8 work started before §6 was finished. The `Lsd` lines were void. Only
`SpotsInfo.bin` survived the mistake, and only because it is genuinely geometry-independent
(§8e: `ProcessParams` reads no `Lsd`, `BC`, tilt, wavelength or `MaxRingRad`).

Root cause was a **terminology collision on the word "delta"**: the handbook's `δ` is the
Lsd offset `L₁ − DetZ₁`, while the number supplied was (most likely) the *step* between
detector positions, `ΔD` = 2000 µm for `nfz` 7/9/11 mm. Only `ΔD` is trustworthy from the
motor; `δ` must be measured. Confusing them is a millimetre-scale error — here ~2.9 mm.
Recorded as a trap in handbook §6i-bis.

**F4. The position-scrambled null was written as a NO-OP — the exact bug the handbook
documents.** §6i:995 says in as many words: *"Permuting whole rows is a no-op … the null
silently re-runs the real analysis."* The first triangulation script did
`dB = dB[permutation]`, i.e. permuted whole rows, and the null returned **contrast 1.00×** —
peak 1.3262 against a real peak of 1.3262, identical to four decimals. It tested nothing.

The reason row-permutation is a no-op here is worth stating because it generalises: the
matcher searches **all** `(i, j)` pairs, so reordering B leaves the *set* of B vectors
unchanged. A null must destroy the *correspondence*, not the *ordering*. The fix is to
permute the two position components **independently**, which breaks the `(z, y)` pairing
while preserving both marginals exactly.

**This is the second time this identical bug has appeared** (the first cost a falsely tight
5.1 µm null scatter at 1-ID). Two occurrences of the same documented mistake is an argument
that the null belongs **inside** the function that reports the number, not with the caller —
a `triangulate()` that runs its own nulls and refuses to return a value when a gate fails
makes the error structurally impossible.

**F5. A dropped minus sign put the tomo mask 571 µm away from the particle.** The
sample-shadow sinusoid's phase was measured as **−35.1°** (consistently, in all six files).
The mask generator was written with `PHI_DEG = 35.1`, so its discs sat at
`±(406.4, +285.7)` µm while the true positions are `±(406.2, −285.5)` µm — **571 µm apart,
against a disc radius of 80 µm.** The off-axis cube was therefore masked *out* of the
reconstruction it was supposed to reveal.

The failure mode is what makes this worth recording: the run would have completed
normally and reported "no off-axis cube found", which reads as **the geometry is wrong**.
A sign error in a mask manufactures a false negative in the physics.

**There are TWO independent binary ambiguities, so FOUR candidates, not two.** The
magnitude `|r| = 496.8 ± 0.7 µm` is measured, but turning the phase into a sample-frame
`(x, y)` needs (i) the ω sign convention and (ii) the detector Y handedness. Each flips a
sign independently, giving `(±x, ±y)`. Masking only the `±(x, y)` pair — a point reflection
— covers half the possibilities. Mask all four; the cost is two extra discs (kept fraction
2.43 % → 4.05 %) and it removes the guess entirely.

**How it was caught:** by writing `beam_calib/shadow.py` and testing it against a
two-particle synthetic with known truth. The module returned
`position_candidates_um = [(406.2, −285.5), (−406.2, 285.5)]` and the disagreement with
the hand-built mask was immediate. The one-off script had no truth to check against and
would not have caught it. This is the argument for the module in one example.

**F6. R2 was documented and then not applied — every run of the campaign omitted
`EdgeLength`.** This is the most expensive mistake in the session and it was
self-inflicted: R2 (above) was WRITTEN on 2026-08-01, establishing that `EdgeLength`
is an independent knob and that letting it default to `GridSize` caused a
94 GiB-per-voxel blowup. Every paramfile then produced that same day — `params_calib`,
`params_layer1`, `params_refined_std`, every `run_step4`/`run_cube2` variant — omitted
the key, so `EdgeLength` silently became `GridSize`: 4, then 6, then 16 µm triangles.

The cost, from `screen.py:229-230` (a `(T, P, Q)` tensor over each triangle's
detector-pixel bounding box, `P,Q ≈ EdgeLength/px`):

| `EdgeLength` | triangle | px/triangle | relative |
|---|---|---|---|
| 1 µm | 1.8 px | ~4 | **1×** |
| 16 µm | 29 px | ~850 | **~200×** |

A 4202-voxel annulus scan at `GridSize 16` with `EdgeLength` defaulted had not
finished after **3.4 h** and was killed. The same region at `GridSize 4` with
`EdgeLength 1` — **66,864 voxels, 16× more** — was submitted in its place.

**Second-order error: the cost model was inverted.** Believing coarse grids were
cheap, `GridSize` was raised to 6 and then 16 "for speed". With `EdgeLength`
defaulting, that buys nothing — voxels fall as `1/GridSize²` while per-voxel cost
rises as `GridSize²`. The evidence was already in hand and went unread:
**step4_std 9038 voxels → 7900 s; step4_std_g6 6676 voxels → 8265 s.** Fewer voxels,
more time.

**Lesson beyond the parameter:** a finding recorded in the notebook is not a finding
applied. R2 was written, the handbook §10e was rewritten around it, and the very next
paramfile ignored it. When a retraction changes how a run should be configured,
**grep the actual paramfiles** — do not assume the documentation propagated.

**F3. A wrong reference produced a wrong conclusion, twice.** Normalising the beam stripe
by a median filter *along the column axis* could not separate the stationary absorber from
the moving one; the dip finder latched onto the stationary feature and the sinusoid fit
returned rms 230-310 px. It also produced the false conclusion "the sample is not in the
beam for files 598-600". The correct reference is the **ω-median of the stripe profile**,
which cancels everything stationary by construction — with it, the moving shadow is found
in 95-99 % of frames in **all six** files. Retracted.

---

## 9. Sensitivity — what actually buys signal on a weak sample

Campaign context: `nf_Ce5Y_ht450_s2` (Ce-5%Y at 450 C, 2 distances DetZ 7/9, 0.1 deg,
1800 frames/distance, StartNr 10913) and its 0.25 deg twin, both derived from
`fastpar_pokharel_jul26_NF.par`. Same sample, same samY, so they cross-check each other.

### 9a. Spots span 0.30 deg FWHM but were sampled at 0.1 deg

`recon/omega_width.py`, 657 spots over 6 reference frames: profile 1.000 / 0.696 / 0.692
/ 0.422 / 0.401 at 0, +/-1, +/-2 frames. **FWHM 0.30 deg = 3 frames.** The control
(random footprints of matched area) shows no peak at all, so the profile is real.

Each frame therefore holds only about a third of a spot while carrying the full
background. `SumFrames 3` recovers it: 2.38x the peak-frame signal against 1.50x the
noise (sigma_MAD 2.965 -> 4.448, close to the sqrt(3) = 1.73 expected) = **~1.6x**, NOT
the sqrt(N) I first claimed. The profile is peaked, not flat.

Loop-0 result at 10 um, identical grid/geometry/seeds, sigma-matched cut (thr 4 = 7.00
sigma vs the baseline's 7.10 sigma), so the gain is attributable to summing alone:

| | grains >=0.3 | grains >=0.5 | max C | neighbour agreement (C>=0.5) |
|---|---|---|---|---|
| base thr2 sum1 | 128 | 39 | 0.8900 | 0.960 |
| thr1 sum1 (3.55 sigma) | **548** | **303** | **0.9889** | 0.758 |
| sum3 thr4 (7.00 sigma) | 268 | 76 | 0.8500 | **0.950** |

`sum3` doubles the grain count while keeping coherence at baseline level. `thr1` gains
far more but its coherence drops -- consistent with genuinely smaller grains AND with
some spurious ones; those two are not separable from this data.

**Unexplained:** `sum3` max C *fell* to 0.8500. Coarser omega binning should if anything
make matching easier. No explanation; it is the one number here that does not fit.

**Do not compare lit-pixel counts across SumFrames values.** A spot spanning 3 frames
lights 3 patches unsummed and 1 summed, so fewer lit bits is the signature of correct
merging, not lost signal (base 339,822 vs sum3 183,558 lit bits per distance-degree).
The same trap voids the spot-component ladder of §9b for this comparison.

### 9b. The threshold floor is set by the DENOISED noise

`BlanketSubtraction` applies AFTER NLM. Raw sigma_MAD 2.965 -> **0.282 denoised**
(10.5x), so the usual `BlanketSubtraction 2` is **7.1 sigma**, not the ~0.7 sigma the
raw number suggests. Ladder in handbook 8k. Floor is between 1.0 and 0.5; the key is
parsed as `int`, so 2 -> 1 is the only step. Isolated single pixels are the noise
indicator -- a real spot cannot be 1 px.

`thr1` lifted the confidence ceiling 0.89 -> 0.9889 and reproduced the baseline's
confident voxels (426/432 coincident, 93.2% agree <5 deg, median misorientation
**0.05 deg**), so it adds rather than overwrites.

### 9c. Stronger NLM is a NEGATIVE result

NLMH 1.5 and 2.0 barely move the noise (0.282 -> 0.270 -> 0.269) but destroy 35-42% of
spot-like components. Only (NLMH 1.0, thr 1) increases the spot count; every stronger
setting trades spots away. **Keep NLMH 1.0.**

### 9d. Ce5Y is fine-grained -- and the control moves the OTHER way

Grains = spatially connected AND orientation-connected (<5 deg), >=3 voxels:

| C >= 0.3 | 10 um | 5 um | |
|---|---|---|---|
| Ce5Y ht450 | 128 | **220** | +72% |
| Ce ht525 (control) | 62 | **44** | -29% |
| Ce5Y, C >= 0.5 | 39 | **77** | +97% |
| Ce control, C >= 0.5 | 36 | **36** | flat |

Refining the pitch MERGES fragments in the coarse-grained control (count down, size up,
exactly flat at C>=0.5) and FINDS NEW GRAINS in Ce5Y (count nearly doubles while total
area is ~unchanged, 65,000 -> 73,000 um^2). The control moving the opposite way under
identical processing is what rules out "refinement just fragments things".

**The Ce5Y grain size distribution is NOT resolved at 5 um.** Median equivalent diameter
18 um at 10 um pitch is under 2 voxels across -- treat those sizes as upper bounds.

Operator confirmed independently: the rod is wider than the Ce one and the grains are
finer.

---

## 10. beta-Ce: a four-step correction

The most instructive sequence of the campaign. **Every intermediate conclusion here was
wrong, and each was killed by a control.** Read the whole thing before quoting any part.

### 10a. RETRACTION: "no detectable beta-Ce" does not stand

Finding 9 of §1 said CONTROLLED NULL, and §4c quotes max C 0.191 / median 0.052 at two
different `Rsample` values. **Two separate problems with that record, and they are not
the same problem:**

1. **The numbers cannot be re-derived.** No `.mic`, no run directory, no log produces
   0.191. Whatever run generated it is gone. By the provenance rule those numbers should
   never have been quotable, and they are not now.
2. **The only DHCP run still on disk CRASHED**, before fitting a single voxel:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.80 GiB
  in calc_bragg_geometry (midas_diffract/forward.py:916)
```

`/local/s1iduser/nf_ce_dhcp/dhcp.log`. It generated hkls and seeds and died there;
`LayerNr_1/` contains no `.mic` at all. It was also run on the 0.5 deg dataset the
operator had already declared failed, at `MaxRingRad 1600`.

So the honest statement is **not** "it was never measured" -- I cannot prove that, and an
earlier run may well have completed and been overwritten. It is: *the surviving evidence
for the null is a crashed run, and the quoted numbers have no source.* Either way the
conclusion falls, and it falls independently on the FF evidence below.

Cause: dhcp needs **994 reflections and 486,755 seeds** against fcc's 228 and 243,129 --
8.7x the footprint -- on one 47 GB A6000 with `expandable_segments` unset.

**A crash and an absence look identical in a summary table.** The record said
"controlled null", which implied the fit ran and found nothing. Re-run on sentosa's
H200s (143 GB) it completes in 22 minutes.

The operator's FF phase analysis independently refutes the old claim: in this rod the
dhcp-unique lines are strongly occupied -- (101) 91.9%, (110) 29.7%, (103) 25.6%,
(105) 17.2%. In `Ce5Y_ht450_s2` the same probe collapses ((101) 1.9%, (110) 0.0%), so
that sample IS single-phase fcc, a = 5.1638 +/- 0.0011 A.

### 10b. WRONG: "the dhcp map is leakage"

The completed run reached max C 0.4938 (median 0.113) where fcc reaches 1.000 on the
same voxels -- and DHCP confidence correlated **+0.749** with FCC confidence, with all
1,167 voxels above 0.3 sitting where gamma-Ce was already strong. Read as leakage.

That reading was wrong, and the correlation had an obvious physical explanation I did
not consider: **gamma-Ce and beta-Ce are polytypes** (ABC vs ABAC on the same
close-packed planes), so beta forming where gamma is well-ordered is what intergrowth
looks like. A correlation alone was never evidence either way.

Also: the coherence test **would have fooled me the other way**. It returned z = +44.5,
which looks decisive, but its null was 0.523 rising to **0.995** -- nearly every
high-DHCP voxel carried the SAME orientation, so shuffling changed nothing and the
statistic was empty. Quoting it alone would have produced a confident false positive.

### 10c. Two artifact mechanisms, both killed

**Shared reflections** -- killed by a per-spot audit (operator's suggestion). Of 128,937
valid DHCP predicted spots, only **3.2% are FCC-explainable** (an FCC spot within 3 px,
+/-1 frame); those match at 0.992, the DHCP-only 96.8% match at **0.377**. The fit is
matching its own reflections, not fcc's in disguise.

**Lit-pixel density** (`FracOverlap` has no background term, so any orientation scores
higher in a busy region) -- killed by a **sham phase**: same space group, same 486,755
seeds, MORE reflections (868), lattice scaled 1.060x so no ring can be real. It never
exceeds **0.1333** anywhere, against 0.4938. Density alone cannot produce that.

### 10d. What IS established

The dhcp c-axis is parallel to an fcc <111> of the co-located grain to within
**0.17 deg median, 94.8% under 5 deg** across all 3,339 voxels at C >= 0.20 (100% of the
254 at C >= 0.40). That is the polytype relationship, and it explains the "one
orientation everywhere" worry: those voxels lie inside essentially one large gamma grain
(FCC orientation spread median 0.2 deg, FCC confidence median 0.959), so the beta
orientation is locked to that parent.

**Status: PROVISIONAL.** Not through a fresh adversarial pass. And it is NOT a beta-Ce
grain map -- one orientation tied to one parent, whereas the collaborator's independent
reconstruction shows ~15 distinct grains. Whether beta-Ce genuinely occurs only as faults
within gamma grains, or our fit only finds it where the parent is strongest, is open.

---

## 11. Structure factors — reflections that cannot exist were in the denominator

### 11a. The defect

`write_nf_hkls_csv(space_group, lattice, ...)` took **no atom basis**, so it applied
space-group extinction rules and nothing else. Basis-dependent extinctions were
invisible. For dhcp beta-Ce (Ce at 2a + 2c) **126 of 736 reflections have |F|^2 = 0**.

They are predicted, can never be matched, and sit in the confidence denominator:

| phase | reflections | forbidden | max achievable FracOverlap |
|---|---|---|---|
| dhcp beta-Ce | 736 | 126 (17.1%) | **0.829** |
| fcc gamma-Ce | 228 | 0 | 1.000 |

The fcc row is the control and it validates the calculation: cap 1.000 predicted,
1.0000 observed. **This is why the defect never showed up before** -- single-atom fcc and
bcc cells have no extra extinctions. It affects HCP, DHCP, intermetallics, oxides.

The failing planes are exactly the forbidden ones -- (002), (303), (306), (222), (307),
(2,1,10), (130) all at 0.000-0.016 -- while the operator's gamma-clean FF lines (101),
(102), (103), (105), (106) match at **0.72-0.94**.

### 11b. The fix and its preregistered test

`PhaseAtom` + `DropForbiddenReflections` (handbook 8l). **Predicted before running:**
0.4938 / 0.829 = 0.596. **Measured: 0.5962.**

| | max C | median | >=0.3 | >=0.5 |
|---|---|---|---|---|
| 736 reflections | 0.4938 | 0.1126 | 1,167 | **0** |
| 610, forbidden dropped | **0.5962** | 0.1346 | 1,894 | **213** |

Landing ON the prediction rather than above it says the orientation SEARCH did not
improve -- the same orientations win, they were simply mis-scored. The consequential
number is the last column: beta-Ce could not clear `MinConfidence 0.5` anywhere before,
so the multi-res ladder had nothing to seed from.

Three metrics via one hook, `hard_fraction(refl_weight=...)`: ones -> C_raw,
(f2 > 0) -> C_filt, f2 -> C_weight. Prototyped on real voxels before touching the
package (C_filt 1.196x, C_weight 1.849x; sham null gains only 1.09x and 1.45x, so
weighting favours the true phase rather than inflating everything). **C_filt drives**;
C_weight lifts the null's ceiling 0.16 -> 0.31 and needs thresholds re-tuned first.

### 11c. No Lorentz factor — it CANCELS, and this was measured

Rotation method: I_int ~ |F|^2 P L with L = 1/(sin2theta |sin eta|), and the same slow
Ewald crossing smears the reflection over Domega = w_rlp L. So
I_peak = I_int/Domega ~ |F|^2 P -- **L cancels** for a per-frame threshold.

Confirmed on the fcc 111 ring of `nf_Ce_ht525_s2`, where 2theta and |F|^2 are constant
by construction so only eta varies:

| \|sin eta\| | ~eta | spots | median peak | if L mattered | spots / 1000 ring px |
|---|---|---|---|---|---|
| 0.15-0.30 | 13 | 6 | 43.0 | 43.0 | 4.4 |
| 0.50-0.70 | 37 | 60 | 31.0 | 16.1 | 26.9 |
| 0.90-1.01 | 73 | 172 | 39.8 | 10.1 | 32.8 |

Peak is flat (28-43, non-monotonic) where 1/|sin eta| predicts a 4.3x decline. The last
column is the independent half: spot DENSITY rises 7.5x with |sin eta|, close to the
`~ sin eta` the same model requires (reflections near the pole linger longer but far
fewer land there).

**Limit of validity:** the cancellation holds only while Domega >> step. At 0.1 deg with
Domega ~0.3 deg we are comfortably there; `SumFrames 3` puts the step AT the crossover,
so this needs rechecking if summing becomes standard.

**A failed attempt worth not repeating:** measuring Domega vs eta directly
(`recon/omega_width_vs_eta.py`) was inconclusive -- its random-footprint control returned
FWHM of 1.6-2.2 deg, LARGER than the 0.4-0.8 deg measured for real spots, because
frame-to-frame background drift swamps per-spot width fitting. Peak height at fixed ring
is the observable; width is one step removed and much noisier.

### 11d. `forward_batched_grains` returns (frame_nr, valid, y_pixel, z_pixel)

`frame_nr`/`valid` are `(B,K,M)`; `y_pixel`/`z_pixel` are `(D,B,K,M)` -- exactly as
documented (`soft_overlap.py:407`). An earlier note here and in the checkpoint claimed
`valid` came back `(D,B,K,M)` "not as documented". **That was wrong**: it came from
unpacking the return as `(frame, y, z, valid)`, which silently puts `valid` in `y` and
`z` in `valid`. Production code and the repo test unpack correctly; only analysis
scripts did not. Symptom: a positive control matching **0 of 91,200** spots. This is also
the cause of the long-broken `residual_check.py`.

**Every spot-level script must carry a positive control** -- forward-simulate voxels the
fit scored C >= 0.99 and require ~1.0 matched before believing any other number. That
control caught this twice.

### 11e. The validator was reporting three errors on every correct NF paramfile

`NrOrientations`, `SeedOrientations`, `SeedOrientationsAll` were marked required and/or
`file_exists`, but all three are **outputs** the pipeline writes. They cannot exist at
preflight. Ten live keys were also unregistered and warned as unknown -- including the
whole NLM group, which was delivering the campaign's single biggest gain while the
validator called it unrecognised. Both fixed; `params_ce5y_full_fcc.txt` now validates
0 errors, 0 warnings.

**Nothing prevents this drift recurring.** A test that walks the keys each package
parses and asserts they are all registered would catch the next one at CI time.

---

## 8. `xzhang_jul26` / `s6061_NF` — a second 20-ID campaign, and what did NOT transfer

Two distances (nominal 9 and 11 mm), two layers 10 µm apart, files `000629`-`000632`,
`000628` = dark. Same beamline, same detector **serial 20514670**, ~two weeks after
`nfdev_jul26`. The point of the exercise was to test whether the earlier distance
calibration transferred. Most of the campaign was spent discovering that four separate
things did **not**.

### 8a. What the data is

| | `nfdev_jul26` (Au) | `xzhang_jul26` (`s6061_NF`) |
|---|---|---|
| encoding | 10-bit stored ×64, max 65472 | **12-bit unscaled, max 4092** |
| exposure | 0.200 s (period 275 ms) | **1.25 s** (period 1325 ms, 32 min/scan) |
| frame mean | 1.02 counts | **7.16 counts** |
| beam stripe row | 4479 | **4537** |
| beam FWHM | 11 px = 6.0 µm | 14 px = 7.7 µm |
| illuminated width | 128 px = 70 µm | **452 px = 247 µm** |
| beamstop | present, R ≈ 1100-1240 px | **absent** — stripe unbroken, cols 1720-3640 |

`exchange/data_dark`, `data_white`, `data_white_post` are all present and **all exactly
zero** — placeholders, not flat fields. The separate dark file is zero too. The temporal
median is the only background available. There is no acquisition log (the parent directory
is permission-denied) and the HDF5 carries no energy, so **63.314 keV is inherited**, not
measured, for this campaign.

`AcqPeriod` reads 0.0339 s and is a stale PV; `NDArrayTimeStamp` gives the real 1325 ms
period. Believe the timestamps.

θ runs −180 → +180.25 in 1442 frames. Frames 1440-1441 sit at +180.00/+180.25, which are
−180.00/−179.75 **mod 360**, so they duplicate frames 0-1 exactly as in `nfdev_jul26`:
`NrFilesPerDistance 1440`. A naive `theta[0] == theta[1440]` test returns False and is
wrong — compare angles, not values.

### 8b. Four wrong answers in a row, all from the same root

The question "is the distance calibration consistent?" was answered wrongly three times
before the right tool was reached. Recorded because the *sequence* is the lesson.

| attempt | answer | why it was void |
|---|---|---|
| raw max-projection + peak finder | "ratio 0.9017 vs 1.2457, **27.6 % off**" | ÷64 applied to unscaled data ⇒ threshold 128 not 2 ⇒ peak finder tracking the **pedestal**; its "peaks" at 56-416 px are inside the beamstop radius, where nothing can be |
| median-corrected profile scaling | "δ = −937 ± 71 µm" | used the **Au beam centre**, 57 px wrong for this campaign |
| same, correct BC | "δ ≈ −1550 µm, **disagrees with Au**" | BC now right, but powder-ring fitting is the wrong tool entirely |
| **spot triangulation** | δ = −634 to −704 µm | passes its own gates and nulls — see 8d |

**The control that should have been run first.** Two layers of the same material at the
same distance, 10 µm apart, must give **identical** ring radii. They gave 1028/1152/1294/
1364/1496 and 1108/1212/1304/1386/1574. Nothing else needed checking after that.

### 8c. Why there are no rings — and it is not a defect

An NF spot lands at *grain position* + `Lsd·tan(2θ)·d̂`. The first term smears every ring by
the illuminated width. Here that is ±226 px against a 111→200 spacing of 225 px — **2.0×
the spacing**. The Au cube, at 70 µm, gave ±64 px = 0.6× and its rings were visible.

Confirmed independently by histogramming **spot radii** (each spot once, unweighted): the
on-ring excess was **+0.2 %** at a = 3.59 and **+2.4 %** at a = 4.05 for the 9 mm scan,
and +2.0 % / −0.2 % at 11 mm. No rings at either candidate. ⇒ The lattice parameter is
**not readable from this detector image**; the indexer has to decide. Handbook §5e.

### 8d. Triangulation — what it gave and what it is worth

`beam_calib.triangulate`, 240 ω samples, ~89 spots/frame (13.5 chance matches/frame):

| `ang_tol` | `r_min` | k | n_peak | L(9 mm) | y-vs-z | verdict |
|---|---|---|---|---|---|---|
| 0.30° | 300 px | 1.0146 | 162 | 136659 | 129412 | **REJECTED** by the module |
| 0.15° | 800 px | — | — | 8361.7 | — | OK, nulls 7.4× / 4.5× |
| 0.30° | 800 px | 1.2391 | 108 | 8366.4 | 142 | OK, nulls 8.3× / 4.2× |
| 0.50° | 800 px | — | — | 8296.4 | — | OK, nulls 8.4× / 4.9× |

`r_min=300` is the instructive row: near BC ray directions are degenerate and the matcher
pairs noise, producing a 136 mm "distance". The module rejected it without being asked.

**But the precision is ~200 µm, not ~2 µm.** The model assumes a point source at BC; a
247 µm specimen perturbs `|p − BC|` by up to 11 % per spot (3 % for the Au cube). The
y-vs-z split rose by the same factor, 57 → 142 µm. Handbook §6i-ter.

⇒ Triangulation **alone** cannot resolve the Au δ = −860 µm from its own −658 ± 35 µm;
the 200 µm gap is inside its own systematic.

> **This was written as "the dataset cannot resolve it" and that was WRONG — see §8h.**
> Triangulation alone could not. Triangulation **followed by geometry refinement** could,
> and the refined answer lands 6.8 µm from the Au value. The lesson is not "the
> measurement is impossible"; it is **"triangulation is a SEED for refinement, not the
> final Lsd, whenever the sample is wide."** Its 211 µm error here was correctly
> predicted by the point-source argument — the caveat was right, the conclusion drawn
> from it was too gloomy.

### 8e. ybc could not be measured — and the tracker default is wrong at 20-ID

zbc is clean: stripe centroid 4537.1, four files agreeing to 0.7 px.

ybc is the **rotation-axis projection**, and neither route reached it:

* `find_stationary` returns a fixed 26 µm absorber at col 3408 (T = 0.89) — a piece of
  hardware, not the sample. There is no on-axis particle here, which is exactly what made
  it easy in `nfdev_jul26` (§7d).
* `fit_axis` **refused at every setting** (`is_reliable` False). The specimen is extended
  and irregular — its shadow width swings 56→886 px with ω — so the deepest-dip centre
  does not trace a rigid sinusoid.

Two method errors were made getting there, both caught by validating against the known Au
answer (axis col 2625.47) rather than by inspection:

1. A hand-rolled "robust" centroid of `1 − T` over the whole band returned **2697** for Au
   (off by +72 to +80 px, rms 132). This is the module's documented trap 3, reimplemented
   by hand after reading it. **Use the module.**
2. The module itself, at its `band_frac=0.30` default, returned **2721** for Au (off by
   +96 px, rms 216, amplitude clipped to 634 px). `band_frac=0.70` returns **2625.88**
   (+0.41 px, rms 1.5, amplitude 918 px = 503 µm, which independently reproduces the
   496.8 µm cube-2 offset). The default is wrong for this beamline; `is_reliable` flagged
   every bad row as False and should be branched on. Handbook §6e-0.

ybc for the reconstruction is therefore **inherited** from the Au campaign at the matching
nominal distances (9 mm → 2625.47, 11 mm → 2631.91), marked as inherited in the paramfile,
with `BCTol` widened in y so refinement can move it. Justification for inheriting: the
diffuse-halo centre here is col 2624.5 (three files, 1.5 px), matching Au's 2625.47 — the
detector appears to have moved vertically only. That is an argument, not a measurement.

### 8f. Reduction settings, and the threshold translation

`sigma_MAD` is **EXACTLY 0** here as well (84.4 % of the residual is exactly zero), so the
σ-scaled NLM path would silently skip denoising — the same defect `NLMHAbsolute` was added
for. NLM is worth it: at matched blob yield it cuts noise singletons **12×**
(200 blobs/1596 singles raw → 105 blobs/92 singles denoised).

Because full scale is 4092 rather than 1023, **one Au count = four counts here**. The
operator's recipe (NLM h = 1, threshold 2) therefore translates to **h = 4, threshold 8**.
Do the translation explicitly; the numbers are not portable.

### 8g. `nfdev_jul26` postscript — cube 2 found, and it validates the shadow method

The `full_e1` run (`EdgeLength 1`, `GridSize 4`, annulus mask, 66,864 of 181,656 voxels)
located the second cube at **(−279.2, −416.7) µm, r = 501.6 µm, angle −123.8°** — 446
voxels, maxC 0.766; the on-axis cube gives 661 voxels at maxC 0.9130.

**Absorption said 496.8 ± 0.7 µm; diffraction says 501.6 µm — agreement to +4.8 µm, about
one voxel.** Two unrelated physics (X-ray attenuation and diffraction indexing) agreeing
to 1 % is the strongest cross-check in either campaign, and it retires the open item
"cube 2 has never been located". The −123.8° angle back-derives the phase→(x,y) convention;
verify it on one more dataset before treating the sign as general.

### 8h. Setup optimisation — the distance calibration IS consistent, and how it was checked

Run at the operator's `a = 3.60` (which beats 3.59: maxC 0.7059 vs 0.6765, n≥0.6 49 vs 17
— `a` and `Lsd` are degenerate through `r ∝ Lsd·λ/a`, so most of a 0.28 % change is
absorbed by `Lsd`, but the *relative* ring spacing is not, and that is what discriminates).

Three passes: **A** search on the starting geometry → **B** refine `Lsd`/`BC`/tilts from
the best voxel (hard objective, `NumIterations 3` in ONE invocation) → **C** search
**again** on the refined geometry. Pass C is the point — it scores voxels that were *not*
used to fit the geometry. Re-seeding a refinement with its own output is the trap
(hard rule 15); a fresh search is not a re-seed.

Pass B: seed voxel **0.705882 → 1.0000000000** (19577 evals, 27.5 s).

| | Lsd 9 mm | Lsd 11 mm | δ = L − DetZ |
|---|---|---|---|
| triangulated (this data) | 8366.4 | 10366.4 | −633.6 |
| **REFINED (this data)** | **8155.6906** | **10155.3298** | **−844.31 / −844.67** |
| `nfdev_jul26`, refined | 8162.2839 | 10162.2596 | −837.72 |

**⇒ The earlier distance calibration IS consistent: δ agrees to 6.8 µm on a 9 mm
distance (0.08 %),** and the two distances independently give δ agreeing to 0.36 µm.
Refined BC `(2691.4720, 61.1784)` / `(2680.8706, 62.2930)`: ybc moved only −2.1 and
−6.2 µm from the INHERITED Au values and nowhere near the ±20 µm `BCTol` edge, so
inheriting it was justified and these are now dataset-local. Tilts tx=0.0347,
ty=−0.3431, tz=0.0068 — **distrust ty**, the objective is 26× less sensitive to it.

#### The check that matters when confidence jumps to 1.0

| | maxC | medianC | n≥0.5 | n≥0.7 | n≥0.8 |
|---|---|---|---|---|---|
| PASS A | 0.7059 | 0.2286 | 217 | 1 | 0 |
| PASS C | **1.0000** | 0.3684 | **2911** | **1415** | **810** |

The whole distribution moved, so it is not a single-voxel overfit. **But the median rose
too** (0.229 → 0.368) and 2911/7350 ≈ 40 % of the disc indexing is exactly the §7b "wrong
plateau" signature. maxC and median **cannot** tell a solved geometry from a plateau.
Neighbour misorientation can:

| pair type | median miso | frac < 5° |
|---|---|---|
| ADJACENT voxels (≤22 µm, grid pitch 20) | **0.23°** | **78.0 %** |
| RANDOM voxel pairs | 40.98° | 4.5 % |

17× enrichment ⇒ **coherent grains, real microstructure.** A wrong-plateau geometry
produces a spatially *random* orientation field; a real one does not.

> **RULE: whenever a refinement drives confidence to 1.0, test the ORIENTATION FIELD, not
> the confidence.** Compare misorientation between spatial neighbours against random
> pairs (`midas_stress.misorientation`, cubic m-3m, radians in and out; `.mic` binary
> columns 7-9 are RADIANS). Confidence statistics are blind to the failure this catches.

Two supporting results from the same run, for the record:

* **Material settled by the indexer, as §8c said it would have to be.** Al 6061
  (a = 4.05) scored maxC **0.1951** — *below SS316L's median of 0.2286* — with **zero**
  voxels above 0.4 anywhere. The `6061` filename is misleading; the material is SS316L.
* **The empirical chance floor is the MEDIAN over the search volume (~0.22), not the
  naive lit-fraction product.** With 1.65 % and 1.40 % of pixels lit at the two distances
  and `hits_d.prod(dim=0)` ANDing them, independent-pixel arithmetic predicts 2.3e-4.
  The truth was ~1000× higher, because observed and predicted spots cluster in the *same*
  regions. **Never compute a chance floor from pixel fractions; read it off the median.**

### 8i. `NF_Au_cube_0802` — and a recorded lesson repeated, with a new excuse

A second Au cube at 20-ID, three distances, ΔD = 2000 µm, taken after the
beamstop was addressed. Steps 0-3 went cleanly and are worth stating because
they show what the procedure achieves when it is followed:

* **Lsd = 6138.7 / 8138.7 / 10138.7 µm** by triangulation (3 pairs, y-z splits
  25/58/54 µm, nulls 13.75× and 8.46×) — reproducing `nfdev_jul26`'s
  6139.7/8139.7/10139.7 **to 1.0 µm**, with k(0→1) agreeing to four decimals.
* **Step 3 reached FracOverlap 1.000000 with NO refinement** (1.0000 → 1.0000,
  improvement +0.0000000000; Lsd, BC and tilts all returned unchanged). The
  beamstop that capped `nfdev_jul26` at 0.913 is gone, as the beamline said.
* Single-crystal control: **100 % of the 1071 C ≥ 0.9 voxels within 1°** of the
  best voxel (median 0.707°). maxC 1.0 alone would not have shown that.

**Two defects worth carrying forward.**

**1. The spot budget was spent on halo before the radius cut.** The first
triangulation attempt failed at every setting with `k ≈ 1.007` and y-vs-z splits
of hundreds of millimetres. Cause: the spot finder kept the `MAXSPOT` brightest
blobs *before* `r_min` was applied, and there are **524 blobs/frame inside
r < 800 px** against ~120/frame outside it, so the real spots never reached the
matcher. `k ≈ 1` with an enormous y-z split means the matcher is pairing noise —
it does **not** mean the detector failed to move. Apply the radius cut first.

**2. The point-mask null, repeated.** §7f (F5/F6) already records that a
candidate-point tomo mask cannot be interpreted: on `nfdev_jul26` masking four
candidate angles returned 0.0000 at every one of 5316 off-axis voxels, and the
cube turned out to be at −123.8°, none of them. The recorded fix is *mask the
whole locus*. On `NF_Au_cube_0802` a **two**-candidate mask was written anyway
and returned the same nothing: 10,272 voxels, on-axis cube perfect (maxC
1.000000), **0 off-axis**.

> **The new excuse, which is the part worth recording.** This time there WAS a
> well-measured phase — +35.0°, rms 0.81 px, 18 mutually consistent fits — where
> the previous campaign had none. Having a precise phase made a point mask feel
> justified. It is not. **The precision of the phase says nothing about whether
> the phase→(x, y) MAPPING is right, and the mapping is the unpinned part.** A
> precise measurement pushed through an unverified convention is still
> unverified. F6 said "a finding recorded is not a finding applied"; this adds
> that a *better measurement* is the thing most likely to talk you out of a
> recorded lesson.

**3. A new shadow failure mode: two absorbers confuse the tracker.** `fit_axis`
refused at every setting (rms ~112 px) until the STATIONARY on-axis cube was
masked out of the profile; then 18 settings returned `is_reliable`, agreeing on
r = 499.8 ± 2 µm with rms down to 0.81 px. This did not appear on `nfdev_jul26`
because the swing there (906 px) separated the two absorbers cleanly; here it is
only ~500 px. **With more than one absorber, freeze the stationary one to the
ω-median before tracking the moving one.**

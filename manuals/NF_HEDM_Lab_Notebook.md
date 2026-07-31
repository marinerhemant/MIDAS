# NF-HEDM Lab Notebook — `pokharel_jul26` / Au5 calibrant + `Ce_ht525_s2`

**Companion to `NF_HEDM_Handbook.md`.** The handbook says what to do; this records what
was actually found, how it was measured, and what turned out to be wrong. They are kept
apart on purpose: the handbook has to stay short enough to follow, and this has to stay
honest enough to stop a refuted idea coming back.

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
| 9 | No detectable β-Ce (DHCP); γ-Ce (FCC) indexes to 1.000 | CONTROLLED NULL | §4c |
| 10 | NLM on the median-corrected residual gives 3.6× the voxels ≥0.9 | ESTABLISHED | §4d |
| 11 | The reconstruction is spot-detection-limited, not geometry-limited | ESTABLISHED | §4a vs §4d |

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

### 4c. No detectable β-Ce — a controlled null

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

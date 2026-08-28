# Phase 2 — Build the parameter file

> Part of the **FF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 6. STEP 5 — Build the parameter file

Start from `FF_HEDM/Example/Parameters.txt` and replace the geometry block. Generate it
from the calibration JSON rather than retyping — hand-copied geometry is how a recon ends
up describing a calibration it isn't using.

**The calibrant's `LatticeConstant` and `SpaceGroup` must be replaced with the sample's.**
The calibration paramstest carries CeO₂ (5.4116, SG 225); a gold sample needs
`4.0782, SG 225`. This is the single most common copy-paste error.

**`ImTransOpt` must match what the calibration was fitted on.** If the calibrant image was
read straight out of `exchange/data` with no transform, the recon must use `ImTransOpt 0`.
A mismatch mirrors the geometry relative to the fit.

Ring selection: only rings with **full azimuthal coverage** are safe defaults. With BC near
the centre of a 2048² detector, that is everything inside the *nearest-edge* radius;
rings between the nearest-edge and far-corner radii are partial and bias η coverage.

### `MinNrSpots` — never below 3 on a full rotation, never below 2 ever

**HARD RULE: `MinNrSpots` must be ≥ 3.** The only exception is a **partial rotation**
scan, where it may be dropped to **2** — and never below 2 under any circumstances.

Two spots do not constrain a grain: an orientation has three degrees of freedom, so a
2-spot "grain" is under-determined and the refiner will happily fit it, report a position
and a lattice, and pass every downstream filter. On a full 360° sweep every real grain has
far more than three reflections in range, so `MinNrSpots 2` buys nothing except
under-determined false grains that dilute every population statistic you then quote.

The example file's recommended working value is `MinNrSpots 3` (§10). Beamline parameter
files in circulation carry `MinNrSpots 2` — that is a defect to fix, not a template to
copy.

`Rsample`, `Hbeam`, `BeamThickness`, `Vsample`, `GlobalPosition` are **not** descriptions
of the sample. **HARD RULE: never set `Rsample`/`Hbeam` to the true sample dimensions.**
They are a deliberately generous *search bound*; tighten them to the real size and any
grain whose true position lies near the boundary is pushed onto it, giving an artefactual
pile-up of grain positions at ±`Rsample` and ±`Hbeam`/2 that reads as real microstructure.
Leave the generous defaults (2000 µm here, matching `FF_HEDM/Example/Parameters.txt`).

---

## 6b. Set `RingThresh` from the data, not from a template

> **Which ring is "ring 30"? Read the run's own `hkls.csv` (README rule 16).** MIDAS's
> numbering diverges from a fresh `generate_hkls()` above ring ~19 and **fails silently**.
> On 20-ID alumina, adding "ring 30" and "ring 32" believing they were the 2nd and 4th
> strongest available actually selected reflections at 2.7 % and 0.33 % relative
> intensity; they yielded 115 and 88 spots against the ~3000 expected. Map ring → (hkl,
> radius) from `hkls.csv` and match on **(h,k,l)**, never on the ring index.
>
> Two things to check before adding any ring:
> * **Radial isolation.** Rings closer together than the radial margin are **duplicated,
>   not split** — two rings 3.9 px apart emitted all 2930 peaks *twice*, once under each
>   label, with byte-identical `YLab`/`ZLab`/`Omega`. A spot mislabelled between them is
>   unmatchable, which is worse than not adding the ring.
> * **`Width` is the lever, and it is in µm.** It is the band half-width either side of
>   each ring, so two rings collide once they are closer than `2·Width/px` pixels apart.
>   Where they overlap the **later entry silently wins** — no warning, no duplicate count,
>   just spots assigned to the wrong ring. Worked example, 20-ID alumina at 150 µm pixels:
>   rings 12 and 13 sit at 842.6 and 861.7 px, **19.1 px apart**. The default `Width 1500`
>   is ±10 px = 20 px of band and overlaps them; `Width 1200` (±8 px) does not, and is
>   what that run used. Compute `2·Width/px` against your closest *adjacent used* pair
>   before accepting a `Width`, not against the closest pair in `hkls.csv`.
> * **Signal, measured with hot pixels rejected.** Do **not** judge a ring from a
>   max-projection: one stuck pixel gives a full-amplitude "ring" at its own radius
>   forever, which scored several dead rings at SNR > 30. Mask any pixel firing in more
>   than a few percent of frames, then count detections per unit annulus area.
>
> And check the payoff is real before spending a run on it: on that alumina layer, going
> from 8 to 13 rings changed the internal-angle distribution not at all (median 0.544 →
> 0.545). Corundum simply had no more usable rings.

`RingThresh <ring> <threshold>` is the single most consequential number in the peak
search, and the value in any example file is meaningless for your detector, exposure and
sample. Measure it.

The peak finder labels 8-connected blobs above threshold **inside the ring bands only**
(`Width` µm either side of each ring), then applies a **strict** size filter —
`minNrPx < nPx < maxNrPx`, both bounds exclusive (`midas_peakfit/connected.py:91-100`).
With the default `minNrPx 1`, **any single-pixel blob is discarded**. So a threshold that
is slightly too high does not degrade gracefully: it shaves every spot down to a few
isolated pixels and you get exactly zero peaks.

Use the calculator — do not hand-roll the sweep:

```bash
# PREREQUISITE: needs hkls.csv for the ring radii and refuses without it
# ("No ring radii available... This tool needs hkls.csv"). Only the `hkl` stage
# writes it, so on a fresh run do BOTH conversions first:
midas-pipeline run --params Parameters.txt --result results/ --layers 1-1 --only zip_convert
midas-pipeline run --params Parameters.txt --result results/ --layers 1-1 --only hkl

midas-ring-thresh <run>/LayerNr_1/<name>.MIDAS.zip \
    --result-folder <run>/LayerNr_1 --n-frames 40
```

It sweeps thresholds through the **production** peak-search path
(`compute_good_coords` → `preprocess_frame` → `find_regions` →
`filter_regions_by_size`), reports two independent criteria per ring, and prints
paste-ready `RingThresh` lines.

**Do not reimplement the band mask.** Earlier revisions of this section carried a
seven-line snippet that built its own mask from a plain radius-from-beam-centre. That is
wrong: the production band uses the *distortion-corrected* `Rt` after
`apply_image_transformations` + `transpose_square`. Measured on `Au3_cubes_ff_000008` the
naive mask shares only **13.4 %** of its pixels with the real band, which made blob counts
disagree with the pipeline by ~67× and manufactured a spurious "background varies by 20σ
around the ring" result. Through the real band that background is flat (spread 0.4σ).

The two criteria:

- **A — blob SNR.** Lowest threshold at which ≥90 % of surviving blobs have local
  SNR > 5, measured on the *ungated* frame. (It must be ungated: after thresholding, every
  sub-threshold pixel is 0, so the local background and its MAD collapse and every SNR
  reads as 0.) The annulus is restricted to in-band pixels — a band is only `2*Width` wide,
  so an unrestricted annulus is mostly out-of-band zeros.
- **B — expected false positives.** Lowest threshold whose predicted noise-blob count over
  the *whole scan* is under 10, from the per-cell σ and the `minNrPx` size filter. This is
  the criterion that matters when the sample is sparse: a 2-grain dataset has ~1–2 real
  peaks per frame against ~3×10⁵ in-band pixels, so a tiny per-pixel false rate still
  swamps the signal.

They should agree; the tool says so explicitly when they do not, which points at a bad band
or a broken dark rather than at the threshold.

**Why the old "pick the knee" rule was not enough.** Blobs/frame surviving the size filter
on `Au3_cubes_ff_000008` go 5.2 (thr 5) → 1.6 (10) → 0.8 (20) → 0.5 (40). The two-orders
jump is between 5 and 10, but noise keeps falling out well past it. The knee locates where
noise *percolates* into detector-spanning blobs (largest blob 645 px at thr 5 vs 393 at
10), not where noise stops being admitted.

Measured on `Au3_cubes_ff_000008` (20 ms/frame, `Width` 7.5 px) the two criteria agree on
every ring and give **`RingThresh` 10 / 20 / 20 / 10 / 10** for rings 1–5.

> **Re-measured 2026-08-12 on the same file: `10 10 10 10 10`**, both criteria agreeing
> flat across all five rings, and matching the production `Parameters.txt` already on disk
> for this dataset. The numbers here are a worked example of the *procedure*, never values
> to copy — this is what that looks like when the tool is actually re-run.
 On ring 5 the
clean fraction goes from 20 % of blobs above SNR 5 at threshold 5 to 100 % at threshold 10.

*(An earlier revision of this section reported "both criteria return 10 on every ring, 92 %
clean at 10 vs 52 % at 5". Those came from a `blob_snr` that restricted its background
annulus to in-band pixels and was over-optimistic; fixed, and the numbers above supersede
them — Lab Notebook §6b.)*

**Caveat:** this tuning is only meaningful once the dark is verified non-zero (§3d). If the
dark is missing, *every* threshold yields zero peaks and the table above is flat — that
invariance is itself the diagnostic.

---

## 6c. Reject spurious peaks by SNR, not by a proxy

`RingThresh` decides what gets *detected*. **`MinPeakSNR`** decides what gets *kept*, and it
is the only quality criterion here that does not smuggle in an assumption:

```
MinPeakSNR 5          # 0 = off (default); (peak - cell_median) / cell_sigma
```

> **`midas-zipper >= 0.1.5` or this key does nothing.** 0.1.4's allow-lists do not carry
> `MinPeakSNR`, `BgSubtract` or `BgNSectors`, so a parameter file zipped by it drops all
> three into the void — the peak search then runs at the defaults with no error and no log
> line (`a440bef6`, §0). The floor is declared as of `midas-pipeline` 0.8.2, so a fresh
> install is safe — but **an existing zarr written by an older zipper stays broken**, and
> re-running with a newer zipper installed does not fix a zip that already exists (§7:
> `zip_convert` is skipped when the zarr is present). The keys are written as **datasets**
> under
> `analysis/process/analysis_parameters` (`ff_zip.py:159-167`), so check the zarr itself
> before trusting any threshold you set:
>
> ```bash
> python -c "
> import zarr, sys
> g = zarr.open(sys.argv[1], mode='r')['analysis/process/analysis_parameters']
> for k in ('MinPeakSNR', 'BgSubtract', 'BgNSectors'):
>     print(k, list(g[k][:]) if k in g else '*** ABSENT — zipper too old ***')
> " <result>/LayerNr_1/<stem>.MIDAS.zip
> ```

Computed per peak against its own (ring, azimuthal sector) cell during the peak search, so
it costs nothing extra and applies to **FF and PF alike** (both use `midas_peakfit`). It is
the natural knob for the pf-HEDM failure mode where spurious signal is admitted.

**Do not substitute a proxy — each of these has been measured to fail:**

| proxy | why it fails |
|---|---|
| `MinIntegratedIntensity` | no noise estimate, so it cannot tell a weak real spot on a quiet patch of detector from a noise excursion on a hot one |
| `FitRMSE` | an **absolute** residual, so it grows with peak intensity — cutting at `FitRMSE < 2000` discarded **58 % of the indexed spots** on the reference dataset |
| ω multiplicity (`NImgs`) | encodes **mosaicity, not reality**. A small or undeformed grain can satisfy Bragg inside one frame; 45.9 % of credible spots were single-frame and 8 indexed spots reached SNR 2511 on one frame |

**No specific value is recommended yet.** Two SNR estimators (per-cell vs a box on the raw
frame) rank spots differently — 94 % vs 53 % clean at SNR 5 on the reference dataset — and
which is correct decides where the cut belongs (Lab Notebook §6d). Start at 5, and check
what it removes against the raw frames before trusting it.

---


## 6d. `RhoD` — compute it, never copy it

`RhoD` is the **beam-centre-to-farthest-corner distance, in micrometres**:

```
RhoD = px * hypot(max(BC_y, N_y-1-BC_y), max(BC_z, N_z-1-BC_z))
```

`midas-calibrate-v2 --mode ff` writes it for you. Compute it if you are writing
the file by hand. Do **not** inherit it from another sample's parameter file.

It is two quantities wearing one name, which is why a wrong value does two
unrelated kinds of damage:

* **the distortion normalisation** — the polynomial is in `ρ = R_µm / RhoD`, so
  an oversized `RhoD` makes every term feeble and the fitted coefficients run to
  their bounds trying to compensate;
* **the hkl cap** — it is aliased to `MaxRingRad`
  (`midas_transforms/params.py`, `__post_init__`), so it sets how far out
  reflections are generated.

**Measured, 20-ID Varex, 2880 px at 150 µm (true `RhoD` = 309 538):**

| `RhoD` | rings in `hkls.csv` | seeds indexed | grains |
|---|---|---|---|
| 2 000 000 | 745 | **0 of 4569** | crash in `process-grains` |
| 309 538 | 46 | 3122 of 4515 | 208 |

The zero-seed run **exited 0**. Nothing between the parameter file and the empty
`Grains.csv` reported a problem.

Three things make this worse than an ordinary bad number:

1. **It is material-dependent.** The same `RhoD 2000000` gave only 70 rings on
   cubic nf709, which reconstructed fine — a low-symmetry cell generates far
   more distinct rings than a high-symmetry one. "It worked on my other sample"
   is not evidence.
2. **Recalibration does not fix it.** `ff_paramstest_from_auto_result` replaces
   geometry, distortion, `px` and `RawFolder`, but **not** `RhoD`; a bad value in
   the template survives untouched. `--mode ff` rewrites it explicitly for this
   reason.
3. **A pixel value looks plausible.** The GUI seed file for this beamtime carried
   `RhoD 2064.118261`, which is the corner distance **in pixels**. That is
   ~150× too small, which makes the polynomial explode rather than go limp.

**The check, after `hkl` has run:**

```bash
awk 'NR>1{print $5}' <result>/LayerNr_1/hkls.csv | sort -n | tail -1   # must be < 500
```

500 is `MAX_N_RINGS` in `midas_index/c_src/IndexerUnified.c`. Builds from
2026-08-16 onward skip out-of-range rows and warn; before that they were written
unbounded, through `RingTtheta` and into the `data`/`ndata` bin pointers, after
which every seed matched nothing. If the count is over 500 and you cannot
upgrade, fix `RhoD` — do not cap `hkls.csv` by hand.

Full evidence chain — the three-way isolating control, the memory layout, and the
confounded parameter sweep that nearly closed the investigation — is in
**Lab Notebook §8a–§8b**.

Validation catches it without running anything:

```bash
midas-params validate <paramsfile> --path ff
```

---

## 10. Parameter-file reference

Full key list: `FF_Parameters_Reference.md`. Keys this runbook depends on:

| key | note |
|---|---|
| `SkipFrame` | 1 at 1-ID; applies to every multi-frame file and the dark (§3e) |
| `dataDataset` / `darkDataset` | both `exchange/data` for DM files (§3d) |
| `OmegaStart` / `OmegaStep` | negated vs the log for `aero` (§2); `OmegaStart` describes **raw** frame 0 |
| `ImTransOpt` | must match the calibration (§6) |
| `LatticeConstant` / `SpaceGroup` | the **sample's**, not the calibrant's (§6) |
| `RhoD` | µm; distortion normalisation radius |
| `Rsample` / `Hbeam` | generous grain-position search bound — **never set to the real sample size** (§6) |
| `RingThresh <ring> <val>` | per-ring detection threshold; set it with `midas-ring-thresh` (§6b), never from a template |
| `MinPeakSNR` | float, default 0 = off. Minimum local SNR to keep a detected peak (§6c). FF **and** PF |
| `BgSubtract` / `BgNSectors` | 0/1 (default 0) + azimuthal cells per ring. Removes a varying background before thresholding. Only helps where the background actually varies — it did **not** on this beamtime (Lab Notebook §6a) |

---

# XRD-CT runbook — worked end-to-end on 11-ID-C CeO₂

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md). Beamline facts:
> [`BEAMLINES.md`](BEAMLINES.md).

A from-scratch run on a real dataset, with every number re-derivable. **Verified 2026-08-18** on
`11idc`; commands were executed, not written from the API.

Use it as the shape of a run, not as a script to paste — the gates matter more than the
commands.

---

## 0. Reach and environment

```bash
ssh 11idc                    # hop authenticates as 11idcuser@chiltepin, NOT bare chiltepin
PY=/home/beams12/S1IDUSER/opt/envs/midas/bin/python     # full path: conda is not on the PATH
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # before any fan-out
```

20 cores, 30 GB RAM. `sentosa` sees `/home/beams/11IDCUSER/...` directly if you need more.

## 1. Survey (phase 0)

```
~/MIDAS/wd/TomoData/     5056 frames + CRLF sidecars, 157 GB
14 translations (hxz 0.000 -> 1.300 mm, 100 um step) x 361 omega = 5054 projections
2880 x 2880, 150 um pixels, qxrd "Subtracted Data" float32 WITH NEGATIVES
```

**Rings are continuous** → XRD-CT, not scanning-3DXRD. Scope gate passes.

**ω sign: UNDETERMINED, and probably undeterminable here.** Both signs give identical diameter
and CV, differing only by a mirror. A near-symmetric rod carries almost no handedness
information. **Record it as undetermined and report the map as mirror-ambiguous** — do not pick
a sign to clear the spine's halt condition.

## 2. Calibrate — unseeded (phase 1)

`calibrate_tomodata.py`. **No `initial_Lsd`, no `BC_guess`.** Bounds in detector pixels only
(350–1200 px), which are geometry-free and so cannot bias the distance.

Result — `calib/geometry_for_dt.json`:

| | value |
|---|---|
| **Lsd** | **1 632 201 µm** |
| BC (y, z) | 1445.976, 1438.596 px |
| tx | **0.0, held** — a powder cannot see it |
| ty, tz | 0.0108°, 0.2051° |
| λ | 0.11595 Å (106.9 keV) |
| **post-residual strain** | **66.2 µε** ✓ (gate: < 100) |
| in-loop strain | 69.8 µε |

**★ The distance the metadata claims is 1600.0000 mm, and the beamline calibration says
1579.5 mm. Both are wrong.** The calibrant filename even carries the nominal value
(`CeO2_FB_D1600-000000.tif`), so the name is not evidence. The survey's independent ring
measurement gave 1632 mm and the unseeded fit returned 1632.2 — and that agreement counts only
because the two were kept apart.

Look at `overlay_full.png` plus zooms on the innermost, a middle and the outermost ring. Then
`build_residual_corr=True` → `residual_corr.bin` (66 MB); it is **silently ignored below
`midas-integrate-v2` 0.3.2**.

## 3. Reduce to a cake, once (phase 1a)

`build_cache.py`. **157 GB → 0.62 GB.**

```bash
$PY build_cache.py --workers 16 --out cake_cache_resid.h5
```

R 350–1200 px at 1 px (850 bins) covers all nine reflections; η 36 bins of 10° rebins down
later. **Cache the whole range** — caching one ring forces a full re-read the moment anyone asks
about a second.

**I/O bound at ~260 MB/s**: 64 workers gave 8.0 frames/s against 6.7 serial. Pin BLAS threads
*before* importing numpy, and build the integration matrix once per worker in the pool
initialiser.

### ★ Verify the axis order. Do not read it from the script.

```
cake  (14, 361, 36, 850)  =  (translation, omega, eta, R)
```

**η before R** — opposite to the 1-ID `.bin` layout. Verified by collapsing each axis on one
frame: axis 0 (len 36) has max/median **1.03** → smooth → η; axis 1 (len 850) has max/median
**181** with sharp peaks → R.

`build_cache.py` itself has `n_r, n_eta = probe.shape`, which is **backwards**, and the file
carries an attr recording that the label arrays were once written swapped and later fixed. The
*data* was always (η, R). Read the `reduce.py` field comment (`intensity: (n_eta, n_r)`) or run
the two-line check.

**Never re-read the TIFFs after this.** Fitting all 5054 projections off the cache takes 1.4 s
on 16 workers.

## 4. Sinogram and rotation axis (phase 1b)

The (111) ring, η-summed, over 16 radial bins → sinogram `(361 ω, 14 translations)`:

```
translation profile (%):  0.7  0.7  2.4  6.2  9.4 10.8 11.7 12.2 12.0 11.4 10.1  7.7  3.5  1.1
centre of mass:           column 7.132 of 0..13   ->   0.7132 mm
```

Consistent with the independently recorded column centre of 0.718 mm.

Then `find_centre(stack, method="com+sweep")` — **two estimators, and it flags disagreement**
rather than preferring one. On a near-symmetric object expect them to disagree, and read that as
the centre genuinely not being well determined.

`RECON_SIGN` is **+1**. It was −1 in the 2023 driver, which **inverted every map** — and an
inverted map is a plausible microstructure with reversed contrast, not an obvious error.

Run `--compare` once: it reports the per-output discrepancy between branches and is the cheapest
way to find out whether a non-additive output (`RMEAN`, `SigmaG`, `SigmaL`, `MixFactor`) is being
back-projected when it must not be.

## 5. Extract per azimuth (phase 2)

**Verified on the real cake with the promoted `midas_dt.azimuthal`**, one (translation, ω) frame,
`max_half_px=16`, `block_bins=30`. 444/850 radii (52 %) are ring-free for the background.

| ring | R px | half | n_max | **peak/bg** | SNR/η | **half-corr** | strain 5–95 % (µε) |
|---|---|---|---|---|---|---|---|
| 111 | 404.0 | 16 | 1 | **137.5** | 198.9 | **+0.219** | 16 |
| 200 | 466.6 | 16 | 1 | 53.5 | 101.9 | −0.024 | 651 |
| 220 | 660.3 | 16 | 1 | 133.1 | 158.2 | −0.121 | 255 |
| 311 | 774.7 | 16 | 1 | 103.8 | 144.1 | **−0.707** | 178 |
| 222 | 809.3 | 16 | 1 | 22.5 | 57.7 | **+0.265** | 562 |
| 400 | 935.2 | 16 | 1 | 25.1 | 57.9 | −0.610 | 455 |
| 331 | 1019.6 | 12 | 1 | 46.6 | 92.1 | −0.740 | 172 |
| 420 | 1046.3 | 12 | 1 | 30.1 | 72.3 | −0.717 | 203 |
| 422 | 1146.9 | 16 | 1 | 38.8 | 85.5 | **−0.737** | 0 |

### ★ CeO₂ is HIGH-contrast data — the opposite regime from the DAC Ti

**peak/background 22–137**, against **0.005–0.17** on the Ti scan. Three to four orders of
magnitude apart, so none of the low-contrast area-vs-centroid pessimism applies here. **Measure
your own contrast; do not inherit a regime.**

**But contrast is only gate one.** This dataset passes it comfortably and still fails the
**grain-count** gate (`ENVELOPE.md` §0a) — which is exactly what makes it a useful worked example.
High contrast does **not** imply the azimuthal pattern is usable.

### ★ Read 111 (+0.219) and 222 (+0.265) as NO SIGNAL, not as amplitude variation

The planted-truth control puts the no-signal baseline at **+0.02 … +0.24** (rising with window
width) against **+0.99** for genuine amplitude modulation. Both of these sit in the baseline band.
See `phase-2-extract.md` §2.5 for the three-band reading.

### ★ The −0.72 anti-correlation reproduces, and it is NOT window truncation

Five rings sit at **−0.61 to −0.74**, reproducing the independently recorded −0.72. The boring
explanation was tested and **excluded**:

* **FWHM is 2–4 px** for seven of the nine rings, against a 32-px window — 8–16× wider than the
  peak. Nothing is being truncated.
* **The correlation is stable across a 5× window sweep** (half-width 6 → 32 px): 422 moves
  −0.803 → −0.641, 311 moves −0.737 → −0.660. Truncation would be killed by widening; this is
  not.

So the movement is **real**. But it is **not** the cause of the azimuthal *area* structure —
a window wider than ~2× FWHM has an area invariant to sub-pixel movement (`phase-2-extract.md`
§2.5). Two separate phenomena, and only one of them threatens a texture fit.

### ★ But the pattern across hkl is unexplained — do not invent a story

The obvious reading, that the effect grows with R because a given strain shifts outer rings
further, **is wrong**: **311 (−0.707) and 222 (+0.265) are adjacent in R and opposite in sign.**
It is not a smooth function of R, and it does not track the standard cubic orientation factor
either (200 and 400 both have Γ = 0 yet read −0.024 and −0.610).

**Resolved 2026-08-20, and not the way this section expected** — see `LAB_NOTEBOOK.md`
`LAB_NOTEBOOK.md` §5b-ter (provisional). Peak-fitted areas were **not** the answer: a window wider than ~2× FWHM has
an area invariant to sub-pixel movement, so movement cannot be what put structure into these areas.
The cause is **finite crystallite counting** — a sample limitation, not an analysis defect. The
hkl pattern of the half-correlation remains unexplained.

### ★ 331 and 420 are unusable at this geometry

They sit **26.7 px apart** — closer than the 32-px windows they would want. `ring_windows`
correctly narrows both to half=12 on the gap rule, but their apparent FWHM comes out at **29 px**
against 2–4 px for every clean ring: that is the neighbour inside the window, not a broad peak.
`count_maxima` still calls each a singlet, because within its own narrowed window each *is* one.

**Drop the pair.** A gap check between adjacent ring centres is a separate test from a
within-window multiplet test, and this dataset shows why you need both.

## 6. Strain (phase 3)

Azimuthal 5–95 % spreads of 0–651 µε across the nine rings, **with 331/420 excluded** per §5.

**On a strain-free calibrant powder these should be small**, and 651 µε on (200) is not. Read
that together with §5: where the half-correlation is strongly negative, the centroid is
responding to a peak that is moving for reasons not yet established, so treat these as
**diagnostic, not as a strain measurement**, until the movement is explained.

Use `strain_from_centroid` with `reference_d` unset → **relative** strain, median-referenced.
That is right here: the distance is confirmed, but ω is undetermined, so the map is
mirror-ambiguous and an absolute scale would imply more certainty than exists.

## 7. Texture (phase 4) — NOT attemptable, and now for a known reason

At peak/bg 22–137 the **contrast** gate passes comfortably. The **grain-count** gate does not:

* the powder null on this dataset is **REFUTED** (`LAB_NOTEBOOK.md` §5b) — spurious structured
  texture on a sample that should have none, residual flat in `L`;
* the cause is **UNIDENTIFIED**. Grain counting was proposed and **refuted** — the supporting
  chord-exponent argument turned out to be an artefact of the analysis (`LAB_NOTEBOOK.md` §5b-ter). What survives is
  that the floor is real and not a sampling artefact;
* **more ω helps but does not clearly finish the job.** Averaging removes the bulk; a fitted
  ω-locked component of ~0.87 % remains. Note the floor is an **extrapolation** — the curve is
  still falling at N_ω = 256 (ring 111 reaches 0.664 %, ~8 % above the fitted asymptote), so
  treat 0.87 % as a fitted value, not an observed plateau.

So the honest status, **updated 2026-08-20**: CeO₂ is neither the demo *nor* a pipeline
test. Its spurious texture is **grain-count statistics** (`LAB_NOTEBOOK.md` §5b-ter, provisional),
i.e. a property of the sample, so there is no pipeline defect here to fix against. Use it to
calibrate the *gates* — it is an excellent worked example of high contrast failing the grain-count
gate — not to validate a texture method.

## 8. Report (phase 5)

Minimum for this dataset:

```
Distance    1632.2 mm, refined UNSEEDED from the data
            (metadata 1600, beamline calibration 1579.5 -- both wrong)
Calibration post-residual strain 66.2 ue (gate < 100), overlays inspected
omega       UNDETERMINED -> map is MIRROR-AMBIGUOUS
Cake        (14, 361, 36, 850) = (translation, omega, eta, R), axis order VERIFIED
Rings       9 indexed; 331 + 420 DROPPED (26.7 px apart, mutual contamination)
Contrast    peak/bg 22-137 -- high-contrast regime
Diagnostic  half-correlation -0.61..-0.74 on five rings; window truncation EXCLUDED
            by a 5x width sweep; hkl pattern UNEXPLAINED
Strain      relative, median-referenced; treat as diagnostic pending the above
Texture     NOT attempted -- the powder null on this dataset is refuted
```

## 8b. Current state — the pick-up point

**As of 2026-08-21.** Read this before re-deriving anything; each line is
recorded in `LAB_NOTEBOOK.md` with the entry that established it.

| what | state |
|---|---|
| 11-ID-C CeO₂ geometry | **SETTLED.** `Lsd = 1632 mm` measured **from the data**; the metadata (1600 mm) and the beamline calibration (1579.5 mm) are both wrong. Do not re-derive from either. |
| ω sign | **UNDETERMINED** — the map is mirror-ambiguous. Not resolvable from this dataset. |
| Rings | 9 indexed; 331 + 420 dropped (26.7 px apart, mutual contamination). |
| Half-correlation −0.61..−0.74 on five rings | **UNEXPLAINED.** Window truncation is EXCLUDED by a 5× width sweep. The hkl pattern is open. |
| Strain | relative, median-referenced. **Diagnostic only** pending the line above. |
| Texture | **NOT attempted.** The powder null on this dataset is refuted (§5 of the notebook), so a per-voxel ODF is not reportable at any spatial scale here. |
| DAC Ti S1 | **OUT OF SCOPE** — coarse-grained (~4 grains per 0.3° column), i.e. pf-HEDM not XRD-CT. All strain/texture claims on it are refuted; geometry and 6.2 GPa stand. |

**Where the next session should start:** the hkl pattern behind the
half-correlation. Everything upstream of it is settled; everything downstream
of it is labelled diagnostic *because* of it.

## 9. Files

| | |
|---|---|
| Frames | `~/MIDAS/wd/TomoData/` |
| Calibration | `~/MIDAS/wd/dt_survey/calib/{geometry_for_dt.json, calibration.json, residual_corr.bin, overlay_*.png}` |
| Cakes | `~/MIDAS/wd/dt_survey/cake_cache.h5`, `cake_cache_resid.h5` |
| Scripts | `~/MIDAS/wd/dt_survey/{calibrate_tomodata.py, build_cache.py, localise.py, attribute.py, dmap.py}` |
| Survey plan | `~/MIDAS/wd/dt_survey/PLAN.md` |
| Worked notebook | `packages/midas_dt/notebooks/02_real_data_end_to_end.ipynb` (executed, with outputs) |

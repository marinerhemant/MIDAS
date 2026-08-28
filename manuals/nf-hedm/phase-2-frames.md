# Phase 2 — Look at the raw frames before building anything

> Part of the **NF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 5. STEP 4 — Look at the raw frames before building anything

### 5a. Pattern: reduce remotely, plot locally

The shared env has no matplotlib (§1). Working example from this session:

```
remote reducer  -> writes au4_reduced.npz  (keys: detz{9,13}_{max,med,f0,f360,f719})
scp to Mac
$ANALYSIS/bt_1id_jun25_nf/plot_au4.py       -> 4 PNGs
$ANALYSIS/bt_1id_jun25_nf/check_artifacts.py -> the null-model check
```

Both scripts live in that campaign directory — **`$ANALYSIS` is not this repo** (README),
so they travel with the campaign, not with a checkout. `check_artifacts.py` re-runs in
seconds and reproduces every number in §5b. Run it before trusting any spot count:

```bash
export ANALYSIS=<the campaign dir; NOT in this repo, see README>
cd "$ANALYSIS/bt_1id_jun25_nf"
conda activate midas_env            # or whichever env carries this project
python check_artifacts.py
```

### 5b. Reference sanity numbers for a real 1-ID NF dataset

`Au4_cubes_nf_96keV`, 2048² uint16, px = 1.48 µm, 720 frames/distance, 2 distances.
All re-derived on 2026-07-29 by `check_artifacts.py`:

| quantity | DetZ 9 mm | DetZ 13.0001 mm |
|---|---|---|
| temporal-median background, mean | **6.68 counts** | 6.53 |
| median background, left half / right half | 3.16 / **10.20** | 2.99 / 10.06 |
| brightest pixel, single raw frame (frames 0/360/719) | 939 / 933 / 688 | 935 / 794 / 850 |
| brightest pixel, max-projection | 1294 | 1441 |
| brightest pixel, temporal median | 260 | 243.5 |
| static-hot pixels (median > 50) | **347** | 328 |
| static-hot set overlap | intersection 328 → **Jaccard 0.945** (i.e. the same pixels) | |
| max-projection px > 100 | **8423** | 8647 |
| of which coincide between distances | **152 → Jaccard 0.009** | |
| of which are static-hot | **164** | 139 |

Blob-size histogram of max-projection px > 100 (DetZ 9): **4893 blobs of 1 px, 435 of
2–3 px, 9 of 4–9, 4 of 10–29, then 17 blobs ≥ 30 px** (8 of 30–99, 9 of ≥ 100), totalling
2535 px. DetZ 13 gives 5149 / 427 / 3 / 4 / 17 — the same structure.

**Interpretation, and the operational rule.** The starfield is **cosmic rays**, not hot
pixels and not spots: Jaccard 0.009 between distances means those bright pixels are
transient, and only 164 of 8423 are static-hot. The **17 blobs ≥ 30 px are the real Bragg
spots.** A naive `npix > threshold` count on a max-projection therefore overestimates spot
content by ~500×.

**Rule: never count spots off a raw max projection.** Subtract the temporal median, then
LoG + connected components (`DoLoGFilter 1`), then filter by blob area.

**But `DoLoGFilter 1` is NOT an unconditional production default.** Operator knowledge
(1-ID, 2026-07-30): *the LoG path sometimes kills real signal*, and on weak-scattering
samples that loss matters more than the cosmic-ray suppression it buys. The rule above is
about **counting spots for a sanity check**, where you must not be fooled by cosmics; it is
not a blanket instruction for the production reduction.

Decision guide:

| Situation | Setting | Why |
|---|---|---|
| counting spots / auditing frame content (§5b) | `1` | cosmics dominate a raw max-projection ~500:1 |
| strong scatterer, dense spots (e.g. the Au calibrant) | either; Au was reconstructed to confidence 1.000 with `0` | |
| **weak signal** (e.g. `nf_sampleB_htB_s2`) | **`0`** | LoG can suppress genuine weak peaks; cosmics are then left in, and must be tolerated downstream |

If you change this key you **must regenerate `SpotsInfo.bin`** — it is baked into the
reduction, not applied at fit time.

Note the dynamic range: background ≈ 6.7, single-frame spot peaks ≈ 700–950, i.e. **~2
decades**. The §5b reductions were computed over 72 of the 720 frames (every 10th) for the
max projection and 18 frames for the temporal median
(`$ANALYSIS/bt_1id_jun25_nf/plot_au4.py:49,65`).

### 5c. Decision tree on what you see

| observation | conclusion | action |
|---|---|---|
| median background ~5–10 counts, single-frame spot peaks ~700–1000 | normal | proceed |
| left/right halves differ ~3× in the median | detector panel asymmetry, expected here | do not "correct" it; `BlanketSubtraction` after the temporal median is the intended knob |
| thousands of 1-px bright dots in max-proj | cosmic rays | ignore; they die in the median + LoG path |
| a few hundred pixels bright in the *median* | fixed hot pixels | expect ~330 at 2048²; they persist across distances |
| max-proj has < 10 blobs ≥ 30 px | too few spots to index | check ω range, energy, and that you have the right scan |
| `tifffile.imread` returns 3-D | multi-page TIFF | wrong layout for this code (§3f) |

### 5c-bis. If confidence has a ceiling: break it down PER DISTANCE, then profile in RADIUS

`hard_fraction` counts a predicted spot only if it is observed at **every** distance
(`hits_d.prod(dim=0)`, `obs_volume.py:395`). So the reported confidence equals the **worst
distance** and tells you nothing about which one. Before theorising about geometry, forward
simulate the best voxel and score each distance separately — on `nfdev_jul26` that turned an
opaque "0.717" into `71.7 % / 91.3 % / 100 %` and localised the whole deficit to the near
distance in one step (lab notebook §7h).

**Then profile in RADIUS about BC, not in rows.** A beamstop is a **disc centred on the
beam**, so every detector row keeps unobstructed columns far from BC and a row profile —
row-max, row-mean, or row occupancy — is **structurally blind to it**. On `nfdev_jul26` four
row-based tests all came back negative before a radial profile found the stop immediately:
R ≈ 1100-1240 px ≈ 600-680 µm, fixed in the detector plane.

The signature to recognise: **rings vanish by RADIUS, not by index**, and the deficit grows
as `Lsd` shrinks, because a given ring moves inward at the near distance. The **strongest**
reflections go missing first ({111}, {200} in FCC) precisely because they sit at the
smallest radii — an inner-ring deficit that looks backwards is the tell.

**Blocked reflections still count in the denominator.** `MaxRingRad` is an OUTER limit only;
there is no inner radial exclusion, so a beamstop caps confidence permanently and **no
geometry refinement can lift it** (multi-point over 12 voxels, objective resolution 1/552,
failed to improve at all). Drop the affected rings with `RingsToUse` instead of dropping the
distance — you keep the geometry leverage of all distances.

**Chance-rate discipline for any "is there a spot near here?" search.** With `N` lit pixels
in an `H×W` frame the density is `N/(H·W)`; a search radius `R` contains `πR²·N/(H·W)` lit
pixels **by chance**. At 3800 lit px on 4600×5320 that is ~1.75 within ±60 px and ~78 within
±400 px — so "found something nearby" is the null result, not evidence. Compute it first.

### 5d. Check the counting regime before choosing a threshold

**First fix the units the threshold is denominated in.** Every number in this section is
in ADU, and whether a raw value *is* ADU is a per-**scan** fact that is never inferred:

```python
np.unique(frame)[:8], frame.max()
#  multiples of 64, max 65472  -> 10-bit stored x64 -> PixelScale 64
#  gap of 2 or 4,   max 4092   -> 12-bit unscaled   -> PixelScale 1  (the default)
```

Set `PixelScale` (§10f) and let the reader divide; do not divide in an analysis script,
or the paramfile and your plots disagree. **Do not carry the answer over from another
scan** — `nfdev_jul26` is ×64 and `NF_Au_cube_0802` is unscaled *on the same detector
serial*, and the SS316L tomography taken the same day as the unscaled NF scan is ×64
again. Get this wrong and the "2 counts" in the table below becomes 128 counts, which
sits above the pedestal and makes the **background** read as signal — the failure looks
like a sample that indexes everywhere rather than like a threshold error (§3h,
lab notebook §8b).

`BlanketSubtraction ≈ 0.7 σ` (§8f) assumes σ is meaningful. **On a photon-starved detector
it is not.** Measure it before trusting it:

```python
res = frame_ADU - temporal_median_ADU          # after clamping at 0
print("frac exactly zero:", (res == 0).mean())
print("MAD:", 1.4826 * np.median(np.abs(res - np.median(res))))
print("counts at 1,2,3,4,>=5 ADU:", [(np.round(res[res>0])==k).sum() for k in (1,2,3,4)])
```

On `nfdev_jul26` (20-ID, 63.3 keV, 0.2 s) this returned **99.734 % exactly zero, MAD
exactly 0**, and of the nonzero pixels 4954 at 1 ADU against 41 at ≥5 ADU. There, `0.7 σ`
collapses onto whatever floor the code clamps σ to and admits the entire single-count
floor as signal.

| regime | test | threshold |
|---|---|---|
| noise is Gaussian-ish, MAD > 0 | MAD is a real number | `0.7 σ` per §8f |
| **photon-starved**, residual mostly exactly 0, **MAD = 0** | as above | **absolute**, in counts. 20-ID production value: **2 counts after median + NLM** |

**And do not pair NLM with a sub-ADU threshold.** NLM spreads an isolated single count
over its patch; the result then clears a 4-px minimum-area cut and is counted as a spot.
NLM plus an *absolute* threshold is correct; NLM plus a σ-derived one manufactures spots.

**Before running any spot-matching statistic, compute its random-coincidence value.** For
N spots placed at random on an `H×W` frame the median nearest-neighbour distance is
`0.4699/√(N/(H·W))`. At N = 35 on 4600×5320 that is **393 px** — so a nearest-neighbour
matching test at that spot density measures nothing but chance (lab notebook §7f, F2).

### 5e. Before ANY ring analysis: is this sample narrow enough to have rings?

An NF spot does **not** land at `Lsd·tan(2θ)` from BC. It lands at

```
p = (grain position, projected)  +  Lsd · tan(2θ) · d̂
```

The first term is the reason NF resolves grains at all, and it is also why NF data is
**not a powder pattern**. Every ring is convolved with the illuminated width of the
sample. Rings survive only if

```
illuminated width (px)   <   ring spacing (px)
```

Measure both before fitting anything radial:

```python
# ring spacing, innermost pair
r_hkl = lambda a, N: LSD * np.tan(2*np.arcsin(LAM/(2*a/np.sqrt(N)))) / PX
spacing = r_hkl(a, 4) - r_hkl(a, 3)          # 200 minus 111

# illuminated width: equivalent top-hat of the absorption profile in the beam stripe
a_prof = np.clip(1.0 - frame_stripe/ref_stripe, 0, None)
width  = a_prof.sum() / a_prof.max()
```

Measured at 20-ID, 63.314 keV, px 0.548 (111→200 spacing **225 px**):

| sample | illuminated width | smearing | rings? |
|---|---|---|---|
| `nfdev_jul26` Au cube | 128 px = **70 µm** | ±64 px = 0.6× spacing | yes |
| `bt_20id_jul26b` `nf_sampleD` | 452 px = **247 µm** | ±226 px = **2.0× spacing** | **no** |

**The control that catches it when you forget.** Two layers of the same material at the
same distance must give **identical** ring radii. On `nf_sampleD` the two 9 mm layers, 10 µm
apart, gave radial "peaks" at 1028/1152/1294/1364/1496 and 1108/1212/1304/1386/1574. They
are not rings; they are grain-sampling structure, and every δ and lattice parameter fitted
to them was void (lab notebook §8b).

A second, independent check: histogram **spot radii** (each spot counted once, not weighted
by size or brightness) and compare the on-ring count against the local off-ring rate. On
`nf_sampleD` the excess was +0.2 % and +2.4 % at the two candidate lattice parameters —
i.e. consistent with **no rings at either**.

⇒ When this test fails, the lattice parameter cannot be read off the detector. **The
indexer is the arbiter** — run the reconstruction once per candidate `LatticeParameter`
and let confidence decide.

---

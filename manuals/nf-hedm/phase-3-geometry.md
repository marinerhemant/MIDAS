# Phase 3 — Beam centre, distance, and calibrant refinement

> Part of the **NF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 6. Detector-distance and rotation-axis calibration (DetZBeamPos)

Validated on `bt_1id_jul26` (95.0000 keV, Retiga, px 1.48 µm) against an independent
operator reading taken in `gui/nf_qt.py`. Everything below is measured, not inferred,
except where explicitly flagged.

### 6a. The split — neither measurement alone is enough

| measurement | gives | CANNOT give |
|---|---|---|
| **DetZBeamPos** (this section) | `BC` per distance, β (beam tilt vs the DetZ stage axis), sample position w.r.t. the rotation axis | **absolute Lsd** |
| **NF spot backprojection** (§6i) | **absolute Lsd** (the DetZ offset δ) | β |

Why DetZBeamPos cannot give Lsd: the sample sits on the rotation axis, and a point on the
beam axis casts its shadow at `BC` *regardless of L*. The beam is parallel (verified: shadow
width constant to 0.3 px across four distances), so there is no magnification cue either.

Why spots cannot give β: a tilt common to every ray is absorbed into the measured ray
slopes. See §6i for the derivation.

**Run DetZBeamPos first, then spots.** BC and the sample position come out of this section;
Lsd comes out of the spot data; together they close the geometry.

### 6b. Pixel convention — VALIDATED, do not re-derive it

```
ybc = 2047 − raw_column_index          zbc = 2047 − raw_row_index
```

where `raw_*` are indices into the array as `tifffile.imread` returns it.

Provenance, in order of strength:

1. `gui/nf_qt.py:1305` applies `self.imarr2 = self.imarr2[::-1, ::-1].copy()`
   **unconditionally**, on both the TIFF and `.bin` load paths — a 180° rotation, i.e. both
   axes reversed.
2. The cursor readout indexes into *that reversed array* (`gui/gui_common.py:714-731`:
   `mapSceneToView` then `_raw_data[iy, ix]` with `ix = int(x+0.5)`). `origin='br'` only
   calls `vb.invertX()`, which changes the drawn axis direction, **not** data coordinates.
3. **Empirical confirmation.** On `bt_1id_jul26_95keVBeamPos_redH_NoBBwithAu0_000267.tif`
   (DetZ 7 mm), this pipeline measured (997.41, 38.26); an independent operator visual read
   in `nf_qt.py` gave **(997.7, 38.2)**. Agreement 0.29 px in y, 0.06 px in z.

**The constant is 2047, not 2048.** It is the plain index reversal on a 2048-long axis. The
2048 form is off by exactly one pixel (39.26 vs the measured 38.2 above) — which matters,
because `BCTol` in `ps_au.txt` is 0.2 px in z.

MIDAS consumes these as one entry per distance: `ybc`/`zbc` are lists and
`midas_nf_fitorientation/params.py:357` raises if `len(ybc) != n_distances`. The forward
model is `y_pixel = yBC + ydet/px`, `z_pixel = zBC + zdet/px`
(`midas_diffract/forward.py:1283-1284`, NF runs `flip_y=False`), and the image stack is
`[N, Z, Y]`, so **zbc pairs with the row axis and ybc with the column axis** in the reversed
frame above.

### 6c. Locate and decode the scan

Logs live in the acquisition-log folder (§3a), one per condition:

```bash
ls ~/new_data/<beamtime>/*BeamPosScan.txt
```

Condition names decode as:

| token | meaning |
|---|---|
| `NoBB` / `withBB` | beam block **out** / **in** |
| `NoAu` | **no sample** — this is the direct beam |
| `withAu0`, `withAu90`, `withAu180` | sample in beam at ω = 0, 90, 180° (as logged; apply the §2 sign rule before using ω for anything else) |
| `redH` | reduced beam height |

The macro line gives the distance series; the per-block summary lines give image → DetZ:

```bash
cd ~/new_data/<beamtime>
grep -E "^Ran|^Image #" <bt>_<E>keVBeamPos_NoBBNoAu_BeamPosScan.txt
#   Ran : DetZBeamPos( <startDetZ> <endDetZ> <step> <exposure> <prefix> )

# image -> DetZ, one row per position:
grep -E "^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\.[0-9]+" \
     <bt>_<E>keVBeamPos_NoBBNoAu_BeamPosScan.txt | awk '{printf "%s->%s  ", $1, $3}'
```

**Extract this per condition and never assume it from the macro** — `withBBNoAu` in
`bt_1id_jul26` was three *separate* macro invocations with a different exposure, so its
mapping does not follow the others.

Images for all conditions usually share one folder, e.g.
`data/nf/Au_DetZBeamPos_95keV/<prefix>_<image:06d>.tif` (zero-padded to 6, §3).

### 6d. STEP A — zbc from the direct beam

The beam is vertically focused, so on the detector it is a **thin horizontal stripe**
(~10–15 rows). That makes the vertical centroid sharp and unambiguous.

For each DetZ, on the **`NoAu`** condition:

```python
p = a.mean(axis=1) - np.median(a.mean(axis=1))     # row profile, background removed
idx = np.where(p > 0.2 * p.max())[0]               # stripe rows
lo, hi = idx.min(), idx.max()
rows = np.arange(lo - 6, hi + 7)                   # pad, then intensity-weighted centroid
seg = np.clip(p[rows], 0, None)
row_c = (rows * seg).sum() / seg.sum()
zbc = 2047 - row_c
```

Do it for **both** beam heights if available — they must agree (see §6g).

### 6e. STEP B — ybc from the sample shadow at ω = 0 and 180

**The direct beam cannot give ybc.** Horizontally the stripe is a broad, slit-defined band
(833 px wide at full height in the reference dataset). Its centre is the centre of the
*illuminated region*, which is set by the slits and has nothing to do with where the
rotation axis is.

The sample gives it. A sample off the axis by `u` projects to `+u` at ω = 0 and `−u` at
ω = 180, so

```
axis = ( dip_centre(ω=0) + dip_centre(ω=180) ) / 2
```

cancels the sample's own offset **exactly**. Since the sample sits on the axis and a point
on the beam axis shadows at `BC`, this axis position **is** `BC_y(L)`.

#### 6e-0. Two preconditions — check both before trusting any ybc from a shadow

**(1) `band_frac` must select the beam's flat core, not its wings.**
`beam_calib.shadow.track_shadow` defaults to `band_frac=0.30`. At 20-ID that admits the
dim wings, the dip finder wanders, and the axis comes back **+100 to +130 px wrong** with a
clipped amplitude. Tuned against the *known* `nfdev_jul26` answer (axis col 2625.47):

| `band_frac` | axis col | amplitude | rms | `is_reliable` |
|---|---|---|---|---|
| 0.30 | 2721.15 | 634 px | 216 px | False |
| 0.50 | 2787.83 | 500 px | 202 px | False |
| **0.70** | **2625.88** | **918 px** | **1.5 px** | **True** |

At 0.70 the amplitude 918 px = 503 µm also reproduces the independently measured 496.8 µm
cube-2 offset. **Never adopt a shadow axis without first reproducing a known one**, and
branch on `fit_axis(...).is_reliable` — it returned False on every failing row above.

If the beam profile has a narrow bright spike (there is one near col 3600 in
`bt_20id_jul26b`), `band_frac * ref.max()` selects only the spike and everything is reported
as clipped. Crop to the flat core first and add the offset back.

**(2) The absorber must be COMPACT.** The method assumes the shadow centre traces a rigid
sinusoid, which is true for a particle and false for an extended irregular specimen. On
`nf_sampleD` the shadow width swung 56→886 px with ω, and `fit_axis` refused at **every**
setting. There is no on-axis feature to fall back on either (`find_stationary` returned a
fixed 26 µm absorber at col 3408, not the sample).

**(3) Turning a shadow into a particle POSITION — use the annulus, not the point.**
`ShadowTrack.position_candidates_um` returns the sample-frame `(x, y)` as
`(−a sin φ, −a cos φ)`, i.e. **θ = −φ − 90°**, with the antipode second as a fallback
(`beam_calib/shadow.py:115-153`). That relation is a *measured convention* — 20-ID, aero,
ω = −θ — pinned on two campaigns that located the same off-axis Au cube to
`θ + φ = −90°` within 0.43°. It replaced an earlier `(a cos φ, a sin φ)` form that was
wrong **in form, not sign**: the true position is 90° away, so neither the point nor its
antipode was near the particle, and a candidate-point tomo mask returned **exactly 0.0000
at every off-axis voxel** on both campaigns — indistinguishable from "the particle is
absent." Corrected, it predicts both reconstructions to 3 µm; the old form was out by
985 µm (`8a5f0184`).

> **On a new beamline, do not build a point mask from this.** The relation encodes an ω
> sign and a detector handedness. Mask the full **annulus** at `amplitude_px` — the radius
> is convention-free — and let the reconstruction pick the angle. Two campaigns agreeing
> is evidence, not proof.

⇒ When both routes fail, ybc is **not measurable from this scan**. Inherit it from a
campaign that measured it at the same nominal distance, **mark it inherited in the
paramfile**, widen `BCTol` in y, and let §7 refinement move it. Do not present the
inherited number as measured.

Build the transmission profile along the stripe, using the matching `NoAu` image at the
same DetZ as the reference:

```python
lo, hi = stripe_rows(ref)                     # from the NoAu image, as in 6d
Iref = ref[lo:hi+1, :].sum(axis=0)
I    = au [lo:hi+1, :].sum(axis=0)
band = Iref > 0.05 * Iref.max()
T    = np.where(band, I / Iref, np.nan)
```

> **ESTIMATOR TRAP — this is the one that will bite you.** Do **not** centroid `1 − T` over
> the illuminated band. The shadow is a ~28 px dip to T ≈ 0.57, but the band is up to 833 px
> wide with ~2% noise, so the noise integral swamps the dip. Doing it that way on the
> reference dataset gave **66 px of scatter and non-monotonic** axis positions, and a
> sample offset swinging ±29 µm. Corrected, the same data gives **0.2 px** agreement.

Use the **midpoint of the two half-depth edges** — a knife-edge measurement, robust to the
dip's flat bottom:

```python
base, bottom = np.nanmedian(T[band]), np.nanmin(T[band])
half = 0.5 * (base + bottom)
imin = int(np.nanargmin(np.where(band, T, np.nan)))
# walk left and right out of the dip to the half-depth crossing, linear-interpolate each
xl = <left crossing>;  xr = <right crossing>
dip_centre = 0.5 * (xl + xr)                  # sub-pixel
ybc = 2047 - dip_centre
```

Reference implementation: `axis_from_dip.py` (`dip_centre()`), alongside `beam_center.py`
in `$ANALYSIS/bt_1id_jul26_beampos/`.

**ω = 90 is the cross-check**, not an input: `dip_centre(90) − axis` is the orthogonal
component of the sample offset. In the reference dataset it came out −2.0 to −2.8 µm at
every distance and both beam heights.

### 6f. STEP C — fit β, then emit BC per distance

Fit each axis linearly against the **motor readback** (not absolute L — δ is unknown at this
stage, and it only shifts the intercept, leaving β unchanged):

```python
A = np.column_stack([np.ones(n), detz_um])
intercept, beta = np.linalg.lstsq(A, bc_values, rcond=None)[0]
```

Report `BC(DetZ) = intercept + β · DetZ[µm]` and evaluate it at each distance used by the
sample scan. Because β is a property of the beam/stage alignment, **this transfers to DetZ
values the calibration scan never visited.**

> **β MUST be measured per beamtime. Never borrow it.** Borrowing β from `ps_au.txt` (by
> differencing its two `BC` lines across its two `Lsd` lines) and applying it to
> `bt_1id_jul26` was wrong by **62× in y** and **2.1× in z**, with y's magnitude
> underestimated so badly that the horizontal misalignment — which is in fact the *dominant*
> one, 4.6× larger than vertical — looked negligible.

**`BCTol 2 0.2` from `ps_au.txt` is too tight** for a seed carrying any per-distance
uncertainty. If β is measured as above, the seed is good to ~0.5 px and the stock tolerance
is fine. If β is *not* available, seed all distances with the same BC and open BCTol in the
affected axis to tens of pixels.

### 6g. Acceptance gates — check all five before using the numbers

| # | check | reference dataset achieved |
|---|---|---|
| 1 | full-height vs `redH` agree at every distance | 0.13–0.20 px (ybc), 0.10 px (zbc) |
| 2 | BC linear in DetZ, max residual | 0.74 px (y), 0.85 px (z) |
| 3 | sample offset `u` small and *consistent* across distances | −0.4 to −1.2 µm, all 4 distances |
| 4 | shadow width constant across distances (⇒ parallel beam) | 28.1–28.4 px = 41.8 µm |
| 5 | ω=90 offset small and consistent | −2.0 to −2.8 µm |

Gates 1 and 3 are the ones that caught the estimator bug in §6e: the broken estimator
passed neither.

### 6h. Reference numbers — `bt_1id_jul26`, 95.0000 keV, px 1.48 µm

Images 251–285 in `data/nf/Au_DetZBeamPos_95keV/`, DetZ 7/9/11/13 mm, 9 conditions.

| DetZ (mm) | ybc | zbc |
|---|---|---|
| 7 | 997.00 | 38.31 |
| 9 | 1014.01 | 41.83 |
| 11 | 1029.68 | 44.13 |
| 13 | 1043.94 | 48.80 |

```
ybc(DetZ) = 942.91 + 0.007825 · DetZ[µm]        beta_y/p = +0.007825 px/um
zbc(DetZ) =  26.38 + 0.001689 · DetZ[µm]        beta_z/p = +0.001689 px/um
```

Sample on the rotation axis to 0.5 µm; sample width 41.8 µm (ω=0/180) and 47.5 µm (ω=90).

### 6i. Fallback when there is no DetZBeamPos scan

This happens — in `bt_1id_jun25` all three `Au_DetZBeamPos*` folders exist but are
**empty** (§3f). Then BC has to come from the sample's own diffraction spots, with a real
loss of information.

Match one spot across two distances. Its slope is measured exactly and the unknown offset δ
cancels:

```
r_k = Δy_k / ΔD          s_k = Δz_k / ΔD          (ΔD from the DetZ motor)
```

Substituting back at distance 1 leaves a **linear** system in three unknowns
`[A_y, A_z, L1]`, two equations per spot, so N ≥ 2 spots suffice:

```
y_k1 = A_y + L1 · r_k          z_k1 = A_z + L1 · s_k
```

Equivalently, the distance-1 → distance-2 map is a pure radial scaling about `A` by
`k = L2/L1`; two correspondences give `(A_y, A_z, k)` in closed form, which is the RANSAC
hypothesis. **This is where absolute Lsd comes from** (`δ = L1 − DetZ₁`).

What you get and do not get:

- **`L1`, hence δ — yes.** On `bt_1id_jun25` Au this gave δ = 153–178 µm across six
  accepted solves at two energies, bootstrap ±4 µm within a dataset.
- **β — no, structurally.** With a tilt β the true model is
  `y_k(L) = A_y + L·(β + a_k)/p`, and the measured slope `r_k` *is* `(β + a_k)/p`. β is
  absorbed. The fit returns `A`, the projection of the **rotation axis** at L = 0 — which is
  **not** MIDAS's `BC`. They differ by `β·L`: `BC(L) = A + β·L`.
- Therefore **per-distance BC is unrecoverable from spots alone.** Seed one BC for all
  distances and widen `BCTol`.

Mandatory controls, both cheap, both of which caught real problems:

- **Position-scrambled null:** permute the *position* columns of the distance-2 spot list
  independently of (ω, area). Permuting whole rows is a **no-op** — the ω/area gate matches
  on column values, so row order changes nothing, and the "null" silently re-runs the real
  analysis. This bug produced a falsely tight 5.1 µm null scatter before it was found.
- **ω-shuffled null:** pair distance-1 spots with distance-2 spots at a *different* ω. Every
  pair is then physically impossible.

Both must fail decisively. On Au4 they returned **0 of 200** consensus solutions against 136
inliers for the real pairing.

Also gate every solve on: y-only vs z-only `L1` agreement (< 200 µm), `cond(A)` (7–8 when
healthy), and leave-one-out stability. On `bt_1id_jun25` Au3 the `6→7` pair failed five
gates at once (δ = 5012 µm, cond 115, y/z split 947 µm); naively averaging all three pairs
would have given δ = 1786 µm instead of 174 µm — a 10× error.

#### 6i-bis. When you DO have the direct beam *and* spots — the better-posed case

Everything above assumes BC is unknown. If the direct beam is on the detector (20-ID, §3h)
you get `BC(L)` per distance from §6d/§6e **first**, and then the triangulation is strictly
better posed: a ray is `p(L) = BC(L) + L·d`, so the map between two distances is a scaling
about the **respective** beam centres,

```
(p₂ − BC₂) = k · (p₁ − BC₁)          k = L₂/L₁
L₁ = ΔD / (k − 1)                     δ = L₁ − DetZ₁
```

`A` never enters, so §6i's "the fit returns A, not BC" caveat does not apply. **Do not use
one shared centre for both distances.** BC moves with distance by β·L, and at 20-ID
β_y/p ≈ 0.0036 px/µm gives ~14 px between adjacent distances; a common centre biases `k`
by ~0.7 %, which is ~4 % in `k−1` and hence **hundreds of µm in `L₁`**.

Run the same nulls and gates as §6i. **Two spots at the same ω match only if their ray
directions agree**, not merely their radii — match on the angle between `(p₁ − BC₁)` and
`(p₂ − BC₂)`, and check the accidental-match rate: with N spots/frame and an angular
tolerance `t`, roughly `N²·t/180°` pairs match by chance per frame. At N ≈ 35 and
t = 0.3° that is ~2 accidental pairs per frame, so a peak built from a handful of pairs is
not a measurement.

> **A gate failure may mean STARVED, not BROKEN — check the inlier count first.** On
> `nfdev_jul26` at 40 ω samples the `0→2` pair failed the y-vs-z gate at **1534 µm** (limit
> 200) on 6 inlier pairs, and §6i:1005 says to drop such a pair. At 240 ω samples the same
> pair passes at **57 µm** on 18 pairs. Sampling more ω is far cheaper than a wrong `Lsd`.
> §6i's instruction to drop a gate-failing pair applies to a pair that fails **with enough
> statistics**, not to one that has too few matches to be measured at all.

Worked reference — `nfdev_jul26` layer 1, 240 ω samples, BC known per distance:

| pair | k | n_peak | L(dist 0) | y-vs-z |
|---|---|---|---|---|
| 0→1 | 1.3258 | 71 | 6139.0 | 10.6 |
| 1→2 | 1.2457 | 47 | 6140.2 | 15.2 |
| 0→2 | 1.6513 | 18 | 6141.3 | 57.1 |

Both nulls die at 8.88×; leave-one-out std 0.4 µm; the three pairs agree to 2.3 µm ⇒
`Lsd = 6139.7 / 8139.7 / 10139.7 µm`, δ = −860 µm. **δ is a motor zero offset and may be
negative** — it is not a physical distance and its sign carries no meaning.

> **TERMINOLOGY TRAP — "delta".** It is used for two different things and confusing them
> is a millimetre-scale error. `δ` in this handbook is the **Lsd offset**, `δ = L₁ − DetZ₁`
> (a motor zero offset; it can be negative). The **step between detector positions** is a
> different number entirely (`ΔD`, e.g. 2000 µm for `nfz` 7/9/11 mm). Only `ΔD` is
> trustworthy from the motor (hard rule 10); `δ` must be *measured* by triangulation. When
> anyone hands you "delta = N", establish which one they mean before writing an `Lsd` line.

##### 6i-ter. The point-source assumption — quantify it before quoting a precision

`p(L) = BC(L) + L·d` treats every ray as leaving **one point at BC**. Real rays leave
grains spread across the illuminated width, so each spot carries a position term the model
does not know about. The fractional perturbation of `|p − BC|` is

```
perturbation  ≈  (illuminated half-width, µm) / (typical spot radius, µm)
```

| sample | half-width | typical radius | perturbation | y-vs-z split |
|---|---|---|---|---|
| `nfdev_jul26` Au cube | 35 µm | 1096 µm | 3 % | **57 µm** |
| `bt_20id_jul26b` `nf_sampleD` | 124 µm | 1096 µm | **11 %** | **142 µm** |

The **y-vs-z split is the symptom** — it rose by the same factor. The mode of the `k`
histogram is more robust than its mean, so the answer does not collapse, but the honest
precision degrades from ~2 µm to ~200 µm. Quote it accordingly, and do **not** read a
200 µm difference between two campaigns as a calibration change when the sample is this
wide.

Two further cautions from `nf_sampleD`:

* **`r_min_px` matters more than the angular tolerance.** At `r_min=300` the module
  returned `k = 1.0146`, `L = 136 mm`, y-vs-z split **129 mm**, and correctly REJECTED it:
  near BC, ray directions are degenerate and the matcher pairs noise. `r_min=800` gave
  `k = 1.2391` on 108 pairs and passed. Sweep `r_min`, and believe the rejection.
* **Apply the radius cut BEFORE any per-frame brightness cap.** `triangulate` applies
  `r_min_px` internally, but if the *spot finder* caps at the N brightest blobs first, the
  bright halo around the direct beam fills the budget and the real large-radius spots never
  reach the matcher. On `NF_Au_cube_0802` there were **524 blobs/frame inside r < 800 px**
  against ~120/frame outside it: a `MAXSPOT=150` cap taken before the cut left the matcher
  nothing but chance pairs, and every gate failed with `k ≈ 1.007` (the pattern "not
  scaling" between distances). **`k ≈ 1` with huge y-vs-z splits means the matcher is seeing
  noise, not that the detector did not move** — check the spot list before doubting the
  motor.
* **δ is not an instrument constant.** `δ = L − DetZ` contains where the *sample* sits
  along the beam, so a remounted sample legitimately changes it. Comparing δ across
  campaigns tests the mounting as much as the detector. What transfers is the *method*,
  not the number.

---

## 7. STEP 5 — Refine the geometry on a calibrant, and know what it cannot do

§6 gives `BC` per distance and a starting `Lsd`. That is *not yet a usable
geometry*: the three detector tilts `tx/ty/tz` are still unknown, and `Lsd` is
only as good as the triangulation. This step refines them against a known
single-crystal calibrant (a gold cube), then hands the result to the real sample.

**Read §7b before running anything here.** The obvious ways to do this
refinement have all been tried on real data and all fail silently — they return
a confident, wrong answer rather than an error.

### 7a. What is measured, what is refined, what is fixed

| Quantity | Where it comes from | Refined here? |
|---|---|---|
| `BC` per distance | DetZBeamPos direct beam + shadow (§6) | only within `BCTol`, tiny |
| `Lsd` per distance | spot triangulation (§6a), or DetZ + δ | yes |
| `tx` | direct-beam stripe slope (§6f) | yes, from that seed |
| `ty`, `tz` | **nothing measures them** — start at 0 | yes |
| ω convention | `NF.par` f9 (§2) | never — fixed input |
| energy / λ | `fastsweep_Emon.txt` f10 (§4a) | never — fixed input |

A single direct-beam stripe carries no first-order signature of `ty` or `tz`, so
they must come out of the calibrant fit. This is exactly why the fit is
under-determined in the way §7b describes.

### 7b. Three verified negatives — do not rediscover these

All three were established on `bt_1id_jul26` / `Au5_cubes_nf_96keV`, 4
distances, 95 keV. They are properties of the *problem*, not bugs.

**(1) Confidence 1.0 is a plateau, not a unique solution.**
Single-voxel refinements seeded at `ty` = 0.559, 1.507 and 2.622 deg all converge
to confidence **exactly 1.0000**. Reaching confidence 1 tells you the geometry is
*self-consistent with the spots you kept*; it does not tell you it is *the*
geometry. Never report "confidence 1.0, therefore calibrated".

**(2) `-multiGridPoints` does NOT break the degeneracy.**
The natural fix — refine against many voxels at once — was run. Seeded from the
`ty`=0.559 plateau it converged to `ty` 0.683 (mean confidence 0.9562); seeded
from `ty`=2.622 it converged to `ty` 2.985 (mean confidence 0.9753). The two
answers are **2.3 deg apart in `ty` and ~48 µm apart in `Lsd`**, and both look
excellent. Cause: on a calibrant cube every voxel belongs to **one grain**, so
twelve voxels contribute one orientation's worth of constraint, not twelve.
Multi-point helps only if the voxels sample *different* grains.

**(3) Never iterate single-point refinement.**
`TiltsTol` is interpreted **relative to the current seed**, not as an absolute
bound. Feeding a refinement's output back in as the next seed therefore ratchets
the tilts outward by roughly the tolerance each pass (~1 deg/iteration observed);
`ty` walked to 4.6 deg while confidence stayed high the whole way. Run the
refinement **once** from a defensible seed. If you want more iterations, use
`NumIterations` inside a single invocation, which does respect the original seed.

**Also observed, unexplained:** `LsdRelativeTol 5` (a value used successfully in
the past) **stalled** at confidence 0.27 on this dataset, while
`LsdRelativeTol 1` succeeded. Do not assume 5 is a safe default here; it is
recorded as an anomaly, not as a rule.

### 7c. The procedure that works

Single voxel, one invocation, tight tolerances, everything you actually measured
held nearly fixed.

```bash
# params.txt for calibration -- differences from a recon paramfile:
#   Rsample small (30), GridSize 2         -- a calibrant cube is tiny
#   MinConfidence 0.7                      -- reject junk seeds
#   BCTol 0.02 0.02                        -- BC is MEASURED (§6); barely let it move
#   LsdTol 500 / LsdRelativeTol 1          -- see the LsdRelativeTol 5 anomaly above
#   TiltsTol 1                             -- NOT the 0.05 code default: ty/tz start
#                                             at 0 but may really be ~0.5 deg, and a
#                                             0.05 deg tanh box makes them unreachable
#   NumIterations 3                        -- iterate INSIDE one call, not by re-seeding
#   GridPoints <a full 12-column .mic row> -- see the format trap below
```

`GridPoints` takes a **raw 12-column `.mic` data row**, not an abbreviated
coordinate triple (`FitOrientationParametersMultiPoint.c:697` `sscanf` reads 12
tokens). Passing 6 tokens parses without error and silently refines nothing
useful. Take the row straight out of a previous `.mic`.

> **There is no documented bootstrap for the first session on new data**, which has no
> previous `.mic` to take a row from. This is an open gap, not an oversight you are
> expected to route around silently. Two candidate routes exist and **neither has been
> verified**: seed from the shadow measurement's `position_candidates_um` (§6e), or from a
> rough sample centroid, in both cases synthesising the remaining columns of the 12-column
> row. If you use either, say on the report that the seed was synthesised and how, because
> a silently-wrong seed refines to a confident answer.

> **CHECK THE SPOT COUNT BEFORE TRUSTING A SINGLE-VOXEL FIT — this recipe is not
> portable.** The hard FracOverlap is `matched / predicted` for that voxel, so with `N`
> predicted spots the objective is a **step function quantised at 1/N**, and a
> derivative-free simplex cannot move 12 parameters (3 tilts + 3 `Lsd` + 6 `BC`) across a
> plateau tread. On `nfdev_jul26` a voxel had only **N = 46**: the refinement reported
> 0.695652 → 0.717391, i.e. **32/46 → 33/46 — exactly one extra spot** — and `tx`, `ty`,
> `tz` never left 0.0000 while iterations 2 and 3 were bit-identical to iteration 1
> (lab notebook §7g).
>
> **Diagnostic:** print the objective to full precision; if the values are ratios of small
> integers you are quantisation-limited, not converged.
>
> `N` is a property of the beamline geometry — how much of each diffraction cone the
> detector covers. A beam near the detector edge (20-ID: 121 px from the bottom) sees
> roughly half the azimuthal spots a centred one does. **With V voxels the resolution
> becomes `1/(N·V)`**, so when `N` is small, go multi-point immediately; on such a geometry
> multi-point is load-bearing, not a polish step. See §7b(2) for why those voxels must come
> from *different grains*.

Then verify the fit reproduces the *known* answer on one voxel before trusting it
anywhere:

```bash
midas-nf-fit-orientation params.txt 0 1 16 --device cuda --fp32
# read Confidence out of the .mic; on Au it should be 1.000000 with BoxSize set
```

### 7d. `BoxSize` — implemented in Python, and it changes the answer

`BoxSize` was parsed but **never applied** in the Python path until this tree.
The gate is:

- units **µm on the detector at `Lsd[0]` only** — applied before displacement,
  tilts and `BC`;
- comparisons are **strict** (`>` low, `<` high);
- paired with `OmegaRange` **by index**; a spot is kept if it satisfies **any**
  (`OmegaRange`, `BoxSize`) pair;
- rejected spots are **excluded from the confidence denominator**, not counted as
  misses.

That last point is why it matters so much: with the gate off, a single Au voxel
scores **0.949153**; with it on, **1.000000**, matching the C reference exactly.
The Triton fused kernel implements the same gate (verified: eager 1.000000 /
triton 1.000000 gate-on, 0.949153 / 0.949153 gate-off).

**If confidence plateaus just below 1 on a calibrant, check `BoxSize` before
touching the geometry.** A missing gate looks exactly like a slightly-wrong
geometry.

### 7e. Discriminating two candidate geometries with a full map — a WEAK test

When §7b leaves two plausible geometries, the instinct is to reconstruct under
each and pick the better map. Do it, but calibrate your expectations: **on
`bt_1id_jul26` this test did not cleanly separate them.**

Both geometries produced a single coherent gold crystal, and:

| | geometry A | geometry B |
|---|---|---|
| mic rows | 5012 | 5044 |
| median confidence (whole grid) | 0.170 | 0.273 |
| voxels `C >= 0.9` | 384 | 477 |
| equiv-area radius at `C>=0.9` | 14.55 µm | 16.22 µm |
| Eul1 (high-C) | 2.5807 rad | 2.5800 rad |
| Eul1 spread | 0.148 deg | 0.186 deg |
| high-C voxels beyond shadow radius, about own centroid | **0.000** | **0.000** |

The two orientations agree to **0.0007 rad (0.04 deg)** — far below either map's
internal spread. B has uniformly higher confidence at every threshold; that is a
shift of the confidence scale, not a structurally different microstructure.

**The metric trap that nearly inverted this conclusion.** Measuring "fraction of
high-confidence voxels outside the known sample radius" **from the grid origin**
gave A 0.02 and B 0.09, which reads as B smearing signal into empty space. It is
an artifact: **the sample is not on the rotation axis.** Both blobs are offset
(A centroid −4.62 µm, B −6.89 µm), and once radius is measured about **each
map's own high-confidence centroid** the difference vanishes entirely — both are
0.000. Always centroid-correct before comparing blob sizes.

Useful external anchor: the DetZBeamPos shadow gives a sample radius (20.9 µm
here) that is **not** derived from the fits being compared, so it is the only
independent size check available. But note it is threshold-dependent — at
`C>=0.90` both blobs read ~15-16 µm, at `C>=0.70` both read ~23 µm — so it
bounds gross errors rather than resolving fine ones.

**Bottom line: budget for the map being a weak discriminator.** If two geometries
survive §7b, expect to choose on aggregate confidence and operator judgement, and
label the choice provisional.

### 7f. Reference — `bt_1id_jul26` Au5, the geometry actually adopted

95.0000 keV, λ 0.1305097 Å, px 1.48 µm, 2048², DetZ 7/9/11/13, stage `aero`.
Adopted **geometry A**:

```
Lsd 7228.584913     BC 996.716776  37.941506
Lsd 9229.709611     BC 1013.675328 41.313334
Lsd 11229.713336    BC 1029.377525 44.328838
Lsd 13228.960327    BC 1043.678358 47.979195
tx 0.788229   ty 0.683384   tz 0.082687
```

Geometry B (`Lsd` 7276.153002/9278.833913/11280.854593/13281.673523, `tx` 0.705447
`ty` 2.985390 `tz` 0.446467) was the competing plateau and was judged slightly
worse. **The margin was small — treat A as the working geometry, not as a
refutation of B.**

Because the calibrant and the real samples sit on the same rotation axis at the
same DetZ positions, these `Lsd`/`BC` values transfer directly to any scan in the
beamtime that uses the same DetZ set. That transfer is the entire reason to
reconstruct the gold first.

---

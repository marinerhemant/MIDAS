# Tomography lab notebook — sample shape for FF/PF/NF grain size

**Companion to `README.md`.** The handbook says what to do; this records what was found,
how it was measured, and what turned out to be wrong. They are kept apart on purpose: a
handbook has to stay short enough to follow, and a campaign record has to stay honest
enough to stop a refuted idea coming back.

**One notebook per campaign, not per dataset, started on day one.** The retractions decay
fastest and backfilling them does not work.

`§n` without a qualifier means a section of *this* file; handbook sections are written
`Handbook §n`.

Campaign opened 2026-08-23, out of the "physics-aware completeness" thread
(`packages/midas_index/dev/CHECKPOINT_completeness_physics.md`).

---

## 1. What this campaign established

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | Absolute grain size in MIDAS is a canned constant: `V_gauge` is built from two search bounds, and 6112 grains sum to 6.5 % of it | VERIFIED | §4.1 |
| 2 | Correcting the numerator without the powder denominator inflates every volume by `⟨1/A⟩` | VERIFIED (arithmetic) | §3.1 |
| 3 | Three one-pixel defects in the NF tomo grid filter; 75 % of grid points landed on a different pixel than the C it claims parity with | FIXED | §2.1 |
| 4 | μD decides whether an absorption correction exists to be measured: 0.05 on NMC811, 1.63 on Ce | VERIFIED | §4.2 |
| 5 | The V1 sinogram check has **zero power** on a cylinder, which is the commonest FF specimen shape | VERIFIED | §3.2 |
| 6 | Whether the corrected sizes are *right* — no dataset has been through the whole chain yet | OPEN | §5.1 |
| 7 | `tomocupy_args.yml` records the wrong pixel size for both beamtimes; the per-scan `tomo_metastr` is authoritative | VERIFIED | §4.3 |
| 8 | The bt_1id_jun25b NMC811 tomogram **cannot produce a sample mask** — no threshold plateau, mask fills the field of view | VERIFIED | §4.4 |
| 9 | In-plane handedness IS recorded in the acquisition metadata; I wrote that it was not | CORRECTED | §4.3 |
| 10 | The scan record is self-describing: derived frame indices reproduce the hand-counted ones, and the regenerated `.raw` is byte-identical | VERIFIED | §3.4 |
| 11 | Automatic centring reproduces the human's +13.00 exactly — but the two-criterion consensus declines to certify it | PARTIAL | §3.5 |
| 12 | Evenly spaced centring probes land on empty rows and return sweep-edge picks | FIXED | §3.5 |
| 13 | `sharpness(method='tv')` is biased low on a high-contrast reconstruction | OPEN | §3.5 |
| 14 | Detector roll is measurable three ways against two references; `RotationAngle` had a consumer and no producer | VERIFIED (synthetic) | §3.6 |
| 15 | The 1-ID **vertical** beam boundaries are not straight slit edges (rms 15 px vs 0.3-0.5) | VERIFIED | §3.6 |
| 16 | `compare_tilt_estimates` recommended a value that had flagged itself invalid | FIXED | §2.2 |
| 17 | Ce detector roll is **-0.006 +/- 0.030 deg**, consistent with zero, from the rotation-axis route | VERIFIED | §3.6 |
| 19 | **RETRACTED:** the "4.64x in reported radius" for bt_1id_jun25b was an artefact of percentile-derived thresholds, not a measurement | RETRACTED | §3.7 |
| 20 | Paganin implemented and gated on synthetics; it did **not** rescue the bt_1id_jun25b mask | VERIFIED | §3.7 |
| 21 | One command, `midas-tomo-reconstruct`, replaces the per-sample prepare+driver script pair | DONE | §3.8 |
| 22 | **Phase 3 validation gate CLOSED** — the whole chain ran on Ce ht525 s2, the specimen that also has an FF reconstruction | VERIFIED | §4.5 |
| 23 | `Vsample 1e7` on the Ce FF corresponds to a 12.95 um beam height — a real setting, not a canned constant | VERIFIED | §4.5 |
| 24 | **Variance inverts on a smooth strong absorber**; TV inverts on a high-contrast one. Neither criterion is universal | VERIFIED | §3.9 |
| 25 | **`Vsample` now has a producer** — measured from a tomogram, written into the FF parameter file with its uncertainty | DONE | §3.10 |
| 26 | **The beam height comes from the beamline log, not the slits** — the Ce FF ran a vertically focused NF beam, 1 um knife-edge; the slits would have been 100x wrong | VERIFIED | §3.11 |
| 27 | **Ce gamma grain radii were overstated 2.34x**: Vsample was 1e7 where the measurement gives 7.79e5 | VERIFIED | §4.6 |
| 28 | **RETRACTED:** "the Ce gauge volume looks like a real setting" — it was ~13x too large | RETRACTED | §4.6 |
| 29 | Both Ce phases corrected: gamma 4.076 -> 1.741 um, dhcp 6.678 -> 2.852 um | VERIFIED | §4.7 |
| 30 | **NEGATIVE: the structure-factor drop is a NO-OP on this dhcp run** — the 7 rings used contain zero extinct reflections | VERIFIED | §4.7 |
| 18 | Route 2 needed four real-data fixes; rms 40.3 -> 0.56 px. Three were one root cause: unilluminated detector reads as huge fake attenuation | FIXED | §2.3 |

## 2. Defects fixed

### 2.1 Three one-pixel defects in `sample_tomo` (2026-08-23)

`midas_nf_preprocess/tomo_filter/filter.py` transcribes
`NF_HEDM/src/filterGridfromTomo.c:39-43`. All three were found by reading the C against the
Python after a crash report; none had a test, because **every existing test used integer
coordinates, which is the one case where the buggy and correct conventions agree**.

1. **Row flip off by one, reading out of bounds.** `row = n - y_pos` maps `y_pos ∈ [0,n-1]`
   onto rows `[1,n]`: row 0 is never read, and `y_pos == 0` indexes `imTomo[n*n + xPos]`,
   past the end of the `calloc`. Out-of-bounds heap read in C; reproduced as
   `IndexError: index 8 is out of bounds for dimension 0 with size 8` in Python.
2. **Truncation in the wrong place.** The Python computed `int(x/px) + n//2`, truncating the
   *quotient*; the C truncates the *sum*. Truncation is toward zero, so the Python's centre
   pixel is **twice as wide as every other pixel** — both `(-1,0]` and `[0,1)` collapse onto
   it. Measured on 200 000 points over a 1 mm sample at 1.5 µm: **75.0 % of grid points
   landed on a different pixel than the C** (50.2 % in x, 50.0 % in y, max |Δ| = 1 px).
3. **Integer `n//2` where the C uses `(double)n/2`.** Identical for even `n`; for odd `n` it
   puts the origin on the *edge* of the centre pixel instead of its centre.

Fixed behind `legacy_c_parity`, defaulting to `True` so existing reconstructions reproduce.
Both modes now drop `y_pos == 0` rather than reading out of bounds — undefined behaviour
cannot be reproduced, and that is the one parity-mode change that is not a bug fix.
Pinned by 8 tests in `tests/tomo_filter/test_filter.py`.

### 2.3 Route 2 took four fixes, three of them one root cause

Residual scatter across the four rounds: **40.3 -> 41.5 -> 47.9 -> 12.2 ->
0.56 px** (limit 2.0). Only the last is trustworthy. In order:

1. **Row selection** — the fit used all 2031 rows including the unilluminated
   ones. Restricted to rows with signal.
2. **"Signal" was defined from the attenuation.** Outside the beam
   `white - dark` is ~0, so the ratio is noise and the clip floor gives
   `-log(1e-6) = 13.8`: an unlit row scores as the strongest absorber on the
   detector. All 2048 rows passed. **The illuminated region must come from the
   flat field.**
3. **Columns were never masked at all.** Same mechanism in the other
   direction: fake attenuation at both ends of every row dominated the centre
   of mass. This was the big one, 47.9 -> 12.2 px.
4. **Truncated rows.** A centre of mass locates the axis only if the whole
   specimen is in view. Rows 300-500 had an air fraction of 0.005-0.024
   against 0.04-0.06 in the bulk and gave axis positions of 1158/1193/1154,
   while rows 600-1500 sat at 1146.6-1148.2. A `min_air_fraction` guard drops
   173 such rows.

Items 2, 3 and 4 are all "the detector is not uniformly illuminated and the
un-illuminated part does not read as zero". That is the same trap as the
"furnace with two windows" misreading earlier in this thread, met three more
times in one function.

### 2.2 `compare_tilt_estimates` recommended a number it knew was invalid

Found by running it on real data, 2026-08-23. On the Ce scan the
rotation-axis route returned **-2.5172 deg with 40.28 px of residual scatter
against a 2 px limit** and correctly set `trustworthy=False`. The comparison
function then printed:

    recommended : -2.5172 deg

because its disagreement branch chose the rotation-axis value without ever
looking at `trustworthy`. The existing test passed two *valid* inputs, so it
could not have caught this.

Fixed by checking validity first: two invalid inputs give `NO MEASUREMENT`,
one invalid gives `UNCERTAIN` and recommends only the valid one, and the
informative reading of a disagreement — "the slits are not square to the
rotation axis" — is now reachable only when both numbers are real
measurements. Regression test
`test_a_disagreement_with_an_INVALID_input_is_not_a_recommendation` uses the
actual Ce numbers.

The underlying cause of the -2.52 deg was that route 2 fitted **all 2031
rows**, including the ~700 unilluminated ones whose centre of mass is noise
about the frame centre. It now restricts to rows with signal.

## 3. Method findings

### 3.6 Detector roll, three routes against two references (2026-08-23)

`midas_tomo.hdf5` **consumes** a detector roll — `RotationAngle` is read at
`hdf5.py:154` and applied by `_rotate_stack` — and nothing in `packages/` ever
measured it, so the correction is dead code in every scan. New
`midas_tomo.detector_tilt` supplies it. The angle returned is the
**correcting** one: pass it straight to `RotationAngle`.

| route | reference | data |
|---|---|---|
| `tilt_from_beam_box` | the **slits** | flat fields, no sample |
| `tilt_from_slice_shifts` | the **rotation axis** | a multi-shift reconstruction |
| `tilt_from_rotation_axis` | the **rotation axis** | the projections |

The split that matters is the reference, not the count: routes 2 and 3 are two
estimators of the same quantity. Prefer route 3 — the centre of mass needs a
clean attenuation baseline and is dragged by background and truncation, while
the best-shift criterion is a sharpness optimum and is what the reconstruction
actually depends on. Route 3's limit is quantisation: sweep step over row span
sets the finest visible angle, and below it the result is reported as an
**upper bound**, since a slope fitted through quantised picks always returns
something.

Synthetic gate: planted rolls of 0.05, 0.3, 1.0 and 2.0 deg recovered to
<0.02 deg, and — the definition that matters — rotating by the returned angle
squares the box, leaving <0.02 deg residual.

**Measured on bt_1id_jul26 Ce `ht525_s2` flats:**

| edge | angle | rms | span |
|---|---|---|---|
| top | +0.0361 deg | 0.53 px | 689 px |
| bottom | +0.0001 deg | 0.31 px | 689 px |
| left, right | **discarded** | 15.9, 15.1 px (limit 3.0) | — |

Two findings. **The vertical beam boundaries at 1-ID are not straight slit
edges** — 30x the residual of the horizontal pair — so only one edge family is
fittable and the orthogonality null cannot run. The result is therefore
`trustworthy=False` despite a clean-looking number. And the roll itself is
~0.02 deg, i.e. 0.8 px across a 2320-px frame: consistent with zero. The front
and back whites agree independently (+0.0181 vs +0.0215 deg).

*Provisional:* top and bottom differ by 0.036 deg against a formal slope
uncertainty of ~0.007 deg — about 5 sigma, so the illuminated region looks
like a slight wedge rather than a rectangle. Residuals along an edge are
probably correlated, which would lower the true significance, so this is not
promoted. The 0.05 deg parallelism tolerance let it through and may be too
loose for a 689-px lever arm.

**Route 2 on Ce, after four fixes (§2.3):** **-0.0059 +/- 0.0300 deg**,
`trustworthy=True`, 83 rows over a 1072 px span with 0.56 px residual. The
detector roll is consistent with zero. Route 1 independently gives +0.0181 deg,
agreeing to 0.024 deg — but the cross-check correctly returns **UNCERTAIN**
rather than AGREE, because route 1 could not self-certify (one edge family).
Numerical agreement with an input that flagged itself invalid is not evidence.

*A caution against arithmetic shortcuts:* a two-point estimate from the
diagnostic table (rows 600 and 1500) gave -0.053 deg. The real fit over all 83
clean rows gives -0.0059 deg. The two-point version was wrong by nine times the
final uncertainty.

**bt_1id_jun25b, via route 3:** the T3 fine sweep's per-slice picks were +13.00 on
rows 65, 106 and 127, identical at the 0.1 px sweep resolution over a 62-row
span. That is an **upper bound of 0.09 deg**, not a detection. (Row 40 gave
+15.00 and is the same row that failed in the coarse sweep; the edge-pick
filter drops it.)

### 3.1 The powder reference is the trap, not the correction

`powder_int` (`radius/core.py:153-160`) is a sum of **observed** spot intensities, so it
already carries every effect the numerator does. Writing the corrected estimator over the
legacy one, everything cancels except

    V_cor(s)/V_leg(s) = (V_illum / V_gauge) · C_cov(r) · f(s)/⟨f⟩_r

Only the **spread** of a per-spot correction survives. Dividing `I_spot` by `f` while
leaving the denominator raw multiplies every volume by `⟨1/f⟩`: measured at 1.6× in volume
and 1.17× in radius for μD uniform on [0.2, 0.8]. Uniform across the dataset, in the
direction everyone expects, with no symptom.

`normalise_per_ring` enforces `⟨f⟩_r = 1`. It short-circuits a constant ring to **exactly**
1.0 rather than trusting `f/mean(f)` to round there, because the invariant asserted
downstream is bit-identity, not approximate identity.

### 3.2 Registration checks that cannot fail

A cylinder on the rotation axis has a lit volume **independent of ω**. So the V1 sinogram
check compares a flat prediction against any measured curve, "agrees" to within noise, and
passes having tested nothing. `sinogram_check` measures the predicted modulation first and
returns `NO_POWER` (not `PASS`, and `bool()` is False) below 2 %.

The same problem in subtler form for V2: on a near-symmetric sample a **mirrored** mask
contains the grain centroids just as well as the correct one. `meta_null` reruns any check
on `shape.mirrored()` and demands the statistic degrade. Verified both directions: it
returns `NO_POWER` on a centred box and `PASS` on an L-shaped cross-section, so it is not a
function that always says one thing.

**V2 is blind to a pure translation by construction** — the fit starts from the difference
of centroids. It tests the *shape* of the registration, never its origin.

### 3.3 The reconstruction pad

`recon_xdim = next_power_of_2(det_xdim)` (`midas_tomo/config.py:198`): a 1365-wide detector
gives a 2048 grid, so a third of every slice is padding no ray sampled. Pass `det_xdim`;
without it the check falls back to the grid's own inscribed circle, which is weaker, and the
provenance says which one ran.

**Revised 2026-08-23 after it fired on real data.** The first version refused a mask with
*any* occupancy outside the disc. That is too strict to be usable: the corners of a square
reconstruction grid are outside every projection's field of view by construction, so a few
rung-up corner voxels are expected. It refused all 12 thresholds on bt_1id_jun25b NMC811 s5,
including ones where only 0.5 % of the mask was outside. The readers now **clip** the mask
to the disc — that region is not data, so zeroing it is the correct treatment — and raise
only when the overflow exceeds `max_pad_fraction` (1 % by default), which means truncation
or a wrong rotation axis rather than ringing.

### 3.4 The scan record replaces the hand-written prepare scripts (2026-08-23)

`<prefix>_TomoFastScan.dat` states `WF#`, `DF#`, `Proj#`, the white-field start
number, and independently the first/last image numbers and the total count.
`midas_tomo.scanrecord` derives the frame layout from the first set and
**cross-checks it against the second**, refusing when they disagree — an
off-by-one block boundary averages projections into the flat field, silently.

Gate, on bt_1id_jun25b `nmc811s5tomo1`: the derived block starts are
`{front_white: 7317, projections: 7327, back_white: 10928, dark: 10938}`,
matching all four indices `prepare_data_nmc811_s5_tomo1.py` hard-codes. Then
`midas_tomo.ingest` regenerated `data_nmc811s5tomo1.raw` from the source TIFFs:
**118 194 176 bytes, sha256 `d933c7167a271406`, byte-identical** to the
beamline's file — calibration header and projection block both. Means are
accumulated sequentially in float32 for exactly this reason (`np.mean` is
pairwise and differs in the last bits).

### 3.5 Automatic centring: right answer, withheld verdict (2026-08-23)

The beamline picked **+13.00** for `nmc811s5tomo1` by eye from a 501-panel
contact sheet. Coarse-then-fine automatic centring, scored on four slices:

| criterion | coarse picks | fine median |
|---|---|---|
| variance | −25.00, **+13.00**, **+13.00**, **+13.00** | **+13.000** |
| total variation | −6.00, +6.00, +11.00, +11.00 | +11.400 |

**Variance reproduces the human's answer exactly, 0.000 px.** The consensus
still returns `trustworthy=False`, because TV sits 1.6 px low — over the 0.2 px
fine tolerance. That is the designed behaviour and the flag is not being
relaxed to make a gate go green.

Two things came out of getting there.

**Evenly spaced probes are the wrong default.** The first run scored slices
16/48/80/112 and two of them returned −25.00 and −23.00, the bottom of the
sweep: those rows are empty, their sharpness curves have no interior optimum,
and `argmax` returns an end. Fixed two ways — `slices_with_signal` chooses rows
by attenuation contrast, and a pick landing on the first or last candidate is
now discarded as not-an-optimum. With signal-selected rows (40/65/106/127),
three of four found +13.00.

`slices_with_signal` had to rank by **within-row contrast (p99 − median), not
mean attenuation**: on this specimen the strongest row mean was **−0.0005**,
slightly negative, because front-to-back flat-field drift is larger than a
29 µm specimen's contribution to a 128-pixel row average.

**OPEN — `sharpness(method='tv')` looks biased for centring.** It maximises
`−mean|∇f|`, i.e. prefers the image with the *least* gradient. Mis-centring is
supposed to add variation (doubled edges, streaks), but on a reconstruction
whose specimen has strong edges, the well-centred slice has the most gradient
and TV prefers the degraded one. Seen three times: on a blurred-disc fixture TV
picked the blurriest every time; on a split-copy fixture it picked the most
split; and here it lands 1.6 px low. A criterion that detects artefacts
specifically — residual energy outside the sample support, or negative
undershoot — would be independent of specimen contrast in a way this is not.
Not changed yet: it is pre-existing behaviour and the fix is a design choice.

### 3.7 Paganin, and a RETRACTION about the threshold diagnostic

`midas_tomo.phase_retrieval` implements the single-material Paganin filter.
Gated on synthetics: an exact round trip (synthesise propagation from a known
disc, filter it back, residual <1e-6) and a **bit-identical null** at
`delta_beta = 0`, so enabling the filter with no strength provably changes
nothing.

**On bt_1id_jun25b it did not rescue the mask.** Swept over
`delta/beta = 0, 50, 200, 800, 3000, 10000`, no configuration produced a
stationary threshold, and the mask extent stayed at 73-80 um against a ~29 um
specimen in a 90.6 um field of view. Reported rather than tuned away, as the
gate said it would be.

**RETRACTED: the "4.64x in reported radius" figure for bt_1id_jun25b.** Running the
sweep across six reconstructions exposed that `radius_spread` came back as
*exactly* 4.642 every time. It is `100**(1/3)`, and it is pinned by the
threshold choice, not the data: with thresholds at `linspace(p50, p99.5)` of
each reconstruction the volume always runs from ~50 % of the voxels to ~0.5 %,
a ratio of 100. **The number was an artefact of my own diagnostic.**
`fractional_spread` does still carry information (it varied 9.2 / 33.9 / 9.5 /
7.4 / 7.0 / 6.9) and the "not stationary" verdict stands.

The obvious alternative is also wrong: a *fixed absolute* threshold range
penalises anything that changes the value scale, and Paganin lowers peak
attenuation as it smooths — on a fixed range its volumes collapsed from 72.9 um
of extent to 5.7 um for reasons unrelated to mask quality. **A scale-invariant
stationarity measure is still owed**; what the check really asks is whether the
histogram is bimodal with a plateau between the modes. Pinned meanwhile by
`test_percentile_thresholds_pin_radius_spread_to_a_CONSTANT`.

### 3.8 One command

`midas-tomo-reconstruct <scan_record> --root <dir> --out <dir>` replaces the
per-sample `prepare_data_*.py` + driver pair: read the record, ingest, optional
Paganin, coarse-then-fine automatic centring, reconstruct, optional detector
roll, NXtomoproc with provenance, and the `SampleShape` call printed at the
end. It **stops rather than reconstructing on an uncertified shift**, since the
alternative is a plausible mis-registered volume; `--no-strict` overrides and
marks the geometry unverified. The hint it prints deliberately leaves
`in_plane`, `threshold` and `slice0_z_um` marked unresolved rather than
guessing them.

### 3.9 Both sharpness criteria invert, in opposite regimes

Measured on two real datasets, and it explains why the beamline centres by eye.

| specimen | character | variance | total variation |
|---|---|---|---|
| bt_1id_jun25b NMC811 | mu*D 0.05, phase contrast, feature/edge dominated | **correct** (+13.00, matches the human exactly) | inverted, 1.6 px low |
| Ce ht525 s2 | mu*D 1.74, strong, near-uniform disc | **inverted** | flat (1.6 % total range) |

On Ce, `find_center` maximises variance and therefore ran to the sweep edge and
picked the *worst* reconstruction. Widening to +/-120 px shows why: variance has
a clean **minimum** at -4.0 px, rising monotonically both ways (8.55e-8 to
1.26e-7). A well-centred reconstruction of a smooth strong absorber is *smooth*;
mis-centring adds streak variance. On a weak, feature-dominated specimen the
opposite holds — centring concentrates what little signal there is.

The -4.0 px minimum independently matches the rotation-axis measurement
(axis at 1147.5 px, crop start 256, crop centre 896, so -4.5 px).

The consensus refused in **both** cases rather than reporting the wrong sign,
which is the behaviour it exists for. But a criterion that is specimen-dependent
in its *sign* cannot be automated as it stands. What is universal is that
mis-centring adds artefacts; an artefact-specific measure (energy in the air
annulus, or negative undershoot) is the fix. Tested here on Ce: air-annulus
variance gives a smooth interior optimum near 0, negative-undershoot is
monotonic and unusable. **Not yet implemented** — it needs testing in both
regimes, not one.

*Also measured:* on this specimen the optimum is a broad, shallow parabola, so
the shift is only determined to about +/-5-10 px. That turns out not to matter
here (below), but it is a property of a featureless specimen, not of the method.

### 3.11 The beam height is in the log, and the slits are not it

`Vsample = cross-section x beam height`, and the tomogram cannot supply the
height. The obvious source — the slit settings — is **wrong whenever the beam
is focused**, which is exactly the FF case here. From `FullLog.log` at the
moment of the Ce scan:

    131.FOURC1IDE> switch_to_NFFocusedBeam
    switch_to_NFFocusedBeam HxV: 1200 slitted x 1 focused beam
    Setting E US slit size to:  1.2 x 0.3 mm ...
    Setting E DS slit size to:  1.3 x 0.1 mm ...

The vertical slits are 0.1-0.3 mm and are **guards**; the beam is focused to
**1 um, measured with a knife edge**. Taking the smallest slit would have given
100 um — **100x too large**, and a 100x error in every grain volume.

Two further traps, both real here:

* **The scan macro does not say.** `ff_Ce_ht525_s2_line_center` carries
  `#switch_to_HxV_beam 1.2 _ystp` — *commented out* — so the scan inherited
  whatever was already set. Only the chronological log records it. (The
  commented line would have given 2 um, from `_ystp`; also wrong.)
* **A focus scan is not a substitute.** I first read column 6 of
  `FocusScan6_FocusScan.log` as a beam width and got 3.85 um. What that column
  holds is **not established**; the knife-edge value in the log is the
  measurement. `focus_scan_minimum` now reports the column without claiming it
  is a width.

`midas_transforms.radius.beamlog.beam_config_for_scan` reads this
automatically. **Cross-check that it works:** run on the *tomo* scan it returns
`TomoBeam`, slitted, 900 um tall — against the 940 um illuminated box measured
independently from the flat field.

### 3.10 Vsample from a tomogram — the integration that never existed

`Vsample` is the gauge volume that divides into every reported grain volume,
and **nothing has ever produced it**. Runs either omit it (falling back to
`Hbeam * pi * Rsample^2`, two search bounds) or inherit a template constant
(`midas_calibrate_v2` writes 50000000). `midas_transforms.radius.vsample`
measures it from a tomogram of the same specimen and patches the FF parameter
file, keeping the superseded line as a comment and backing up the original.

**Why this integration is safe where the absorption path is not.** Vsample
needs the specimen's *volume*, not its pose, and volume survives almost every
registration error: a mirrored mask has identical volume, an 8 px centring
error moved the Ce cross-section by **0.25 %**, and the vertical registration
matters only if the cross-section varies with height — which is checked
(Ce: 0.54 %). What it will not guess is the **beam height**, a slit setting
that `Hbeam` is not.

**Run on Ce ht525 s2** (stripe removal snr 3.0 / la 31 / sm 11), with the beam
height read from the log (§3.11) rather than supplied:

    cross-section    778 634 um^2   (equivalent diameter 995.7 um)
    height CV        0.0054
    beam height      1.0 um  [FullLog.log knife-edge, NFFocusedBeam, command 131]
    Vsample          7.786e5 um^3
    uncertainty      +/-8 % volume, +/-2.7 % radius
    replaces         1e7  ->  grain volumes x0.0779, radii x0.427

Nothing here was supplied by hand: the cross-section is measured, the beam
height is read from the beamline log, and the value is written with both.

**A gate that was recalibrated, deliberately.** The first version refused any
volume that was not threshold-stationary, and it refused Ce (spread 0.157-0.22).
That is the wrong trade: it withholds a number good to +/-8 % in favour of a
template constant that can be wrong by orders of magnitude. The spread is now
*recorded with the value* and only a spread above `max_spread` (25 %) is
refused. The Ce softness is real and diagnosed — the specimen's edges sit in
the beam penumbra (20-50 % flux), so the boundary reconstructs gradually — and
artefact removal improves it from 0.211 to 0.157 but cannot remove it.

**What stays refused:** no threshold report at all (silence is not evidence);
a height-varying cross-section without the vertical registration; and an
omega-varying illuminated volume, because `Vsample` is a scalar and no single
value is correct when a narrow beam overlaps a non-cylindrical specimen
differently as it turns.

## 4. Scientific findings

### 4.3 The pixel size is not where it looks like it is (2026-08-23)

`tomocupy_args.yml` says `pixelSize: 1.17` µm for **both** beamtimes. Both are
wrong for the scans in question. The authoritative record is the per-scan
`tomo_metastr`, written into `<expt>/metadata/<expt>/<scan>/<scan>_TomoFastScan.dat`:

| scan | metastr | pixel size |
|---|---|---|
| bt_1id_jun25b `nmc811s5tomo1` | `D~100.000000mm, FLIR-GH1, 5X, 0.708 um/px, aero axis, left handed` | **0.708 µm** |
| bt_1id_jul26 `tomo_Ce_ht525_s2` | `D~100.000000mm, FLIR-GH1, 5X, 0.69 um/px, aero axis, left handed` | **0.69 µm** |

1.17 µm is the PointGrey value. `~/new_data/<expt>/ad_settings.csv` has a
`PIXEL_SIZE_UM` column listing pg1/pg5 1.17, gh1 0.69, gh2/pg6 2.95, ge1-5 200,
varex 150 — so 1.17 is a real number for a real camera, just not this one.
`sample_bt_1id_jun25b.mac` carries three commented-out `tomo_metastr`
lines (PointGrey 5X 1.172, PointGrey 7.5X 0.7813) above the active FLIR-GH1
one, which is presumably how the stale value propagated.

**A 1.65× pixel-size error is a 4.5× error in every volume**, so this is
load-bearing, and it is the same class of trap as the stale `exp_setup.yml
EDGE:` field.

**It also settles the Ce inconsistency.** At 1.17 µm, 1365 px of sample = 1597
µm against the 861 µm implied by the measured μ·D 1.63 — a 1.9× contradiction.
At 0.69 µm it is 942 µm against 861 µm, **1.09×**, which heat-treatment
porosity or a capillary wall covers easily. The discrepancy was the pixel size,
not the density.

**Correction to §3.2 and to `ENVELOPE.md` §1 as first written:** I recorded
that in-plane handedness "is not given by metadata, unlike the vertical". That
is wrong — every scan writes `aero axis, left handed`. What remains true is
that the *reconstruction* carries no handedness, and that mapping the string
onto one of the eight `TOMO_IN_PLANE` signed permutations is still unresolved:
the metastr names a convention, not an axis assignment. Verify with the N3
meta-null regardless.

### 4.4 The bt_1id_jun25b tomogram fails the mask gate (2026-08-23)

Run on `s5_tomo1_cleaned/nmc811s5tomo1_CLEANED_BEST_SHIFT_+13.00.tif`, 128³
float32 at 0.708 µm, via `phase3_bt_1id_jun25b.py`:

| diagnostic | result |
|---|---|
| threshold stationarity | **FAIL** — 372 134 → 3 721 µm³ over a p50–p99.5 sweep, fractional spread 9.205 (**the 4.64× in radius is RETRACTED** — see §3.7) |
| mask extent at the coarsest accepted threshold | 81.4 × 90.6 µm inside a 90.6 µm FOV, 128 of 128 slices occupied, 759 150 of 2 097 152 voxels (36 %) |
| projected sample width from the projections | ~29 µm at 0.708 µm/px — the mask is **3× too wide** |

The mask is thresholded background and edge enhancement. This is the ENVELOPE's
phase-contrast prediction confirmed on real data: at μ·D 0.05 with a 100 mm
propagation distance there is no absorption contrast to threshold.

**So bt_1id_jun25b cannot be the Phase 3 validation dataset either** — it fails at
the tomography end, independently of the fact that its paired scan is PF and
therefore does not use the intensity/gauge-volume estimator at all.

*What the machinery did, which is the reportable part:* every refusal fired for
the right reason. The threshold sweep returned `stationary: False` with the
band; the pad check clipped corner ringing and refused the thresholds where the
overflow was structural; nothing silently produced a number. Reading V1's
`PASS` here would be a mistake — it was fed its own prediction as the
measurement, so it is circular. The only non-circular V1 readings are that the
predicted modulation was 0.0402 (above the 0.02 power floor) and that the
meta-null degraded by 0.615, i.e. V1 *would* have power on this shape — but the
shape is the field of view, not the specimen, so even that says nothing useful.

### 4.1 The gauge volume, measured

`ff_refiner_prepost/result/LayerNr_1` (6112 grains) has **no `Vsample` line**, so
`V_gauge = Hbeam·π·Rsample² = 2000·π·2000² = 2.513e10 µm³` — built entirely from two search
bounds. `Σ(4/3 πR³) = 1.642e9 µm³` = **6.5 %** of it. Median `GrainRadius` 31.4 µm
(p25 26.1, p75 40.4).

The 6.5 % is the re-derivable number. Any "radii are N× too large" figure stays
**illustrative** until a measured shape supplies the denominator — that is the open item in
§5.1, not a result.

### 4.2 μD decides whether Phase 4 exists

| dataset | energy | μ (cm⁻¹) | measured μD | verdict |
|---|---|---|---|---|
| bt_1id_jun25b NMC811 s5 | 51.9 keV | 6.53 | **0.05** | null, at the ±2.5 % flat-field noise floor |
| bt_1id_jul26 Ce `ht525_s2` | 95 keV | 18.94 | **1.63** ± 0.02 over 8 angles | testable; transmission 18 % |

NMC811 was imaged by propagation phase contrast (`propagationDistance: 50 mm`) precisely
*because* its absorption contrast is a few percent. That is the tell.

### 4.5 Phase 3 closed on real data: Ce ht525 s2 (2026-08-23)

`tomo_Ce_ht525_s2` and `ff_Ce_ht525_s2_line_center_000027` are the **same
specimen**, so this is the FF-grains-plus-tomogram pair the whole phase needed.
The FF reconstruction already existed: 2496 gamma-fcc (a 5.1645, SG 225) and 23
dhcp grains.

Reconstructed 200 slices, cols 256-2048, at 0.69 um/px from the scan record:

| quantity | value |
|---|---|
| cross-section | **772 241 um^2** (equivalent diameter **991.6 um**) |
| uniformity along height | std/mean **0.0063** — a straight rod |
| shift sensitivity | **0.25 %** in cross-section for an 8 px shift error |
| threshold stationarity | spread 0.22-0.31, **still not stationary** |
| independent check | peak mu*D 1.74 implies ~919 um of bulk Ce vs 991.6 um reconstructed, ratio 1.08 |

**The shift sensitivity is the reason the ill-determined centring (§3.9) does
not matter here** — measured, not assumed.

**The headline number.** The Ce FF `Parameters.txt` sets `Vsample 10000000`, so
V_gauge = 1e7 um^3 and it is *not* on the search-bound branch. Against the
measured cross-section that implies an FF beam height of **12.95 um** — a
thoroughly plausible line-beam setting. So unlike the FF reference run, where
V_gauge came from `Hbeam * pi * Rsample^2` and was 2.5e10 um^3, **the Ce gauge
volume looks like a real setting rather than a canned constant**, and the
grain sizes are approximately right.

The correction as a function of the one remaining unknown:

| FF beam height (um) | V_illum (um^3) | volume scale | median R (um) |
|---|---|---|---|
| 5 | 3.86e6 | 0.386 | 2.97 |
| 10 | 7.72e6 | 0.772 | 3.74 |
| **12.95** | **1.0e7** | **1.000** | **4.08** (unchanged) |
| 20 | 1.54e7 | 1.544 | 4.71 |
| 50 | 3.86e7 | 3.861 | 6.40 |

**Still open:** the actual FF beam height. The tomogram supplies the
cross-section; the beam height is a slit setting and is not in the FF parameter
file (`Hbeam 2000` is a search bound). Until it is read from the beamline log
the median radius is 4.08 um x (h/12.95)^(1/3).

*Caveat, stated:* the threshold is not stationary (spread 0.22-0.31), so the
cross-section itself carries perhaps 20 % uncertainty — 7 % in the radius. The
0.63 % uniformity along the height is consistency, not accuracy.

*Packing:* the 2496 indexed grains account for **7.4 %** of the gauge volume
(scale-invariant, as established). For a 1 mm rod with 4 um grains a filled
gauge volume would hold ~35 000 grains, so this is an indexing-completeness
statement, not a size error.

### 4.6 The Ce grain sizes were overstated 2.34x

With `Vsample` measured (7.786e5 um^3) rather than the 1e7 in the FF parameter
file, the 2496 gamma-fcc grains move by `radii x0.427`:

| percentile | as reported | corrected |
|---|---|---|
| p10 | 3.681 um | 1.572 um |
| p25 | 3.855 | 1.646 |
| **p50** | **4.076** | **1.741** |
| p75 | 4.317 | 1.844 |
| p90 | 4.557 | 1.946 |
| mean | 4.103 | 1.752 |

**RETRACTION.** Earlier in this campaign I wrote that "the Ce gauge volume
looks like a real setting rather than a canned constant, and the grain sizes
are approximately right", on the grounds that `Vsample 1e7` corresponded to a
plausible 12.95 um beam height. **That was circular** — I had derived the
12.95 um *from* `Vsample 1e7` and this cross-section — and it is now refuted by
the log: the beam was **1 um**, so `Vsample 1e7` was **12.8x too large** and the
sizes were not approximately right. The lesson is the one this campaign keeps
relearning: a number derived from the thing it is being compared against
cannot check it.

*What is still owed:* the +/-8 % volume uncertainty from the soft boundary
(§3.10), and an independent check of the corrected sizes — NF grain sizes on
the same specimen would do it, though NF on this sample is known to be hard.

### 4.7 Both Ce phases, and a negative on the structure-factor drop

Same specimen, same FF scan, same beam, so **both phases take the same gauge
volume** and the same radius scale 0.4270:

| phase | grains | median R reported | corrected | packing |
|---|---|---|---|---|
| gamma-fcc | 2496 | 4.076 um | **1.741 um** | 0.0738 |
| dhcp | 23 | 6.678 um | **2.852 um** | 0.0032 |

The dhcp grains are 1.64x larger in radius than the gamma grains, and that
ratio is **unchanged** by the correction — a common gauge volume cancels
between phases. It was already in the old numbers; only the absolute scale
moved.

**NEGATIVE, and it was worth checking rather than assuming.** The obvious
expectation was that the Phase-2 structure-factor work would recover dhcp
grains, because a dhcp reflection list is ~10 % basis-extinct and those
reflections sit in the completeness denominator. Computed for this cell
(P6_3/mmc, a 3.6671 c 11.805, 2a + 2c) over the detector's 2theta_max
10.13 deg: **9 of 91 reflections have |F|^2 = 0, ceiling 0.9011** — the effect
is real for the full list.

But the run used only `RingThresh` rings **2, 3, 5, 6, 7, 9, 12**, and

    WITHIN the used rings: 0 extinct of 7  ->  ceiling 1.0000

Every extinct ring — 1, 8, 11, 21, 28, 38, 39, 41, 45, 48, 51, 54, 55, 67, 70,
72, 79, 85, 89 — is *outside* the selection. They are the `(0,0,l)` reflections
with `l != 4n`, which is the 4-layer ABAC stacking signature; the used rings are
all `(1,0,l)`. **So `DropForbiddenReflections` / `ConfidenceMetric filtered`
would change nothing on this run.** The operator's ring selection had already
avoided the problem by hand.

*Verified against the run's own file, not a regenerated one* (the hard rule):
`ff_dhcp/results/LayerNr_1/hkls.csv` gives ring 1 = (0,0,-2) d 5.9025,
ring 2 = (1,0,0) d 3.1758, ring 4 = (0,0,-4) d 2.9513, ring 8 = (0,0,-6)
d 1.9675, with 91 rings and 1226 reflections — matching the generated list
exactly, including the multiplicity sum.

**What is still live:** `ConfidenceMetric weighted` is *not* covered by this
negative. Median |F|^2 across the seven used rings runs 0.052 (ring 7), 0.065
(2), 0.143 (9), 0.171 (6), 0.191 (3), 0.389 (12), 0.548 (5) — a **10x spread**.
Treating a ring-7 reflection as exactly as obligatory as a ring-5 one is the
mis-weighting `weighted` exists for, and testing it needs a full FF re-index.
The same shape as NMC811: `filtered` a no-op, `weighted` still open.

## 5. Open questions, and claims that were RETRACTED

### 5.1 OPEN — nothing has been through the whole chain

`SampleShape`, the four readers, the corrected estimator and the registration checks all
have unit gates. **No real reconstruction has been read, registered, and used to correct a
real `Grains.csv`.** Until one has, the correction is machinery, not a result. Specifically
unverified:

- the bt_1id_jun25b NMC811 tomo is 128×128 raw projections needing a reconstruction first;
- ~~the Ce tomo pixel size is unresolved~~ **RESOLVED 2026-08-23, see §4.3**:
  0.69 µm from the per-scan `tomo_metastr`, not 1.17 µm. The 1.9× inconsistency
  closes to 1.09×.
- the bt_1id_jun25b route is **closed** (§4.4): its tomogram cannot produce a mask.
  Ce is now the only candidate, and its μ·D 1.63 is 33× bt_1id_jun25b's, so its
  reconstruction has real absorption contrast to threshold — but that is a
  prediction, not a measurement, until its tomo is reconstructed.
- V3 (transmission channel) and V4 (NF grain-map Dice) are not implemented. V4 is the
  strongest check and the plan gates Phase 5 on it.

### 5.2 Not yet retracted, but flagged

Nothing has been retracted in this campaign yet. The nearest thing is the reading of the
Ce transmission profile as "a furnace with two windows" — wrong on both counts (no furnace;
the plateaus were **unilluminated detector** outside the beam's columns 252–2006), caught
before it reached any doc. Recorded in the parent thread's checkpoint, not here, because it
concerned the projections rather than the shape.

## 6. Measurement ledger

| quantity | value | file + command that produced it |
|---|---|---|
| grains summed / `V_gauge` | 6.5 % | `ff_refiner_prepost/result/LayerNr_1/Grains.csv` + `paramstest.txt`; `Σ(4/3πR³)` over col `GrainRadius` |
| `V_gauge`, FF reference run | 2.513e10 µm³ | `paramstest.txt` (`Hbeam 2000`, `Rsample 2000`, no `Vsample`) via `GaugeVolume.from_param_file` |
| C-vs-Python grid pixel disagreement | 75.0 % | 200k uniform points, `n=2048`, `px=1.5 µm`; `trunc(x/px + n/2)` vs `trunc(x/px) + n//2` |
| powder double-count inflation | 1.60× volume, 1.17× radius | μD ~ U[0.2,0.8], 4000 spots; `test_shape_correction.py::test_the_guard_HAS_POWER...` |
| μD, NMC811 s5 | 0.05 | `tomo/data_nmc811s5tomo1.raw`, transmission over the illuminated columns |
| μD, Ce ht525_s2 | 1.63 ± 0.02 | `tomo/tomo_Ce_ht525_s2`, 8 angles, cols 252–2006 |
| pixel size, bt_1id_jun25b s5 tomo | 0.708 µm | `metadata/bt_1id_jun25b/nmc811s5tomo1/nmc811s5tomo1_TomoFastScan.dat`, `tomo_metastr` |
| pixel size, Ce ht525_s2 tomo | 0.69 µm | `new_data/bt_1id_jul26/tomo_Ce_ht525_s2/tomo_Ce_ht525_s2_TomoFastScan.dat`, `tomo_metastr` |
| bt_1id_jun25b mask threshold spread | 9.205 (4.64× in radius) | `tomo/phase3_shape/phase3_report.json`, `phase3_bt_1id_jun25b.py` |
| bt_1id_jun25b mask extent | 81.4 × 90.6 µm in a 90.6 µm FOV | same |
| regenerated `.raw` sha256 | `d933c7167a271406` (== reference) | `phase3_shape/ingest_parity.log`, `ingest_parity.py` |
| auto-centring vs human | +13.000 vs +13.00, 0.000 px | `phase3_shape/shift_gate3.log`, `shift_gate.py` |
| TV bias on this reconstruction | −1.6 px vs variance | same |
| strongest row-mean attenuation | −0.0005 (drift-dominated) | `shift_gate2.log` |
| μ(Ce) at 95 keV | 18.94 cm⁻¹ | `midas_hkls.absorption` (NIST), ρ = 6.77 g/cm³ |
| μ(NMC811) at 51.9 keV | 6.53 cm⁻¹ | `midas_hkls.absorption`, ρ = 4.8 g/cm³ assumed |

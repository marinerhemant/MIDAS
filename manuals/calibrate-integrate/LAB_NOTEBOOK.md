# Lab notebook — evidence, measurements, and retractions

The spine says what to do. This says **why**, and records what had to be
withdrawn. Read this before re-investigating anything here: several attractive
hypotheses are recorded as refuted, with the measurement that killed each.

§1–§6 are 2026-08-18/19 on a 48-panel Pilatus (1475 × 1679, 172 µm), APS 20-ID,
CeO2 at 63 keV, frame `ceria_..._000868.tiff`. **§6bis–§11 are a second detector
and a second failure mode** — the 4-panel GE Hydra at 1-ID, two calibrants in one
exposure, beam centre off the panel — and are scoped in §6bis. Scripts and raw
outputs are archived with the release handoff for this work; the paths live there
rather than here, because they are on beamline scratch and will not outlive it.

**Status labels are load-bearing.** ESTABLISHED = survived adversarial review.
PROVISIONAL = measured, not independently attacked. RETRACTED = withdrawn, with
the reason.

---

## §1. The active geometry block was wrong for its own frames — ESTABLISHED

`ps_manual.txt` carried two calibration blocks. Four independent routes agreed
the active one did not match the ceria frame beside it:

| route | result |
|---|---|
| ring radii from the active `Lsd` | 2.60 px short (+4619 µε) |
| ring radii from the commented `#Lsd` | 0.24 px (−462 µε) |
| `make_seed`, image only | Lsd 649 509 µm; BC within 0.05 px of the commented block, 6.7 px from the active |
| full from-scratch calibration | Lsd 649 795 µm, 66.1 µε held-out |

The frame is named `..._650mm_...`; 650.09 mm is the commented value. The active
block's `FileStem` points at `..._000059` under a different user's directory.

**Why it matters procedurally:** had the calibration been seeded from the active
block, it would have started 3 mm away with a 6.7 px beam-centre error, and the
refiner would have absorbed some of that into distortion rather than rejecting
it. Hence hard rule 2.

Confirmed visually by contouring the R map over the frame — the from-scratch
contours sit on the ring crests, the active ones inside.

---

## §2. `SubPixelLevel > 1` corrupts the CUDA integrator — PROVISIONAL

`MapperCore.c:372-375` stores each sub-pixel's position as a *fractional* pixel
coordinate. `IntegratorFitPeaksGPUStream.cu:916` converts it back with a C cast
that **truncates**, so sub-pixels with a negative centre offset read the
neighbouring pixel. Only this CUDA branch truncates —
`IntegratorZarrOMP.c:1740` interpolates.

**The test that works** holds the map fixed and changes only the lookup, against
a round-to-parent reference (`rint(raw + sp_c) == raw`), at SubPixelLevel 2:

| arm | max rel | bins > 2× | in-band L2 |
|---|---|---|---|
| truncate (the CUDA lookup) | 24.32 | 1218 | 481 799 |
| round to parent | 7.7e-05 | 0 | 9.95 |
| **the real CUDA binary** | 24.315 | 1218 | 481 799 |

Rounding removes **99.998 %**, so the map is not at fault. The third row is the
compiled binary, not a surrogate — it matches the Python proxy to 0.03 %.

`SubPixelLevel 0` is **bitwise identical** to `1` (the C tests
`if (SubPixelLevel > 1)`).

### What was refuted in the first version of this finding — RETRACTED

Four reviewers were given only the claim and the data. Two returned *refuted*.
All of the following was wrong and is withdrawn:

- **`floor` vs `bilinear` as the isolating comparison.** Two wrong arms — both
  misuse the fractional coordinate. It understated the real error **5.8×** (4.17
  vs 24.32). Bilinear is itself 7.8× off the reference.
- **"Six bins go negative."** They are mask-flagged and blanked in production;
  the harness never loaded `maskMap.bin`. Magnitudes −0.009 to −0.71 counts
  against I ≈ 240–340.
- **The severity mechanism.** Displaced reads do *not* land on masked or hot
  pixels: only 0.68 % do, and zeroing every masked pixel leaves the result
  unchanged (24.316 → 24.316). It is a half-pixel shift on a steep ring flank —
  median |dI/dR| 762.7 at the > 2× bins against 1.586 across the band.
- **"Confinement to the cardinal bands is evidence."** True but not diagnostic:
  both candidate causes are gated by the same `spLevel` test.
- **The "0 bins differ at SubPixelLevel 1" control.** Cannot fail — at integer
  coordinates bilinear reduces to weights (1, 0, 0, 0) analytically. A theorem
  dressed as a measurement.

Stays PROVISIONAL: the surviving formulation was built *during* the review and
has not itself been attacked fresh.

---

## §3. Match rings by position, not by rank — ESTABLISHED

Comparing v2's integrated lineout to the C binary's by pairing the *k*-th
strongest crest in each produced an apparent 103 px discrepancy. Matching by
position instead: **17 of 19 rings agree to exactly 0.00 px**, the two exceptions
being at the inner `RMin` cut-off and off by one R bin.

The kernels rank rings differently because they weight differently. Ranking is
not a property of the geometry; position is.

---

## §4. The published cardinal-aliasing result did not reproduce here — ESTABLISHED

`packages/midas_integrate/dev/paper3/cardinal_angle_aliasing.tex` reports σ/µ at
η = 0 rising 5.00 → 7.94 → 8.52 % for SubPixelLevel 1/2/4, and prescribes
`GradientCorrection`.

Aliasing is a *high-frequency* oscillation, so the discriminating metric is the
residual about a smooth fit **with a non-cardinal control**:

| path | η=0 | η=90* | η=180 | **η=45 control** |
|---|---|---|---|---|
| v1 floor | 1.229 % | 1.087 % | 1.614 % | **1.809 %** |
| v1 gradient | 1.142 % | 0.796 % | 1.393 % | **1.670 %** |
| v2 polygon | 1.412 % | 1.075 % | 1.584 % | **2.084 %** |

\* η = 90 is thin — a module gap leaves ~29 % of bins with data and only 4–10 of
40 rings clear the fit threshold. Do not weight it.

**The control is worse than every well-sampled cardinal band.** There is no
aliasing signal above the noise floor on this frame, and `GradientCorrection`
lowers the residual uniformly — control included — so it is smoothing, not
correcting.

### Also refuted: the paper's stated mechanism — RETRACTED (paper text, not code)

The paper explains the rise as splitting "merely reproducing the same
pixel-averaged value at finer spatial locations". That description **is** the
round-to-parent lookup, and implementing exactly it is a no-op:

| lookup | SPL1 | SPL2 | SPL4 |
|---|---|---|---|
| round-to-parent (the paper's words) | 2.11 % | 2.11 % | 2.11 % |
| bilinear (what the paper's code ran) | 2.11 % | 3.71 % | 3.73 % |
| floor (CUDA truncation) | 2.11 % | 8.48 % | 8.48 % |

Splitting the *geometry* changes nothing; resampling the *intensity* is what
moves the number. The paper's table stands; its explanation does not. **Not yet
acted on in the paper.**

**Process failure worth recording:** an earlier draft of the handoff told the
user their integrations were "carrying a ~5 % oscillation". That was the paper's
number, from the paper's detector, quoted before testing it here. Do not import a
number across datasets.

---

## §5. δLsd is the wrong knob for a misplaced module — ESTABLISHED

A module is displaced *in the detector plane*, giving a **constant** ΔR. A
per-panel δLsd gives **ΔR ∝ R**. Fitting the first with the second rails it:

| per-panel DOFs refined | held-out strain | railed |
|---|---|---|
| rigid δy,δz,δθ + δLsd + δp₂ | 76.5 µε | 16/48 + 2/48 |
| radial δLsd + δp₂ | 64.7 µε | 16/48 |
| radial δLsd only | 65.2 µε | 4/48 |
| **modules only, ±1 px, δLsd = 0** | **61.9 µε** | 12/48 |
| **modules only, ±2 px, δLsd = 0** | 66.1 µε | **2/48** |

The ±1 px bound gives the best strain but rails heavily because the detector's
real modules reach 1.28 / 1.54 px in the previous calibration. Geometry was
stable at 649 710–649 910 µm across all five, i.e. **the geometry does not depend
on resolving the panel parameterisation**.

### Identifiability, and a second nullspace

A powder ring constrains only the **radial** component of a module's translation:
with η spread much below 90° on a module, the 2 × 2 (δy, δz) Fisher block is
rank-1 and the tangential component is unidentifiable
(`packages/midas_calibrate_v2/dev/paper/midas_v2_test/sim_chord_analysis.py`).

Separately, a **radial expansion** of the whole panel field — every module pushed
outward in proportion to its radius — shifts ring radii the way an `Lsd` error
does. `fix_panel_id` and Σ panel = 0 do not remove it. Measured: 11 % of the
fitted field sat in that mode, and moving the amplitude into `Lsd` cancelled
**73 %** of its effect on 2θ (0.361 → 0.097 px RMS). Not exact, because a panel
expansion is a step function per module while `Lsd` is smooth in R. Hence the
opt-in `add_panel_no_expansion_constraint`.

**A method error worth recording:** the first attempt to test that degeneracy
compared `R(pixel)` before and after the gauge move and found "not degenerate".
`R(pixel)` is the in-plane radial distance and does **not depend on `Lsd`** — the
invariant is 2θ. Both signs of the correction gave the identical answer, which is
what exposed the mistake.

---

## §6. Python/C parity — ESTABLISHED, with a scope limit

`midas_integrate`'s `mode='floor'` reproduces the compiled
`IntegratorFitPeaksGPUStream` to **2.2e-07** max relative on the 1-D lineout,
0 of 1800 bins differing by more than 1e-4, on real 48-panel data.

**Scope limit:** that was measured at `SubPixelLevel 1`, where the two lookups
are identical by construction. The proxy at SubPixelLevel > 1 was validated
separately in §2 (0.03 % on in-band L2) by running the compiled binary at level 2.

v2 CPU vs CUDA agree **exactly** (max absolute difference 0 over 648 000 bins) —
float64 throughout, no reduction-order noise.


---

## §6bis. Second dataset: 4-panel GE Hydra, mixed CeO2 + LaB6, off-panel beam centre

Sections §7–§11 were measured 2026-08-17/19 on a **different detector class** from
the rest of this notebook: the four monolithic GE panels of the 1-ID Hydra
(2048², 200 µm, `tx` = 300/30/120/210°), CeO2 + LaB6 mixed in one exposure at
80.802 keV, sample-to-detector ≈ 2.73 m. The beam centre lies **beyond a panel
corner**, so every ring is a partial arc and each panel sees one contiguous
azimuthal wedge of 66–73°.

That is outside §0's reference geometry in two ways that matter — mixed calibrant,
and an azimuth budget four to five times smaller than a beam-centre-on-panel
detector. Treat the *procedures* below as transferable and every *number* as
belonging to this detector.

Data: `/gdata/dm/1ID/2025/bt_1id_jul25/data/ge{1..4}/cal_CeO2_LaB6_HL_10f6s_sweep`.

---

## §7. What a second calibrant buys, and what it does not — ESTABLISHED

**It does not buy wavelength.** SVD of the (λ, Lsd) Jacobian over the real ring
sets, at the reference geometry:

| ring set | cond(JᵀJ) | soft direction |
|---|---|---|
| CeO2 only (11 rings) | 1.148e3 | (+0.16792, −0.98580) |
| LaB6 only (19) | 1.037e3 | (+0.16762, −0.98585) |
| CeO2 + LaB6 (30) | 1.060e3 | (+0.16772, −0.98583) |

The soft direction rotates by **0.011°**. Both phases enter the forward model
only through their d-spacings, so λ → kλ, Lsd → Lsd/k acts identically on every
`d_k` — a second calibrant adds *rows* to the Jacobian, not a direction. Linked
distances give cond = 2.6e1 on the same data, a factor 44. This is an independent
route to hard rule 9 and agrees with the planted-error test in §ENVELOPE.

**It does not buy azimuth.** Both powders illuminate the same wedge.

**It does buy √N and a cross-check.** Fit points 533 → 1060 after the blend cut
(2.03×); σ on Lsd / BC / tilts tightened 29–43 %, against 30 % predicted by √N.
Nothing beyond counting.

**Blend accounting.** After merging exact hkl degeneracies (14 rows absorbed on
ge1), a 12 px cut costs 6 or 7 rings of about 40, consistently across all four
panels. The degeneracy merge is not optional: LaB6 (300)/(221) and CeO2
(511)/(333) are one physical ring with two labels, and any blend rule reads them
as zero-separation doublets.

**Powder quality is not symmetric.** Per-η peak-height CV, four panels:
CeO2 0.08–0.11, LaB6 0.36–0.47. LaB6 is 3.5–4.5× grainier on every panel, and it
sets the residual floor: the best joint fits reached 45.6 µε on CeO2 rings and
68–69 µε on LaB6's, whichever ring set was fitted.

---

## §8. A narrow azimuthal wedge does not determine the harmonics — ESTABLISHED

The shipped calibrations for the four panels had **3, 4, 7 and 7 of 15**
distortion coefficients pinned at ±0.002. ge3 and ge4 sat exactly on the bound.

Refitting reproduced the cause. Honest per-iteration strain (re-extracted at each
post-refinement geometry), ge1, everything else held fixed:

| refined | trace (µε) | converged |
|---|---|---|
| full distortion | 232, 181, 613, 779 | no |
| `"radial"` (iso_R2/R4/R6 only) | 199, 284, 1380, 2718 | no |
| `"none"` | 91, 72, 139, 154 | no |
| `"none"` + per-ring filter | 84.2, 84.4 | **yes, 2 iterations** |

Two things worth separating. First, freezing the distortion is the large move
(181 → 72). Second, **`"radial"` was not enough** — the three iso terms are
near-collinear over the available ρ range and diverged worse than the full set.
Do not treat "radial only" as a safe default; run the gate and check the trace.

With distortion frozen the four panels agree on the distance to 0.26 mm
(SD 0.11 mm), so the geometry does not depend on resolving the distortion — the
same conclusion §5 reached about the panel parameterisation.

---

## §9. RhoD is a normalisation, and the wrong value silently kills the radial terms — ESTABLISHED

The shipped files carried `RhoD` = 2e6 µm against an outermost fitted ring at
551–632 kµm, i.e. ρ_max ≈ 0.28–0.32. Then ρ⁴ = 8e-03 and ρ⁶ = 1e-03, and
`iso_R4` / `iso_R6` returned 1σ of 0.9 to 15 on coefficients of order 1e-03 —
unmeasured, and railed. Setting `RhoD` to the outer ring radius (ρ_max ≈ 1)
brought every σ to the same order as its coefficient.

Nothing about the residual announces this: the strain is unremarkable either way.
It is only visible in the covariance, which is why the gate reports the ratio.

---

## §10. The per-ring quality filter buys convergence, not a smaller number — PROVISIONAL

**A claim made earlier in this work was wrong and is corrected here.** The
per-ring filter was described as "worth about 4× in converged residual". Isolating
one knob at a time (the table in §8) shows the 4× belongs to **freezing the
distortion**, not to the filter. The filter's contribution is stability: with it
the loop converged in two iterations and stopped; without it the same fit
wandered 91 → 72 → 139 → 154 µε, and its apparent best (72.5) was an iterate the
next one destroyed.

That is still worth having — an unconverged run reports whichever iterate was
luckiest — but it is a different claim, and the earlier one is withdrawn.

Two implementation notes, both found by measurement rather than design:

- The first baseline-referenced SNR estimator divided by the scatter at the
  radial-window ends, which goes to zero wherever those ends are flat. Measured
  maxima of **1.03e7**, which made any SNR threshold inert — 34 of 36 rings
  "passed" at every threshold. Fixed with a counting-statistics floor,
  `max(std_ends, √baseline)` (`packages/midas_calibrate/midas_calibrate/estep.py:270`);
  the same frame then spans 0.0–22.3.
- `MinEtaBinsPerRing` is an **absolute** count and scales with `EtaBinSize`. On
  one frame the best-covered ring carried 13 fits at 5° bins and ~36 at 2°. A
  threshold tuned at one binning is not portable; read the distribution off
  `ring_quality()`.

Stays PROVISIONAL: measured on one panel, one frame, and not attacked fresh.

---

## §11. Per-phase sample position: an upper bound, not a measurement — PROVISIONAL

Two capillaries stuck together would put each powder at its own distance. Fitting
that with the tilts **shared** (one detector cannot tilt differently for different
powders) gives, on ge1:

| model | dLsd, LaB6−CeO2 | transverse | mean strain |
|---|---|---|---|
| shared position | — | — | 54.9 µε |
| + dLsd | **−71.8 ± 34.4 µm** (2.1σ) | — | 57.4 µε |
| + dLsd, dBC | −191.6 ± 123.0 µm (1.6σ) | −8 µm, −23 µm (0.7σ, 1.2σ) | 57.4 µε |

The powders are co-located to ~100 µm along the beam and ~30 µm across it, adding
the offsets does not improve the fit, and the three-offset model does not converge
(dLsd swinging −192 → +318 µm across iterations).

**Two reasons this is an upper bound rather than a measurement.** It is 2.1σ from
one panel. And `dLsd/Lsd` = −2.6e-05 is *exactly* what a LaB6 lattice constant of
4.15678 Å instead of 4.15689 would produce — a difference of 0.0001 Å. One frame
cannot separate them; several exactly-known distances can.

**A method error worth recording.** The first version of this measurement let each
phase have its own tilts and reported a **1.43 mm** offset at 3.7σ. The tilts were
absorbing the difference. Sharing them collapsed it to 72 µm. The 1.43 mm figure
is withdrawn. Same shape as the `R(pixel)` mistake in §5: a quantity that looked
like a measurement was an artefact of what else was left free.

---

## §12. Bad-pixel sentinels are not always negative — ESTABLISHED

Everything in this doc set had assumed the Pilatus convention: gaps and bad
pixels marked with `-1` / `-2`, caught by `img[img < 0] = 0`. That assumption is
false for a whole detector class, and it fails in the dangerous direction.

**What happened.** A survey of `/gdata/dm/1ID` turned up
`2026/bt_1id_apr26/data/eiger` — CeO2 and LaB6 on an EIGER2 CdTe 16M. Reading a
frame gave `min 0  max 4294967295`. That is exactly `2**32-1`, the uint32
maximum, and it covers **1 285 014 of 18 093 576 pixels = 7.102 %**.

**It is the module map, not noise.** Measuring the sentinel bands directly:

```
horizontal gap bands (row, width): 7 bands, all width 38, at 512/1062/…/3812
vertical   gap bands (col, width): width 12 at 1028/2068/3108, width 2 at 513/1553/2593/3633
→ 8 module rows × 512 active px,  4 module cols × 1028 px (each split 513|2|513)
   8·512 + 7·38 = 4362      4·1028 + 3·12 = 4148
```

which is the Dectris EIGER2 16M layout exactly. So the sentinel marks physical
module gaps and flagged pixels — it is a bad-pixel map the detector hands you,
not something to derive statistically.

**Why the old guard is worse than useless here.** `img[img < 0] = 0` leaves
4.29e9 in place. A small negative that slips through biases a bin slightly; a
value nine orders of magnitude above the signal dominates any centroid, radial
profile or seed it touches. The failure is silent because the array is finite,
positive and the right shape.

**The averaging trap.** `_read_hdf5` means over frames. Detect the sentinel
*after* the mean and it is already gone: blend `2**32-1` with three real counts
of 10 and the result is 1 073 741 831, which equals no sentinel and looks like a
hot pixel. Detection has to run on the raw integers, before the average. The
mask is the union over frames — a pixel bad in any frame is bad in the result.

**Regression check before changing the default.** The new `bad_value="auto"`
only claims the unsigned dtype-max, so it had to be shown not to disturb the
detectors already in use. On real 1-ID frames:

| detector | dtype | pixels at dtype-max |
|---|---|---|
| ge5 ×3, ge3 ×1 | uint16 (max 65535) | **0** |
| varexC | float32 | rule does not apply |
| pilatus | int32, min −2 | rule does not apply (signed) |
| eiger | uint32 | 1 285 014 = 7.102 % |

So the change is inert on GE, Varex and Pilatus and active only where the
problem exists. Signed and float data are deliberately excluded — there the
`< 0` convention already applies and `int32` max is a plausible real count.

**Status.** ESTABLISHED for this detector. The *mechanism* (vendors mark bad
pixels out of band, at either end of the range) is general; the 7.102 % and the
module geometry are this detector.

---

## §13. The 1-ID energy is a K edge, not the monochromator readback — ESTABLISHED

Rule 9 said "take λ from the beamline". Working out what that means in practice
across the 1-ID archive sharpened it into something checkable.

**The claim.** 1-ID tunes the monochromator to an absorption K edge and stays
there. So the energy is the *tabulated K edge of the foil element*, and the
element is recorded in `~/new_data/<expt>/fastsweep_Emon*.txt`.

**Evidence.** Over 116 beamtimes that hold a calibrant:

- 102 name an element in an Emon file.
- Of the 82 that also log an energy, **74 (90 %)** sit within 0.3 % of a
  tabulated foil K edge — edges read from MIDAS's own
  `midas_pdf/midas_pdf/data/fluor_edges.json`, not a typed list.
- Where the metadata names the element, it is the right one in **68 of 80**.
- The elements in use are exactly the ones that make the canonical 1-ID
  energies: Re 71.676 (44 beamtimes), Tb 51.996 (13), Yb 61.332 (11),
  Au 80.725 (9), Ho 55.618 (6), Hf 65.351 (5), Bi 90.526 (5), Ir 76.111 (4).

**The readback is biased low.** `fastpar_*.par` field 10 is the monochromator's
own energy. Against the tabulated edge it sits a median **−0.040 %**
(n = 68, range −0.18 % to +0.10 %). Small, but systematic and it goes straight
into `Lsd`.

**`exp_setup.yml`'s `EDGE:` is stale.** 18 beamtimes have both it and an Emon
element; they **disagree in 9**. In those 9 the Emon element matches the logged
energy 7 times and the yml value **0** times (2 were off-edge). It reads
`EDGE: Re` on runs plainly at the Yb, Tb, Pt or Ta edge.

**Not every run is on an edge.** 8 of the 82 are at round settings — 95, 100,
100.2 keV — where no foil edge is within 0.3 %. There the readback is the only
source and should be carried with ±0.1 %, not quoted as exact.

**Two false leads, recorded so they are not re-run.** (i) The element token in
the `.par`/Emon row is *not* the edge on off-edge runs: `bt_1id_jul26` names
Pb but ran at 95 keV, and Pb K is 88.004. (ii) A first pass took energy from
field 10 of *every* `.par` file and got nonsense (16867, 5147, 27) — only the
`fastpar_*` layout puts energy there; `per_frame_waxs_*.par` and `waxs_*.par`
put a frame counter in the same column.

**Substituting the edge changes the answer, and an independent distance says it
changes it the right way.** The rule was applied for real: per-scan energy from
the filename, snapped to the nearest tabulated K edge. Three beamtimes carry a
detector distance written into the filename by the acquisition, which the fit
never sees — so it is an independent check.

| beamtime | `Lsd` before | `Lsd` after | recorded in the filename |
|---|---|---|---|
| `mpe_dec24` | 492.6 mm | **680.7 mm** | 680 mm |
| `bt_1id_mar23` | 1875 mm | **2371 mm** | 2400 mm |
| `bt_1id_nov24` | 811 mm | **952.6 mm** | 950 mm |

The energies had been recorded per *experiment* on beamtimes that changed energy
mid-run; 18 of 124 records were wrong by up to 46 %. The fit residual barely
moved across that error — which is rule 9 in action, and is why the independent
distance is the only thing that caught it.

**Status.** ESTABLISHED as a description of the archive, and the edge value has
now been used as a fit input at archive scale. It still does not make the energy
a *measurement* of any given exposure — that needs several known distances
(rule 9), and §15 records what happened when this was attempted from the residual
instead.

---

## §14. A detector that cannot see the calibrant still calibrates — ESTABLISHED

The 1-ID archive run (2026-08-20, 252 exposures) surfaced a failure no gate in
this doc set covered: an exposure whose rings are not on the detector at all.

**What happened.** 30 work units were built from `data/pixirad/` folders because
their filenames matched `CeO2` / `LaB6`. The pixirad at 1-ID is a **SAXS**
detector: its folders hold `glassy_carbon_1s`, `gC_1s`, `test_Ag_behenate` and
`bright_1s`/`dark_before` flat-fielding, and Almer's calibration spreadsheet
independently lists "glassy carbon C" as its calibrant for 11 rows. The CeO2
files are stray WAXS test exposures.

**The geometry is decisive and needs no fit.** 402 × 1024 px at 62 µm is
25 × 63 mm, so the panel reaches 34 mm from a centred beam. At 3300 mm and
71.676 keV the CeO2 (111) ring sits at

    2θ = 2·asin(λ/2d) = 3.17°,   R = Lsd·tan(2θ) = 183 mm

— **six times past the edge**. Zero rings, at every distance the archive used
(1780–3600 mm).

**The fitter does not notice.** Of the 30, sixteen failed with `RuntimeError: No
reflections within max 2θ` — the right answer, arrived at late and with a
message that blames "geometry / lattice / wavelength" rather than saying the
detector cannot see the calibrant. The other **fourteen converged**, entered the
results table, and were only caught later by an independent check against the
distance recorded in the filename: one returned **Lsd = 26.9 mm against a
recorded 3300 mm**.

**It is not confined to small detectors.** The same gate halts 12 GE quad panels
from `bt_1id_jul25b` at 3300 mm — a full 2048² panel, simply parked too far for
CeO2 at that energy. Those had fitted 480–573 mm.

**Scored against the run** (`detector_scope_gate`, `min_rings=3`, 252 units):

| | n |
|---|---|
| halted, and the run had failed anyway | 16 |
| **halted, and the run had "succeeded"** | **26** |
| passed, and the run failed | 2 |
| passed, and the run succeeded | 208 |

**Status.** ESTABLISHED — it is arithmetic, not inference. The 26 is the number
that matters: no post-fit diagnostic catches these, because every post-fit
diagnostic is grading a fit that converged.

**Corollary for surveys.** Classifying detectors by folder name and files by a
`ceo2|lab6` regex is how a SAXS detector entered a powder archive. The physical
check — does this detector, at this distance, subtend the calibrant's rings —
is one line of geometry and belongs before the survey, not after the fit.

**Site convention, measured not assumed.** 1-ID mounting distances, taken only
from fits their own recorded distance confirms: single panel **0.5–1.9 m**
(largest confirmed 1905 mm), GE quad **1.0–3.3 m** (largest confirmed 3257 mm).
As a standalone flag over the archive this scores precision 58 %, recall 49 % —
useful as a cross-check, useless as a sole gate. The Eiger is a recent addition
and does not yet have a characterised envelope (`bt_1id_jul26d` runs it at 1935 mm
`in_chamber`); leave it exempt rather than inventing a band for it.

---

## §15. Energy is not recoverable from the fit residual — REFUTED (a claim of mine)

**Registered as a claim, then killed.** Having found that 58 of 252 archive units
had no trustworthy energy, the obvious move was: calibrate each frame at every
plausible 1-ID energy and keep the candidate with the lowest strain residual.
It appeared to work — the true energy in the top 3 of 13 candidates in **25 of
30** units, p = 4.9e-12. It is wrong, and the way it is wrong is instructive.

**Why it cannot work.** The λ/`Lsd` degeneracy (rule 9) is broken only by the
`tan`/`asin` nonlinearity. Expanding, the residual signature of a wrong λ is

    ln[R(λ')/R(λ₀)] = c₀ + c₂ρ² + c₄ρ⁴ + O(ρ⁶)

and that is **exactly the span of `{Lsd, iso_R2, iso_R4, iso_R6}`** — the very
parameters `refine_distortion=True` sets free. The method refines away the only
thing it depends on. Planting a −5.94 % energy error on a real frame:

| free parameters | residual left | detectable against the 11.3 µε floor? |
|---|---|---|
| `Lsd` only | 555 µε | yes, easily |
| `Lsd` + `iso_R2` | 0.70 µε | no |
| `Lsd` + full radial block (**the default**) | 8.8e−06 µε | no, by eight orders of magnitude |

**Then why did the control score 83 %?** Because the score did not come from the
data. The best **constant, data-blind** guess — always answer {Re, Au, 100 keV} —
scores **25 of 30, identical** (McNemar p = 1.00): 1-ID's energies are so
concentrated that guessing the three commonest beats chance by itself. The
"−5.9 % systematic offset" that the bias correction removed is
**−4.260 keV = 67.416 − 71.676 keV exactly**, i.e. the Ta→Re edge spacing,
learned from the 16 of 30 units that are Re. And the leave-one-beamtime-out that
was supposed to make it honest was **vacuous**: all 17 folds returned the same
correction, span 0.000.

**What the refuters cost, and what to copy.** Four lenses, two REFUTED. The
reproduction lens also found that the headline 19/30 does not reproduce from the
stated recipe (a percentage correction gives 17/30). Any "we recovered a
parameter the physics says is degenerate" result should face the same three
questions: *what does a constant blind guess score; is the correction a physical
constant in disguise; does the cross-validation actually vary between folds.*

**The constructive half — use the degeneracy.** Since a wrong λ goes into `Lsd`
almost quantitatively, `Lsd` **is** an energy readout: over the candidate scan
the fitted distance tracks the assumed energy at log-log slope
**1.0066 ± 0.0037**. For the 17 units whose filename records the detector
distance, choosing the candidate whose fitted `Lsd` matches that distance gives
the true energy **17 of 17**, using no residual at all. This does not rescue the
58 units that have neither an energy nor a distance; nothing does.

**Status.** REFUTED (physics, statistics), logged in `~/.claude/skill-log.jsonl`.
The rule-9 consequence is now written into `HARD_RULES.md`.

---

## §16. How well one powder calibration determines a distortion field — ~0.25 px

The archive's 143 ring-verified records make a question testable that a single
beamtime cannot ask: **is a ge5 distortion field reusable?** Preregistered
(`PREREGISTER_pooled.md`), then run — `pooled.py`, output `pooled_result.json`.

| component | `S_within` (floor) | `S_pooled` (cost of reuse) | pooled amplitude | pooled part explains |
|---|---|---|---|---|
| **full field** (primary) | 0.2498 px | 0.2635 px | 0.3934 px | **33 %** |
| isotropic (fold 0) | 0.0737 px | 0.1340 px | 0.3576 px | 63 % |
| folds ≥ 3 | 0.1195 px | 0.1294 px | 0.0571 px | — |

n = 28 different-scan within-beamtime pairs, 26 beamtimes.

**Verdict: INCONCLUSIVE, not the CONFIRM the table literally scores.** The
criterion `S_pooled ≤ 1.5 × S_within` fires (ratio 1.05) — but only through its
"no better is achievable" clause, with **both** terms ~2.5× the 0.1 px that
matters physically. The preregistration's §7 vacuity check is what catches it:
the pooled component explains 33 % of the full field, not the 81–88 % that a
refuter of the earlier drift study had assumed. And the fold≥3 "CONFIRM" is
vacuous in the plainest way — its pooled amplitude, 0.057 px, is **smaller than
the scatter about it**, 0.129 px. There is no field there to reuse.

**The real, reportable finding is the floor itself.** One powder calibration
determines the ge5 distortion field to only about **0.25 px RMS**, against a
field of amplitude 0.39 px. That is the measurement precision of the method as
run — and at 100 µε ≈ 0.075 px, it is roughly 3× the level anyone cares about.
Reuse-versus-recalibrate cannot be settled until that floor comes down;
averaging several frames per beamtime is the obvious first attempt.

**Not claimed:** that the detector is stable over time (that was the earlier
drift study, REFUTED — an inconclusive reported as a finding); that fold 0 is a
detector property at all (it absorbs a median 0.38 mm of `Lsd`, ~25σ); anything
about ge1–ge4, which have no valid null — 0 of 30 beamtimes carry two
ring-verified frames from different scans.

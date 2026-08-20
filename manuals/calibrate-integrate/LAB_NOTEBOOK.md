# Lab notebook — evidence, measurements, and retractions

The spine says what to do. This says **why**, and records what had to be
withdrawn. Read this before re-investigating anything here: several attractive
hypotheses are recorded as refuted, with the measurement that killed each.

All work 2026-08-18/19. Dataset: 48-panel Pilatus (1475 × 1679, 172 µm), APS
20-ID, CeO2 at 63 keV, frame `ceria_..._000868.tiff`. Scripts and raw outputs
are archived with the release handoff for this work — the paths live there
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

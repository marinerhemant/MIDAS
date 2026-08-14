# DFXM Lab Notebook — ESRF ID03 / ID06 real data + the `midas_dfxm` framework

**Companion to `README.md` and the phase files.** The handbook says what to do; this
records what was actually found, how it was measured, and what turned out to be wrong. They
are kept apart on purpose: the handbook has to stay short enough to follow, and this has to
stay honest enough to stop a refuted idea coming back.

Sources throughout: the archived **ESRF ID03** aluminium mosaicity sets (via the `darling`
package) and the **ESRF ID06-HXM** multi-Bragg aluminium set (IH-HC-3803, DOI
10.15151/ESRF-ES-912179821); the `midas_dfxm` simulation-and-inverse framework; and the
paper `packages/midas_dfxm/dev/paper/P_merged/`. Scripts named below live under
`packages/midas_dfxm/dev/paper/runs/`. `§n` without a qualifier is a section of *this* file.

**Read §5 before re-opening any question here** — **twelve** claims are recorded there as
retracted, each with the measurement or reference that killed it.

> **This notebook is seeded from two campaigns, neither of them our own beamtime.** Findings
> §1–§2 are **real-data-proven**; §3–§4 are **verified against the dynamical forward or by
> cross-model test** but on synthetic/model data; **§7 is a re-analysis of an archived public
> deposit reduced by another group's pipeline** — real-data where marked, and the source of
> rules 11–20; §5 are retractions. When a real DFXM beamtime of our own runs, add its
> findings and mark them real-data.

---

## 1. What this campaign established

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | On raw ID03 frames the naive first moment is **~67× too small** — the pedestal carries **98.5 %** of the centroid weight | REAL-DATA, RESOLVED | §1a |
| 2 | Background-subtracted, `midas_dfxm` reproduces the `darling` reduction to **correlation 1.0, RMS ~1e-7°** | REAL-DATA, VERIFIED | §1b |
| 3 | Physical accuracy on ground-truth-free scans is established by **injection-recovery**: gain **0.9998–1.0000** on all four ID03 scans | REAL-DATA, VERIFIED | §1c |
| 4 | Per-pixel orientation precision ~**2 mdeg** (Poisson MC ratio 0.97), ~20–40× finer than the 80 mdeg step | REAL-DATA | §1c |
| 5 | Multi-reflection full-F is gated by **inter-reflection registration**, not photon statistics — ID06 111↔002 could not be fused | REAL-DATA, VERIFIED (negative) | §2 |
| 6 | Refraction is an **absolute strain-scale gauge (144 µε for Cu 002)**, not a per-pixel bias; recovered to 0.03 µε on a relative map | VERIFIED vs dynamical forward | §3 |
| 7 | The kinematic strain/defect inverse holds to **~0.3 Λ**, then biases; orientation is centroid-exact by symmetry | CROSS-MODEL VERIFIED | §4 |
| 8 | Full 2D Bragg (Borrmann-fan) solver needs a **semi-Lagrangian characteristic** scheme; Cartesian upwind is unstable | VERIFIED (6 tests) | §4c |
| 9 | "World's-first differentiable chromatic X-ray microscope simulator" | **RETRACTED** — prior art | §5a |
| 10 | Refraction "shifts every strain map, not maskable" | **RETRACTED** → gauge (§3) | §5b |
| 11 | A published rocking FWHM is the **integrated** width; the per-pixel median is **2.6–2.7× narrower** | REAL-DATA, VERIFIED (2 reimplementations) | §7a |
| 12 | Detector gain is not 1 on an integrating sCMOS (`var = 2.23·y + 149`) — absolute χ²/dof inflated ~2.2× | REAL-DATA, VERIFIED | §7b |
| 13 | A morphological-opening background removes a θ-dependent scalar (r = +0.92…+0.97 with the rocking curve), biasing per-pixel widths | REAL-DATA | §7c |
| 14 | One upstream preprocessing constant moved **~20 %** of a published segmentation's labels; discarded-signal asymmetry **15×** between the two channels | REAL-DATA, VERIFIED | §7d |
| 15 | A hardcoded magnification set every published length and disagrees with four independent optical routes by ~2× | REAL-DATA (archive metadata) | §7e |
| 16 | For vibration the discriminating datum is the **spectrum**, not the amplitude — short exposures do not recover low-frequency power | ANALYTIC, on a published vibration model | §7f |
| 17 | Λ ∝ 1/\|F\|, so weak superlattice reflections sit 10²–10³× further from the dynamical boundary than their parent | ANALYTIC, from deposited structures | §7g |
| 18 | A rocking curve cannot separate dynamical from kinematical diffraction when mosaic spread ≫ Darwin width (62× on public ID03 Al) | REAL-DATA (public), VERIFIED (negative) | §7h |
| 19 | An intensity ratio between two channels of **commensurate** superstructures is not a phase fraction | ANALYTIC (structure factors), real-data checked | §7i |
| 20 | "First inversion of DFXM contrast in the dynamical regime" | **SOFTENED** — proof-of-concept | §5c |
| 21 | CPFEM "coupled to the diffraction observables, not pre-processed maps" | **RETRACTED** — false as built | §5d |
| 22 | "Verified against an independent dislocation-dynamics code" | **SOFTENED** — round-trip | §5e |
| 23 | Reflection-set identifiability / oblique-geometry design presented as new | **RETRACTED** — prior art | §5f |
| 24 | An apparent quantisation of a rocking-angle map into discrete levels | **RETRACTED** — a detrending choice; spacing swings 10.4× | §5g |
| 25 | "A boundary network carries no orientation step, −3.2 σ" | **RETRACTED** — iid bootstrap on autocorrelated pixels; true −0.73, p = 0.47 | §5h |
| 26 | "The centre-of-mass map is robust to lineshape (0.045 mdeg)" | **RETRACTED** — 44× low; the alternative was symmetric | §5i |
| 27 | "A lineshape correction removes the window-truncation bias (39.5 % of labels)" | **RETRACTED** — failed its own positive control | §5j |
| 28 | The "Darwin ladder" — rocking-width/Darwin-width as a dynamical-diagnosability criterion | **RETRACTED** — four independent errors | §5k |
| 29 | An apparent two-population split, and sub-resolution rocking-curve doublets | **RETRACTED** — undersampling + a fixed rocking window | §5l |
| 30 | Reproduced a published DFXM shear-strain wave from public data (2.0 µm, diagonal), null-clean, rotating ~90° between reflections | REAL-DATA (public), VERIFIED (4-lens) | §8a |
| 31 | **Identifiability limit:** single-peak ε_xy + in-plane compatibility does NOT fix the dilatation (1 eqn, 3 unknowns; range 0→upper bound) | ESTABLISHED (/verify) | §8b |
| 32 | Coherence-R / band-SNR scalars do NOT collapse on a non-periodic null; autocorrelation periodicity + reflection rotation do | REAL-DATA (public) | §8c |
| 33 | "Compatibility *demands* a ~0.8× dilatation companion to the measured shear wave" | **RETRACTED** as worded — a closure choice (§8b) | §8b |
| 34 | A single reflection senses a strain as d-spacing (θ,2θ) OR tilt (θ-rock) by geometry; diagonal→strain, cube-axis→tilt (blind in θ,2θ) | PACKAGE-GT VERIFIED | §9a |
| 35 | A single DFXM detector frame is an inclined projection that collapses sample x and z onto one axis — it cannot separate a 2D from a 3D wave; needs a scanned reconstruction | PACKAGE-GT VERIFIED | §9c |
| 36 | "An a–c-plane DFXM image is a clean 2D-vs-3D dimensionality test" | **RETRACTED** — the projection collapses x and z (§9c) | §9c |
| 37 | midas `aligned_resolution` is transverse-**isotropic** (one σ_perp); Poulsen/Carlsen have σ_rock ≠ σ_roll | PACKAGE GAP | §10 |
| 38 | Carlsen's DFXM thesis independently confirms our verified physics (Δg=−Hᵀg₀, strain/tilt channel, inclined projection, kinematic∼deformation-bound) | PRIOR-ART, CONFIRMING | §10 |
| 39 | midas leads on differentiable **inverses**; Carlsen leads on the 3D wave-optics **forward** + **Fourier ptychography** (a midas gap) | PRIOR-ART COMPARISON | §10 |

The through-line: on real DFXM frames, **nothing tells you the answer is wrong** — a
pedestal-dominated map is smooth, an unregistered tensor is plausible, a refraction offset
is a clean uniform strain. Every finding above was forced by an explicit control, not by
the result looking off.

**The second campaign (§7, and retractions §5f–§5l) adds a second through-line: when you
re-analyse someone else's reduction, the boring cause is usually in the toolchain — theirs
or yours.** Of the claims that campaign retracted, none died of new physics. They died of a
detrending choice, an iid bootstrap on correlated pixels, a symmetric alternative that could
not move a centroid, a Gaussian fit on 7 points, a hardcoded constant, a wrong FWHM
convention, and prior art. Run the mundane checks first (README rules 11–20).

Detailed entries follow; the numbered sections match the "Where" column above.

---

## 1a. Real-data: the pedestal dominates the raw first moment (~67×)

Symptom: reducing raw ESRF ID03 frames to a per-pixel orientation map by centre-of-mass
gave a map with the right *shape* but an intragranular misorientation amplitude ~67× too
small.

Measured cause. The detector pedestal (dark + scattered background) is a large positive
floor under every pixel. The first moment $\langle\theta\rangle=\sum I_i\theta_i/\sum I_i$
is a weighted mean; a uniform pedestal $p$ under the signal pulls the weights toward the
frame centre and dilutes the true centroid excursion. On these frames the pedestal carried
**98.5 %** of the centroid weight, so $\langle\theta\rangle_{\text{raw}}$ underreported the
true shift by $1/(1-0.985)\approx67\times$.

Fix. Subtract `darling`'s own scalar background before the moment. On background-subtracted
frames the two pipelines then agree to **correlation 1.0, RMS ~1e-7°**. Do not take a first
moment on un-subtracted frames — the error is silent (the map is smooth and plausible).
Scripts: `make_real_multibragg.py` / `extract_com.py` (ID06); the P_single real-frame
reduction (ID03).

## 1b. Real-data: reproducing `darling` is an arithmetic check, not a physics check

Reproducing the community `darling` moment reduction on the identical frames confirms the
**arithmetic** of the estimator — it does not test the diffraction physics, the geometry,
or any inverse. Quote it as "same-estimator agreement," never as validation of the method.
The four ID03 scans (a 111 mosaicity scan of 5 %-deformed Al in two layers, a
domain-structure scan, a motor-drift scan) all passed a fixed criterion: correlation
> 0.999, RMS < 1 % of step.

## 1c. Real-data: injection-recovery is the physical-accuracy test

Because the public scans carry no ground truth, physical accuracy is established by
**injection-recovery**: a known orientation shift is resampled into the *measured raw
frames* and recovered against the real noise, background and detector. Recovery gain
**0.9998–1.0000** on all four scans. Counting statistics propagated analytically and
validated by Poisson Monte-Carlo (ratio ≈ 0.97) give per-pixel σ ≈ 2 mdeg, ~20–40× finer
than the 80 mdeg step. **A forward/inverse round-trip (1e-16) is NOT this test** — it
inverts its own generator (§5, and rule 5).

---

## 2. Real-data: the inter-reflection registration wall (ID06 111↔002)

The strongest real-data finding of the campaign, and a **verified negative**.

Setup. ESRF ID06-HXM multi-Bragg Al (IH-HC-3803): two reflections, 111 and 002, each
reduced to a clean per-pixel orientation field (`make_real_multibragg.py`). 111 intragranular
spread p95 ≈ 45 mdeg; 002 ≈ 6 mdeg — each a precise, real single-reflection map.

The wall. To recover a deformation-gradient tensor you must co-register the reflections
voxel-for-voxel. But 111 and 002 diffract at **different 2θ (67.5° vs 14.2°)** → different
objective magnification and field of view (frame shapes 142×215 vs 189×186). A search over
scale + shift for the best intensity cross-correlation reached only **NCC 0.43 at the search
edge** (−0.30 native), and the public deposit carried **no co-registration metadata**. The
two maps cannot be fused into a tensor.

Conclusion. **Inter-reflection registration, not photon statistics, is the binding
systematic for full-tensor DFXM.** Each reflection is individually excellent; the tensor is
unrecoverable without fiducials or a shared sample frame. This is a property of the
experiment to report, not a parameter to tune (rule 4). The differentiable substrate can
*self-register* when the field is structured (carry the per-reflection shifts as parameters,
minimise multi-reflection consistency — removes about half; the rest is set by a ~45 µε/px
drift×gradient prefactor), but that is a modelling aid, not a substitute for the metadata.

**Do not** report a fused ID06 tensor. If asked, halt (README STOP table) and show the wall.

---

## 3. Refraction: an absolute gauge, not a per-pixel bias (a mid-campaign correction)

This began as an overclaim and was corrected against the package's own dynamical forward —
the cleanest example in the campaign of the boring, honest explanation winning.

The number. The mean refraction shifts the dynamical Bragg peak by
$\Delta\theta=-\chi_{0r}/\sin2\theta_B$; a kinematic reader (peak = exact Bragg) misassigns
it as a strain $\varepsilon_{\mathrm{ref}}=\chi_{0r}/(2\sin^2\theta_B)$ = **144 µε for Cu
002 at 0.71 Å**, thickness-independent, matching the closed form to 0.1%
(`tt_kinematic_bias.py`; isolating χ₀ on/off control).

The correction. For a fixed reflection and energy this shift is **constant across the
grain**. On a *relative* intragranular strain map it is therefore a **gauge** — absorbed
into the reference lattice, recovered to **0.03 µε** after removing the offset
(`dynamical_sensitivity.py`, injected 30 µε test). It becomes a genuine per-pixel bias
**only where refraction varies** — across a thickness or perfection gradient — aliasing
into strain at transfer slope ~0.43 (= √3/4 for a linear ramp). That residual lives in the
near-perfect matrix where dark-field contrast is weakest, so it is **bounded, localisable
and maskable**.

Why it matters operationally. (i) It sets the **absolute** strain scale: any cross-reflection
or absolute-d strain must apply it (the classical Hart-1988 refractive-index correction).
(ii) Do **not** subtract it as an intragranular field — that removes a real reference and
leaves the gradient part in. See §5b for the retracted framing.

Extinction (a sibling effect) is a *symmetric* reshaping of the rocking curve, so a centroid
is invariant to it (< 1e-4 mdeg): extinction does not bias the recovered orientation even
when its strength varies spatially.

---

## 4. The kinematic validity boundary at ~0.3 Λ (cross-model verified)

### 4a. Orientation is centroid-exact by symmetry

In symmetric-Laue geometry the rocking curve is **even** in the deviation, and absorption
preserves that evenness (symmetric Borrmann transmission), so the per-pixel centroid is
**identically zero-biased** — orientation is exact, not merely approximately so. This is why
the kinematic read has served DFXM: it is exact exactly where DFXM has orientation signal.
The only residual is in **asymmetric/oblique Bragg** geometry, where absorption skews the
Darwin curve and shifts the centroid by ~**0.16 mdeg** (`tt_kinematic_bias.py`) — small, but
the leading correction in the oblique full-tensor geometry now being proposed.

### 4b. Strain/defect inverse holds to ~0.3 Λ, then biases

Generating dynamical DFXM contrast for a dislocation across thicknesses and inverting with
each model: the kinematic fit sits at the noise floor only for **t ≲ 0.15 Λ**, has clearly
left it by **0.3 Λ**, and by **t = 1.1 Λ** biases the recovered amplitude by **+38 %**, then
rails unphysical — while the dynamical inverse stays at truth throughout (`tt_kin_vs_dyn.py`;
pre-registered).

**Cross-model confirmation (not inverse crime).** Data generated by the *dynamical*
Takagi–Taupin forward, inverted by the *kinematic* model, with the inverse-crime
(kinematic-generate) case as baseline (`cross_model_test.py`, Λ = 71.8 µm): recovered to
**0.4 %** for t ≲ 0.15 Λ; cross-model error first leaves the noise floor at **t/Λ ≈ 0.3**;
**7–35 %** amplitude bias by t ≳ 0.5 Λ; the correct dynamical inverse stays at truth
(0.995–1.004). The geometric full-F inverse round-trips **clean** kinematic data to
**4.67e-20** (rank-9, cond 4.69) — so the breakdown at 0.3 Λ is **physics, not linear
algebra**. Note the `penalty` column in that script's table is a small-denominator artifact;
the robust readouts are the A_cross bias and kin_misfit.

**Provisional caveat (do not upgrade):** the cross-model test is a single geometry, one
screw dislocation, non-absorbing χ, y = 0 imaging condition. Consistent with the validated
`tt_kin_vs_dyn`, but not a claim across all geometries.

### 4c. The 2D Bragg (Borrmann-fan) solver needed a characteristic scheme

A naive Cartesian ∂/∂x advection added to the depth march is numerically **unstable** off
the total-reflection plateau (it diverges under lateral-grid refinement, 1e6–1e18). The
stable method is **semi-Lagrangian**: integrate each beam along its own characteristic
(D₀ into the crystal, Dₕ back out) with an exact integer lateral shift per depth step and a
Crank–Nicolson reaction. Energy is conserved only with an **integer** shift (linear
interpolation diffuses; refl_frac drifted with dx until the integer-shift fix → R = 1.0000).
The n_x = 1 collapse reproduces the 1D Riccati Darwin curve to ~7e-3 (grid-limited, **not**
machine precision — do not claim exact reduction). 6 tests in
`tests/test_takagi_taupin_2d.py`.

---

## 5. Retractions — read before re-opening

### 5a. "World's-first differentiable chromatic X-ray microscope simulator" — RETRACTED

An external suggestion framed the pink-beam chromatic forward as a world-first. An
adversarial literature search refuted it: **La Bella 2025**, **Carlsen 2022**, **Thies
2022**, and the differentiable-optics line (**Sitzmann/Tseng**, **Wu/Zhou**) all precede or
overlap it. The pink-beam chromatic subsection was **cut** from the paper; the physics
(chromatic PSF, geometric Bragg shift, μ∝λ³, Λ∝1/λ, objective f∝1/λ²) is kept in the
package (`chromatic.py`, `pink.py`) but claims no primacy. Lesson: run the novelty search
*before* writing the claim (the `verify` skill).

### 5b. Refraction "shifts every strain map, not maskable" — RETRACTED → gauge

The paper's main text once called the 144 µε refraction "the one bias that shifts every
strain map… not localised and not maskable." The package's own SI said the opposite (a
reference gauge, recovered to 0.03 µε, maskable). The **SI was right**; the main text was an
overclaim and was rewritten (§3). Lesson: a uniform offset on a *relative* map is a gauge —
check whether the quantity is relative or absolute before calling an offset a bias.

### 5c. "First inversion of DFXM contrast in the dynamical regime" — SOFTENED

A 2-parameter (core, amplitude) fit on matched-model data is a **proof-of-concept**
parametric inversion, not a demonstrated general capability. Free per-voxel dynamical field
inversion is a different, much harder problem and remains future work.

### 5d. CPFEM "coupled to the diffraction observables, not pre-processed maps" — FALSE as built

The CPFEM adjoint objective fits a *reconstructed elastic-strain field*, i.e. a pre-processed
map — so the "observables not maps" differentiator was false. Reworded to a proof-of-concept
that puts crystal plasticity and the DFXM observable on one graph, so the adjoint *can in
principle* be driven by the data.

### 5e. "Verified against an independent dislocation-dynamics code" — SOFTENED

The ExaDiS DDD network supplied a displacement *field*, which was round-tripped through our
own forward — a **software-consistency** check, not an independent physical validation. Do
not call a round-trip on a supplied field a cross-model verification.

### 5f. Reflection-set identifiability and oblique geometry, presented as new — RETRACTED

We derived, and were about to recommend, a reflection-set/rocking-axis design for improving
the identifiability of the full deformation-gradient tensor. **It is published.** Two
references settle it, and both are **cited in `midas_dfxm`'s own docstrings** —
`deformation_identifiability` and `fisher_information` — which were read while writing the
script that "derived" it:

- **Detlefs et al., J. Appl. Cryst. (2025, in press)**, "Oblique diffraction geometry for the
  observation of several non-coplanar Bragg reflections under identical illumination": ≥ 3
  non-coplanar **symmetry-equivalent** reflections hold the illuminated volume constant. It is
  strictly better than what we proposed — it reaches **rank 9** where an in-plane rock caps at
  6, and symmetry-equivalence keeps |F| identical, which disposes of the intensity problem
  that had already killed two earlier versions of our advice.
- **Kanesalingam et al. (2025)**: the inverse formalism for the full deformation-gradient
  tensor, plus a κ(ν) sensitivity metric for choosing a reflection set.

What survives is a bound worth knowing, and it is why the oblique geometry is *necessary*
rather than merely better: each θ-rocking sensitivity row is an outer product
$\hat{\mathbf Q}\otimes\mathbf v$, so with $\hat{\mathbf Q}$ confined to a plane the row space
lies in (2-D)×(3-D) and the **rank ceiling is 6 for any rotation axis** — verified against 500
random axes. Rank 9 requires $\hat{\mathbf Q}$ out of the plane, i.e. the oblique geometry, not
a cleverer rock.

**Lesson, and it is the fourth prior-art failure in that campaign: the novelty search must
include your own package's docstrings and your own literature folder.** Both of these
citations were inside the tool being used. Two of the other three were found the same way. Run
the gate *before* writing the claim (the `verify` skill), and grep the target publication for
the terms your claim uses before calling anything new.

### 5g. An apparent quantisation of a rocking-angle map into discrete levels — RETRACTED

A rocking-angle map appeared to be quantised into discrete levels whose spacing matched a
parameter-free crystallographic prediction. **It was a detrending choice.** Detrend order 1
gives 3.55 mdeg, order 2 gives 2.3–3.8, order 3 gives 0.7–2.1 — a **10.4× swing** across
admissible orders (0.70 → 7.29 mdeg), and the map's own standard deviation falls
7.89 → 4.96 → 4.72 → 2.33 mdeg with order. **Most of the variation across that field is a
smooth gradient, not discrete levels**, so the initial agreement came from an un-detrended map
and any agreement at all was an artifact of the choice in one direction or the other. The
discriminating check is trivial and was skipped: report the statistic across every admissible
detrend order before quoting it.

Two more errors in the same item, both worth their keep:

- **Every sign was inverted**, caught four independent ways. The prose had dropped a minus
  sign; the script that "reproduced" it carried a 30° frame rotation, because
  `reciprocal_basis` is Busing–Levy so the real-space axis sits at −30°, and **for a traceless
  in-plane deviator 30° is exactly a sign flip**. Two errors cancelled and were logged as a
  reproduction (rule 18).
- **The prediction was not a measurement.** It was read off a refined lattice metric from a
  structure whose refinement models three twin domains at near-equal fractions — which raises
  the apparent symmetry and averages the single-domain distortion out. A second deposited
  structure of the same material gives the opposite sign and 1.9× the magnitude. A twin-averaged
  metric cannot supply a single-domain distortion; the prediction spans a factor ~2 and is not
  sign-definite.

### 5h. "A boundary network carries no orientation step (−3.2 σ)" — RETRACTED

The withdrawal stands but **every stated reason for it was wrong**, which is why it is here.

- **The −3.2 σ does not exist.** It came from an iid bootstrap over 6,812 skeleton pixels in a
  field autocorrelated to **0.90 at 48 px**. A block bootstrap gives σ 0.076 and a
  phase-randomised surrogate σ 0.082 (n_eff ≈ 172): the true value is **−0.73, p = 0.47**
  (rule 19).
- **The test had no power anyway.** Planting the predicted network on the *actual* detected
  boundaries and blurring by the measured resolution returns z = +0.98, and a forward model of
  a fully-structured crystal is indistinguishable from an unstructured control at this map's
  resolution.
- **Matching on intensity alone is not matching.** Adding a |∇I| match flips the sign
  (ratio 0.938 → 1.022).
- The replacement statement — that the feature was spatially *resolved* and therefore could not
  be a sharp wall — was **circular**: the erf width tracked its own fit window (0.95 → 1.53 µm)
  and the pipeline resolution floor alone spans 0.62–1.08 µm.

**Status: genuinely unresolved in either direction**, not "consistent with" anything. Recorded
so it is not re-opened with the same tooling.

### 5i. "The centre-of-mass map is robust to lineshape" — RETRACTED, 44× low

Reported as robust at **0.045 mdeg**; the real figure against an adequate lineshape is a median
**1.98 mdeg (p95 10.67)**, 12 % of the FWHM. The cause is a single design error: the robustness
was tested against a **symmetric** pseudo-Voigt. A symmetric alternative shares a symmetry with
the Gaussian and is therefore blind to skew — **the one misspecification that moves a centre**.
The curves were strongly skewed (per-pixel |asymmetry| median 0.227 against a null of
0.015–0.028) and a **split** pseudo-Voigt fits better than both (χ²/dof 2.83 vs 4.03 vs 6.09).
This was the reassuring half of the claim and it was the wrong half (rule 20).

### 5j. "A lineshape correction removes the window-truncation bias" — RETRACTED by its own control

Correcting each pixel's truncated rocking integral by fitting and extrapolating moved **39.5 %**
of the segmentation labels. The number is void:

- **The positive control failed.** Plant a known truncation at positions where both peaks are
  well centred: it moves **6.4 %** of labels. Applying the correction leaves a **24.2 %**
  residual — **3.8× worse than the bias it was meant to remove** (recovery −276 %).
- **The churn does not track truncation:** 39.1 % on well-bracketed positions versus 40.4 % on
  truncated ones. If truncation drove it, the first would be near zero.
- Cause is mundane: a 4-parameter Gaussian on 7 points with the peak sometimes near an edge is
  ill-posed, and the real lineshape has power-law tails, so the model is wrong even where the
  fit converges. The negative control passed exactly (identical inputs → 0.0000 churn), so the
  machinery was fine; the correction specifically was not.

**One causal number survives, because the positive control is itself an assumption-free
measurement: truncating the narrow channel by two frames moves 6.4 % of the labels** — a lower
bound on what truncation can do, not an estimate of what it did.

This was the first candidate in that campaign killed by its own built-in control, before any
external review and before reaching a user-facing document. The control cost about twenty
lines. **Every earlier failure would have been caught the same way** (rule 17).

### 5k. The "Darwin ladder" — RETRACTED (four independent errors)

The claim was that no open DFXM crystal reaches the Darwin limit, so dynamical diffraction is
everywhere a mere shape perturbation, ranked by rocking-width/Darwin-width. Retracted the same
day by three refuter lenses; all four errors are general:

1. **Non sequitur.** Whether dynamical effects matter is set by **t_coherent/Λ**, not by
   rocking-width/Darwin-width — mosaic spread and coherent block size are independent (§4b, and
   §7h). A wide rocking curve does not bound dynamical effects; that is what extinction
   corrections are for.
2. **Wrong FWHM convention.** Global outermost half-max crossings instead of argmax-local
   contiguous ones inflated everything (one case 59.81 → 12.00 mdeg). One candidate's
   above-half-max index set was **non-contiguous**, so the reported width spanned the gap
   between two disjoint islands. The corrected verdict *flipped*.
3. **Material identity unsupported.** The lattice parameter was hardcoded and the recorded
   crystal-pitch metadata was never read; a different element fits the recorded pitch **better**
   (+0.12 % vs +0.30 %), and the ratio spans 10–37× across candidates. The sample was named only
   `fatigue_test`; the file carries no element or energy field, and λ was itself assumed.
4. **A mundane mechanism accounts for the whole width**: condenser divergence alone gives 17 mdeg
   against a 12 mdeg measured width. No instrument budget had been done.

**What survives: nothing general.** An interim retraction note claiming "≥ 10× under every
material hypothesis" was also wrong and is withdrawn. Call that quantity an angular *acceptance*
width, not a mosaic spread. Resolving it needs the material and the condenser divergence — absent
beamline metadata, i.e. a question for the data's authors, not an analysis fix.

### 5l. An apparent two-population split, and sub-resolution rocking doublets — RETRACTED

Two related positives, both from undersampling and a fixed rocking window:

- **The bimodality coefficient `b = (skew² + 1)/kurtosis` does not measure bimodality on an
  undersampled rocking curve — it measures curve *broadness*.** At ~6 points per FWHM a flagged
  fraction doubled through a temperature window at matched SNR (13 σ) and was entirely artifact:
  flagged pixels are 50 % broader, `b` tracks truncation, and decisively **40 % of flagged
  pixels had ≥ 2 local maxima — and so did 40 % of unflagged pixels**. Do not use moment-based
  bimodality below ~12 pts/FWHM; use a two-component fit with a parametric-bootstrap null, and
  **gate on flagged-vs-unflagged enrichment**, never the statistic's own p-value. A later revival
  attempt on a better-sampled set returned enrichment **0.96×**.
- **A split fraction that looked bimodal at +20.7 σ was θ-window truncation.** Each group reused
  **one fixed rocking window at every raster point** while θ_B drifted across the raster, and the
  two channels had ~2× different widths, so the narrower one left the window far faster. The
  statistic was a map of *which scans caught their peak*: mean split fraction 0.698 in captured
  versus 0.263 in truncated scans, capture failure falling in clean raster blocks, and restricting
  to positions where both peaks are interior takes the dip from **+10.9 σ to +0.0/−2.0 σ**. Each
  group's histogram is separately unimodal.
- Two instrument-level errors in the same item, both recurring: the "0.7423 dip" **cannot be
  Hartigan's dip, which is bounded by 0.25** (it was a KDE valley depth; true Hartigan's gives
  −1.5 σ), and a surrogate built with a single global mean/sd had **zero between-scan degrees of
  freedom** while 71.7 % of the variance was between scans. Check a statistic's admissible range,
  and build a null with the same hierarchy as the data.

---

## 6. Open threads — read together with §7

- No real-data validation of any **capability** inverse (typing, defect model, full-F on a
  real field) — all are simulation-grounded. The single biggest lever for the paper and for
  this doc set is one capability demonstrated on real DFXM frames.
- **APS 6-ID-C DFXM is gated** — a different instrument, and a bilateral collaboration with
  its own credit rules. Confirm before touching that data. Its full instrument geometry and
  its vibration/resolution limits are not in this doc set; the transferable lessons from that
  work are in §7 and rules 11–20, stated instrument-independently.
- The self-registration aid (§2) is demonstrated only where the field is structured; its
  failure modes on real drift are uncharacterised.
- **Resolution recovery on an archived scan** (within-frame vibration blur vs between-frame
  drift) was pre-registered but never run; §7f says what datum it needs and why an archive
  cannot supply it.
- **Do not re-open with the same tooling:** the boundary-orientation question (§5h) is
  unresolved in either direction, and sub-resolution doublets (§5l) are not to be revisited
  without new evidence at ≳ 12 pts/FWHM.

---

## 7. Second campaign: re-analysing an archived deposit reduced by another pipeline

**What this campaign was.** A public archive of DFXM frames, plus the producing group's own
reduction scripts and their publication, re-analysed to see whether more could be extracted.
It produced **four** results that survived adversarial verification, and retracted far more
than that (§5f–§5l). Its lasting value is the list of ways a re-analysis goes wrong quietly,
which is what rules 11–20 encode.

**The anonymisation note that matters for reading this:** the material, the instrument
configuration, the scientific claim and the collaborator are deliberately absent. Every entry
below is stated so it is usable by someone who never sees that data — if a lesson needed the
sample's identity to be useful, it is not here.

### 7a. Real-data: a published rocking FWHM is the integrated width (2.6–2.7×)

The integrated (whole-image) rocking curve is broadened by mosaic spread *across* pixels; a
per-pixel fit sees the narrower per-pixel curve. Measured model-free on one archived scan at
1.000 mdeg steps: integrated FWHM **26.0–28.3 mdeg** depending on convention, per-pixel median
**10.0 mdeg** (5th pct 4, 95th pct 24) — ratio **2.6–2.7×**, stable across every convention
tried and confirmed by two independent reimplementations.

Consequence: dividing the step into the *published* width gave "24 points per FWHM" where the
number governing a per-pixel fit is **10**. That **invalidated the premise of an entire
preregistration** (a model-selection test needing ≳ 12 pts/FWHM). Before claiming a scan is
well sampled for any per-pixel model-selection test, measure the per-pixel width on the actual
frames — never infer it from an integrated or published FWHM.

**Convention, and it bit twice (see §5k):** use **argmax-local, contiguous** half-max
crossings, and check contiguity. Global outermost crossings let one noise spike set the width
(an earlier version of this entry read 95th pct 151 mdeg for that reason; the real value is
24). And do not cite a width tail as evidence of non-Gaussianity — measure χ²/dof, which is
the real evidence.

### 7b. Real-data: the detector gain was not 1, and it recalibrated everything

Measured by photon transfer (variance of nearest-neighbour differences against local mean) on
an **integrating sCMOS**: `var = 2.23·y + 149`, not `var = y`. **Every absolute χ²/dof computed
with `var = counts` was inflated ~2.2×.** With the correct gain, a Gaussian-plus-broad-component
model reaches χ²/dof **1.075** — an adequate model existed all along — while the Gaussian alone
still fails (~2.8). So "these curves are not Gaussian" survived, but every statement of the form
"even the better model is rejected" was wrong.

Three operational points:

- **Remove the pedestal before measuring var/mean.** A pedestal makes the estimate invalid and
  can drive it *below* 1, which is what happened when we first tried it.
- **Never carry one detector's gain onto another's frames.** Applying this sCMOS gain to
  photon-counting frames from the same campaign inflated σ ~2.2× and **flipped the sign of a
  headline result**. The photon-counting detector measures var/mean = **1.0013–1.012** on
  background (and 80 % exact zeros matching e^−λ) — counting statistics, gain 1, as expected.
  Verify which kind of detector you have; do not infer it from the beamline.
- **Ratio statistics are safe.** Likelihood ratios rescale by the same factor and ROC/AUC is
  invariant under a monotone transform, so this bites absolute χ², error bars and "is my model
  adequate" — not relative comparisons.

### 7c. Real-data: a background that follows the signal biases widths, not centroids

A per-frame morphological-opening ("rolling ball") background removes a level that **tracks the
rocking curve**, so it distorts curve *shape*:

- On one ROI the auto-chosen kernel was 1160 px → a **145×145 structuring element on a 70×126
  downsampled grid, larger than the image**. The "spatial" background collapses to a
  θ-dependent **scalar** (within-frame std 1.82 counts) that is **92 % a copy of the rocking
  curve** (r = **+0.919**). There were **zero non-diffracting pixels** in that ROI, so no
  legitimate common-mode reference existed. Switched to dark-only.
- On a larger ROI the same recipe was **not** degenerate — the kernel fitted the grid and the
  filter was genuinely spatial (within-frame std 4.09 counts) — **yet the per-frame level still
  correlated +0.966 with the rocking curve, swinging ~17 counts across θ.** So the milder
  problem is the general one and the degeneracy is only its extreme.

Consequence: any per-pixel **width/FWHM** computed after such a background carries a bias; a
**centroid is far less sensitive**, which is why the map looks fine. The recipe's original
injection-recovery validation tested *feature* recovery, not *shape* preservation, so this was
never covered by its own tests. Always check `kernel/down` against the downsampled ROI size,
and correlate the removed level against the rocking curve.

### 7d. Real-data: the upstream pipeline is where the boring causes live

Three findings, one lesson: **read the producing pipeline's scripts.**

- **A single preprocessing constant.** One column-clip constant zeroed a fixed block of
  detector rows before the aperture crop and the published two-channel comparison. That region
  carried **15× more** of one channel's in-aperture signal than of the other's (22.7 % vs
  1.5 %), the clip boundary fell **inside** the retained aperture at every raster position, and
  toggling it moves **~20 % of the commonly-labelled pixels**. Note what *else* this shows: our
  first statement of the bias **direction** was refuted — because the brighter channel's maximum
  lay inside the clipped band and the pipeline thresholded at a fraction of the frame maximum,
  the clip *protected* that channel. A sensitivity of ~20 % to one constant is the durable
  result; the sign needed a mechanism, and the mechanism was in their code.
- **No flux monitor.** No I0/monitor column existed anywhere in the motor metadata; the entire
  campaign's flux record was **three hand-typed readings at ±11 %** — while the central claim was
  an intensity comparison between two separately-acquired groups. Also: the **frame total is not
  a flux monitor on a rocking scan — it *is* the rocking curve**; use non-diffracting pixels
  (done that way on a public ID03 scan, flux was stable to 2.7 %).
- **The optics moved, not the sample.** An apparent inter-reflection centroid offset of +2 px
  (~4 µm) at a nominal 12.5 σ was **fully reproduced by a null with rigidly identical footprints
  and zero displacement**. The objective had moved 45 µm in x and 10 µm in z between groups —
  logged in the same motor file whose *sample* columns had been checked — and at that
  magnification a 10 µm objective move shifts the image by more than the whole "effect".
  Diagnostic: r(offset, log counts) = +0.92…+0.97, and the partial correlation with position
  given counts goes **negative**. Two further traps in that one analysis: an intensity centroid
  on a mostly-pedestal frame **measures the pedestal**, and a correlation with a raster's **fast
  axis** is not separable from a per-cycle mechanical effect (sample y was collinear with
  within-cycle phase to r = 1.000000; dropping the two scans after each fly-back took r(y) from
  +0.51 to −0.13 — it was stage settling).
- **The correction was already there.** Before reporting that pipeline as missing a registration
  correction, we read it: it re-centred every scan on its own fitted peak, residual
  −0.094 ± 0.041 px. The claim would have been wrong and would have gone to a collaborator.

### 7e. Real-data: the µm/px scale rested on a hardcoded constant

A magnification constant hardcoded in every analysis script set **every length in the
publication**, and disagreed with **four independent optical routes** by about a factor of two.
Worth noting the mechanism, because it is the common one: a factor of exactly 2 is what you get
if a calibration counts a **line pair** as one feature.

Two general corollaries, both computed rather than asserted:

- A refractive objective specified at one energy has **f ∝ E²**. Used well above that energy, f
  can **exceed the working distance** — object inside focus, image virtual — so the geometry is
  not what the nominal number implies. Check f(E) against the working distance actually used.
- **A focus diagnostic tells you the image is sharp, not that the scale is right.** These are
  separate questions and a published wavelet focus metric answers only the first. Do not let one
  stand in for the other.

Neither of these needs the sample. Both are checks on the optics record, and the second is the
kind of confusion that survives peer review.

### 7f. Vibration: the discriminating datum is the spectrum, not the amplitude

Measured resolution can sit well below an instrument's demonstrated best because of
sample/objective vibration in the ~1–100 Hz band, whose resolution penalty **saturates once the
exposure exceeds the slowest period**. Inverting a published degradation factor through its own
model, against a blur re-measured from raw frames, put the archive's exposures **fully in the
saturated regime**.

The useful result is a **negative, and it is general: short exposures alone do not recover it.**
Shortening the exposure by 50× still left a 1.60× penalty, and a stack of short frames failed a
modest recovery target — because **a 0.1 s exposure does not freeze a 1 Hz component**. Whether
short frames help depends entirely on where the power sits: 50–100 Hz yes, 1–5 Hz no.

So the discriminating measurement is the **frequency spectrum**, not an rms amplitude, and an
archive of images cannot supply it. Separately, **whole-ROI cross-correlation "registration" of
a rocking stack cannot fix within-frame blur** — it only addresses between-frame drift, and on a
rocking stack the content itself changes with θ. Halt and ask for a spectrum (README STOP table)
rather than proposing exposure changes.

### 7g. Λ ∝ 1/|F|: a weak satellite and its parent are in different regimes

The extinction length scales inversely with the structure-factor magnitude, so a **weak
superlattice/satellite reflection sits orders of magnitude further from the dynamical boundary
than its parent**. Computed from deposited structures for one archived case: **Λ ≈ 24 µm** for
the strong parent reflection but **7,000–10,000 µm** for the satellites, because the superlattice
|F| is ~400× smaller. Against coherent block sizes of ~0.5–1.5 µm, t/Λ runs 1e-4 to 0.06 versus
the ~0.3 bound (§4b): **the satellites can never need the dynamical forward**, while the parent
crosses the bound if coherent blocks reach tens of µm.

Two consequences worth carrying:

- Classify the regime **per reflection** (§1b), never once per sample. The same crystal can be
  safely kinematical in one channel and marginal in another.
- Do not use the **mosaic** width as the coherent block size when computing t/Λ — that
  substitution is precisely the retracted reasoning in §5k. They are independent quantities.

**Sampling asymmetry, and it is a structural blindness rather than an error.** The weak channel
is usually acquired at coarser sampling than the strong one: in that archive the satellites were
sampled **42× coarser in space and 100× coarser in angle** than the parent. Physics that lives
only in the weak channel is therefore invisible in the highest-resolution map — check this before
proposing to look for anything there.

### 7h. Real-data (public): a rocking curve cannot diagnose the dynamical regime

Measured on real public ESRF ID03 Al(111) at 17.0 keV (`darling`'s `rocking_scan_id03`, sample
`fatigue_test`, 150 pts @ 1 mdeg): rocking FWHM **59.81 mdeg** against a Darwin width of
**0.957 mdeg** — 62×, so the profile is set by mosaic spread plus instrument resolution and the
dynamical share of the variance is ~2.6e-2 %. The scan step (0.999 mdeg) **equals** the Darwin
width, so the dynamical feature is not resolved by even one step: no fitting recovers it, and any
dynamical-vs-geometrical fit gain is bought with model freedom.

This **bounds the domain of the dynamical forward rather than refuting it**, consistent with the
t ≲ 0.3 Λ result (§4b; Λ = 27.98 µm here). Deformed metals are "ideally imperfect", so
kinematical is *correct* there. Test the dynamical forward only on **near-perfect** crystals
(Ge, Si, HgCdTe, oxide films) with step ≲ 0.2 mdeg.

Also established here, and general: **whole-detector sums give fake rocking curves** — find the
ROI by temporal modulation first (a spurious "FWHM 0.773 mdeg, CONFIRMED" came from a
whole-detector sum).

### 7i. Commensurate superstructures share channels — a ratio is not a phase fraction

The one result of that campaign that survived every attack, stated generally.

**The crystallography.** If two candidate superstructures have commensurate periods along the
same axis — periods n·c and 2n·c — then the finer supercell produces reflections at every index
of the coarser one. A reflection assigned to "structure A" is then a **shared channel** that both
structures feed, while only the finer structure produces the intermediate indices. **Twinning
multiplies the problem:** a symmetry rotation permutes which modulation arm a given variant
contributes, so each variant delivers a *different* supercell reflection to the same lab-frame
**Q** — and systematic extinctions can forbid one variant entirely at the shared position.

**The consequence for segmentation, and this is the transferable part.** An intensity ratio
between such channels is treated as a two-state contrast and thresholded to get a phase
fraction. But if one channel is shared, the ratio for a region that is **100 % one phase** is
neither 0.5 nor constant — it depends on the local twin-variant populations. Computing |F|² per
variant from the deposited structures gave a **13.9× spread across variants**, and a
single-phase region lands inside the published "mixed/undecidable" band at the very twin
fractions that structure's own refinement reports. Where one variant is under-represented, a
single-phase region is actively **mislabelled as the other phase**.

**The discriminating test (and it can exonerate):** compute |F|² at the **measured Q** for every
candidate structure *and* every twin/domain variant. If exactly one structure contributes at
each channel's Q, the ratio *is* a phase fraction and none of this applies. If both contribute at
one of them, locate the ratio a single-phase region actually produces and compare it to your
threshold band.

**What this does not say.** It does not show that the phase separation is absent — an independent
measurement supported it, and our own attempt to test it statistically failed its controls
(§5l). It says the thresholded quantity is not a clean two-state contrast, so the threshold
cannot be read as a phase fraction. The fix is upstream of image processing: **a better feature
extractor applied to a mis-specified contrast still returns a mis-specified answer.** Replace the
hard deadband with a per-pixel class probability plus an explicit "undecidable at this dose"
class.

---

## 8. Third campaign: reproducing a published strain wave from public data

Source: **Yay et al., Sci. Adv. 12, eaec8998 (2026)**, public Dryad **doi:10.5061/dryad.rfj6q57pj**
(CC0) — Cu-Ba122 nematic, 040₁, 80 K, figure-level Δε_xy map (not raw frames). Scripts under
`~/Desktop/analysis/yay_strainwaves_dryad/` (`robustness.py`, `compat_dilatation.py`,
`honest_analysis.py`, `build_artifact.py`). This campaign both **confirmed a real effect** and
**caught our own overclaim via `/verify`** — the same pattern the whole notebook is about, this
time on our own analysis rather than someone else's.

### 8a. Real-data (public): the strain wave reproduces, and it is not an artifact

From the public Δε_xy map alone, an independent pipeline recovers a coherent shear-strain wave:
**λ = 2.0 ± 0.4 µm**, diagonal wavevector **ψ ≈ −48°**, direction coherence **R = 0.98**. Confirmed
three independent ways: our windowed power spectrum (1.98 µm), the authors' own published Fourier
transform (1.95 µm), and a real-space autocorrelation (side-lobes at ±2.0 µm). Strongest
is-it-physics signature: the wavevector **rotates ~94° between the 400₄ and 040₁ reflections** —
it tracks the crystal, not the lab, so a detector/optics/processing artifact fixed in the lab
frame is excluded. A four-lens `/verify` returned SURVIVES on the artifact, statistics, and
reproduction lenses.

### 8b. Identifiability limit + a retraction: single-peak strain + compatibility ≠ dilatation

We first claimed in-plane Saint-Venant compatibility **demands** a companion dilatation wave of
~0.8× the shear. `/verify` (physics lens) **REFUTED** it. 2D compatibility is **one equation in
three unknown strain components**, so a measured ε_xy fixes only one combination of the normal
strains. A purely **deviatoric** companion with **exactly zero dilatation** satisfies the same
compatibility with the same ε_xy (residual ~1e-10). So the compatible dilatation ranges from **0**
(all-deviatoric) to an **upper bound** (~1.5e-5 here, all-dilatation); the partition is fixed by the
elastic moduli / Ginzburg-Landau energetics, which single-peak data **cannot access**. The "0.8"
was a closure choice, and was mislabeled (a normal-strain *component* vs the dilatation *trace*,
~2×). This is now **README rule 21**. The closure-free, falsifiable residue — the honest deliverable
— is the source term a multi-peak measurement of ε_xx, ε_yy must satisfy:
`∂²ε_xx/∂y² + ∂²ε_yy/∂x² = 2 ∂²ε_xy/∂x∂y`. Logged: `/verify` REFUTED (closure) / ESTABLISHED (wave).
Lesson: a deterministic linear transform of a measurement is not an independent prediction of it.

### 8c. Null-discriminator: a scalar coherence/SNR does not tell a wave from a non-periodic phase

The pipeline's per-window band-SNR was **higher** on the tetragonal null (Fig3c, 220, 100 K:
SNR ~29) than on the real wave (SNR ~11), and the null's direction coherence R = 0.63 was not
negligible — a smooth but **non-periodic** strain field (large blobs) produces both. So a scalar
coherence/SNR gate does **not** discriminate wave from no-wave. What does: (i) **real-space
autocorrelation** — a wave has periodic side-lobes at ±λ, a non-periodic field decays
monotonically; (ii) **reflection rotation** — a real strain wave's wavevector rotates with the
crystal between reflections; an artifact does not. Lead with these, never with a coherence number.
See DIAGNOSIS ("A claimed strain wave must be told from a non-periodic phase").

---

## 9. Fourth campaign: forward-model ground truth vs a fast approximation

We built a fast browser helper for "what does this reflection/plane image?", then rendered the same
scenarios with the **validated `midas_dfxm` package forward model** (voxel-splat geometrical optics +
Poulsen resolution + objective PSF; scripts `~/Desktop/analysis/yay_strainwaves_dryad/gt_render.py`,
`forward_h0l.py`, `plane_simulator.html`). The package reproduced the planted wave exactly (2.00 µm in
2D; 2 µm × 3 µm in 3D, recovered from the arrays), and in doing so caught three things the fast
approximation got wrong — the same value a `/verify` pass gives, applied to our own tool. Same-team
public context: Yay et al. Sci Adv 2026 (Cu-Ba122); Ba122 cell inferred.

### 9a. A single reflection sees a strain as d-spacing OR tilt, by geometry

The rigorous shift is $\Delta\mathbf g=-\mathbf H^{\mathsf T}\mathbf g_0$ (from $\mathbf g=\mathbf F^{-\mathsf T}\mathbf g_0$),
with $\mathbf H$ the **symmetric** strain tensor here (the loose "$\Delta\mathbf Q=\mathbf H\mathbf G_0$"
is right only for a symmetric field — for an asymmetric distortion the transpose flips the tilt). It has
a longitudinal part (‖$\mathbf g_0$, d-spacing, θ,2θ) + transverse part (⊥$\mathbf g_0$, rotation,
θ-rock/mosaicity). For a shear $\varepsilon_{xy}$ a diagonal reflection gives pure longitudinal → strain,
a cube-axis reflection pure transverse → tilt — coefficients $c_{\mathrm{strain}}=2\hat g_x\hat g_y$ and
$c_{\mathrm{tilt}}=\sqrt{\hat g_x^2+\hat g_y^2-c_{\mathrm{strain}}^2}$, verified to machine precision by
two independent reproductions: 1.00/0.00 for $[110]_T$, 0.00/1.00 for $[100]/[010]_T$, **0.956/0.206**
(not 0.96) for the shallow h0l. So a cube-axis reflection is **blind to the shear in a θ,2θ scan**.
**HAZARD (the /verify strongest surviving issue): the Miller string is frame-dependent.** The paper's/
tool's ortho 400 ($a_O\!\parallel\![110]_T$) is strain-sensing; a literal tetragonal (400) is a cube axis
= tilt-sensing. Quoting "the 400 reflection" without the frame **inverts strain↔tilt**. Verified
ESTABLISHED across physics, reproduction, literature (Yay/Poulsen verbatim), and package/convention
lenses (no factor-of-2: the amplitude is the tensor shear, `small_strain_from_F` recovers it exactly).

### 9b. At exact Bragg the wave images at λ/2; use the weak-beam flank for true λ

A fixed-θ intensity image at the exact Bragg peak responds to the shift at **2nd order** (even), so the
periodic wave images at **λ/2** — the package showed 1.00 µm for a 2.00 µm wave at exact Bragg, and the
true 2.00 µm only when the acceptance centre was offset ~1σ onto the **weak-beam flank**. A θ,2θ strain
*map* (COM, linear) always gives λ; the λ/2 trap is specific to fixed-setting intensity. Two /verify
refinements: (i) this "frequency doubling" is a **correct but generic rocking-curve corollary, not a
named DFXM effect** — the *citable* standard result is only "use the weak-beam flank" (Poulsen; weak-beam
DFXM); (ii) the flank recovers the true λ **only for the strain channel** — the **tilt channel intensity
∝ $|$deviation$|$ (sign-independent), so it doubles at *any* operating point**, on- or off-Bragg
(reproduced numerically on a tilt-only reflection). It also assumes a symmetric rocking curve; dynamical/
extinction asymmetry would leak some fundamental back in.

### 9c. A single detector frame collapses x and z — retraction of the a–c dimensionality test

RETRACTED (findings 36): I claimed an a–c-plane image cleanly separates a 2D from a 3D wave. The
package's detector image is an **inclined projection**: with x‖beam and the diffracted beam at 2θ, the
detector v-axis images $(\cos2\theta\,z-\sin2\theta\,x)$, so **sample x and z both map onto v** and the
x–z checkerboard collapses along one axis in a single frame (see `gt_3d_400_mosaicity.png`, right
panel: the clean checkerboard is only in the sample-frame voxel slice, not the detector image). The
c-structure is recoverable from a **scanned/reconstructed volume**, not one image. /verify nuance: the
collapse is exact for a **box beam**; a **line/sheet beam** illuminating one layer already gives an
un-collapsed 2D map *of that layer*, and you scan layers for 3D — either way 3D needs scanning, so the
"single frame can't separate 2D/3D" conclusion stands. Two more honest notes the GT forced: real
contrast at $A\sim10^{-5}$ is **sub-percent** (0.2–2 %; stripes are in the data, not the eye), and the
DFXM resolution is **strongly anisotropic** (σ_rock ≈ 1e-3 ≪ σ_roll ≈ 9e-3), which a single-σ blur
misrepresents.

The fast helper was then corrected to include the strain **and** tilt channels and the λ/λ-2 weak-beam
behaviour; the projection-collapse, contrast and anisotropic-resolution simplifications remain and are
labelled, with the package renders embedded as the reference. Lesson: a fast forward approximation is
worth building for intuition, but check it against the validated forward before trusting its picture.
The forward behaviour is now **regression-locked** by `packages/midas_dfxm/tests/test_wave_period.py`
(4 tests: the tensor-shear convention, the 2 µm period off-Bragg, the λ/2 doubling at Bragg, and the
[110]-strain vs [100]-tilt channel split) — the last gap the /verify pass flagged.

---

## 10. Prior-art comparison: Carlsen's DFXM forward-model thesis

Source: **Mads Allerup Carlsen, "Phase Resolved Dark-Field X-ray Microscopy," PhD, DTU Physics
2022** (Poulsen/Simons group, ESRF ID06-HXM), read in full 2026-08. His trilogy: **[12]** wave-optics
DFXM simulator (Acta Cryst A 2022), **[13]** Fourier-ptychographic DFXM (Opt. Express 30:2949 2022),
**[14]** a Takagi–Taupin integrator on an arbitrary orthogonal grid (Acta Cryst A 78 2022).

**It independently confirms our verified physics** (a primary authority, so this promotes §9 from
package-GT to literature-backed): the reciprocal shift is $\Delta\mathbf g=(\mathbf I-\nabla\mathbf u^{\mathsf T})\mathbf g_0
=-\mathbf H^{\mathsf T}\mathbf g_0$ (his Eq. 3.17); the strain-vs-tilt channel split is Eq. 3.21 /
Poulsen [58] (a single reflection returns the row $(\nabla\mathbf u)\!\cdot\!\hat{\mathbf Q}$ = one axial
strain + two tilts); the single-frame **inclined projection** is a projection along
$\mathbf k_h=k[\cos2\theta,0,\sin2\theta]$, so the in-plane observation axis is
$\cos2\theta\,z-\sin2\theta\,x$ — **exactly our C3 formula, and only in *projection* (box-beam) mode; in
*slicing* (condenser/line-beam) mode a single frame is a clean z=0 sheet**, which settles the a–c
question. He gives **no λ/2 statement** (consistent with our labelling it a generic rocking-curve
corollary). And the kinematic bound is set by **deformation/near-perfect, not thinness/Z**: his
validating diamond is thin and low-Z (μ_att·t<1) yet **near-perfect**, so it needs the full TT — the
distinction is *absorption-thin* (μ_att·t) vs *extinction-thin* (t/Λ); our doc set already uses t/Λ, the
correct one, so no correction needed (matches our own ~0.3 Λ boundary, §4).

**Where each is ahead (honest, not advocacy):**
- **midas ahead:** differentiable **inverses** — full-F deformation-gradient tensor, Stroh dislocation
  typing, the defect model, CPFEM inference, Fisher design — none of which Carlsen's thesis carries; plus
  breadth and the validated pipeline. midas also has a (2D) TT dynamical forward.
- **Carlsen ahead:** a full **3D wave-optics microscope simulator** (incident beam → crystal TTE → CRL
  propagation → detector, validated vs a diamond stacking fault); a **peer-reviewed general TT integrator**
  [14] (exponential-Heun, arbitrary orthogonal grid, beats the traditional half-step); the **anisotropic**
  resolution (objective NA → σ_par/σ_roll, condenser NA → σ_rock); and **Fourier ptychography / DPC phase
  retrieval**, which midas entirely lacks (only `talbot.py` grating interferometry exists, a different
  method). **Verdict: complementary, not dominant.**

**Two grounded package findings:** (1) `aligned_resolution` collapses the two transverse widths to one
(transverse-isotropic) even though `poulsen_resolution_widths` returns σ_rock ≠ σ_roll (finding 37,
phase-1 §1c) — extend it to two transverse widths for full anisotropy. (2) **Fourier ptychography is
absent** — but FP is a phase-retrieval *inverse*, which fits midas's differentiable, inverse-heavy design
(the aberration pupil PSF + `wave_imaging` + the optimizer already exist), so a **differentiable FP-DFXM**
is a natural next build and a live ask from the 6-ID-C collaborator. Scope kept with the analysis campaign
(`~/Desktop/analysis/yay_strainwaves_dryad/`). Carlsen's methods are published/citable — cite them; do not
recruit (bilateral-collaboration rule).

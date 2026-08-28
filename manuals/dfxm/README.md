# DFXM Reduction and Analysis Runbook — survey → configure → reduce → tensor → analyse → report

Dark-field X-ray microscopy: from raw detector frames to per-pixel orientation and strain
maps, the multi-reflection deformation-gradient tensor where the data allow it, and a
report whose every number names the file and command that produced it.

This is the **spine** of the DFXM doc set. It is the one file meant to stay loaded. It
carries the scope gate, the install gate, the order of operations, the hard rules, and the
halt conditions. The detailed procedure lives in the phase files; the index below says
which file holds which section.

**Path conventions.** `$MIDAS` is the root of whichever MIDAS checkout you are working in
(on a beamline host, `~s1iduser/opt/MIDAS_canonical`). **`$ANALYSIS` is a campaign working
directory that is NOT in this repo** — the harnesses that produced numbers in
`LAB_NOTEBOOK.md` are local, deliberately unversioned analysis scripts, so a `$ANALYSIS/...`
path is *provenance, not a link*: it names the script a number came from, and promises
nothing about reaching it from where you are sitting.

> **Honesty about depth.** The far-field and near-field doc sets encode years of beamtime.
> DFXM in this group has **one** real-data reduction campaign (the archived ESRF ID03
> mosaicity sets and the ID06 multi-Bragg set) plus a simulation-and-inverse framework
> (`midas_dfxm`). Steps that are **real-data-proven** and steps that are
> **simulation-grounded** are marked as such throughout — do not let a clean simulation
> result read as a validated measurement. This file is meant to grow as real DFXM
> beamtimes are run.

---

### The doc set — what to read when

| File | Holds | Read when |
|---|---|---|
| `README.md` (this) | scope + install gate, the order, hard rules, halt conditions, phase index | first, and keep loaded |
| `phase-0-survey.md` | survey the scan folder; write `SURVEY.md` | before promising anything |
| `phase-1-configure.md` | material, reflection, geometry, susceptibility, Λ, resolution | once you know what is there |
| `phase-2-reduce.md` | raw frames → orientation/strain maps; pedestal; injection-recovery | the core real-data step |
| `phase-3-multireflection.md` | multi-reflection full-F tensor; the registration gate | only if ≥2 co-registered reflections exist |
| `phase-4-analyse.md` | typing, defect model, mosaicity, the kinematic validity boundary | for anything past orientation |
| `phase-5-report.md` | report with provenance; what a healthy number looks like | at the end |
| `DIAGNOSIS.md` | symptom → discriminating test → cause → lever | when something looks wrong |
| `LAB_NOTEBOOK.md` | what was found, how it was measured, what was refuted | before re-opening any question |
| `ENVELOPE.md` | what this measurement can and cannot determine, and which of those is changeable | before promising an answer, and before proposing a different measurement |
| `SURVEY_TEMPLATE.md` | the per-scan survey form — copy to `SURVEY.md` | at the start of every dataset (§0b) |
| `RUNBOOK.md` | where it runs, healthy ranges, current pick-up point | at session start and end |

Section numbers (§n) are continuous across the set. `Λ` is the extinction length
throughout; `θ_B` the Bragg angle; `χ_0`, `χ_h` the susceptibility Fourier coefficients.

---

## STOP — read this before touching anything

### When to stop and come back with a question

**"Get back to me if you get stuck" does not fire here.** DFXM's failures finish and look
right: a pedestal-dominated centroid produces a smooth orientation map that is 67× too
small in amplitude; two reflections at different 2θ overlay into a plausible but meaningless
tensor; a refraction offset reads as a clean uniform strain; a kinematic strain inverse on
a thick crystal converges to a biased answer. In each case the run succeeds.

So the trigger is not confusion. **Halt on these named conditions, whether or not anything
seems wrong:**

| Condition | Why you cannot decide it yourself |
|---|---|
| the data is **APS 6-ID-C**, not ESRF ID06/ID03 | different instrument, different geometry, and a bilateral collaboration (credit) — confirm first (scope) |
| you are about to take a first moment on frames you have **not** background-subtracted | the centroid will be pedestal-dominated and ~67× too small, with no error (§2, Notebook §1) |
| you are asked to combine **≥2 reflections** into a tensor and there is **no co-registration metadata** | inter-reflection registration, not photon statistics, is the binding systematic; the fused tensor is meaningless (§3, Notebook §2) |
| you are about to "correct" a **uniform** strain offset (~100s of µε) as a physical field | it is almost certainly the refraction gauge — a reference offset, not an intragranular strain (§4, Notebook §3) |
| you are inverting **strain/defect** contrast on a crystal thicker than ~0.3 Λ with the kinematic model | past ~0.3 Λ the kinematic inverse biases; you need the dynamical forward (§4, Notebook §4) |
| you are about to report accuracy from a **forward/inverse round-trip** | that is a software-consistency check (1e-16), not physical accuracy — use injection-recovery (§2, Notebook §1) |
| you are about to compare **integrated intensities between separately-acquired scan groups** and the metadata carries no flux-monitor column | the missing normalisation cannot be recovered from an archive; it is a question for the data's authors (§0b, Notebook §7d) |
| the **µm/px scale** rests on a constant you cannot trace to an optical record | every length in the result scales with it, and a factor ~2 error is the common one — ask for the optics record (§1, Notebook §7e) |
| you are about to report a **discrepancy with someone else's published reduction** | read that pipeline's own scripts first (the correction is often already in it), and the report is a collaboration matter, not only a technical one (§0b, Notebook §7d) |
| the **measured resolution is worse than the instrument's demonstrated best** and you want to correct for it | you need the vibration **spectrum**, not an amplitude — an archive cannot supply it, and short exposures do not recover low-frequency power (Notebook §7f) |
| a **control returned the expected answer** and you cannot say what result would have refuted it | that is not a control — three of ours could not fail and reached user-facing text (rule 17) |
| this document and the tree **disagree** | report it; do not work around it |

When you halt, say which row fired, what you measured, and what you would need to proceed.
Finish everything not blocked by it first.

### Hard rules

1. **Subtract the pedestal before the first moment (§2).** On raw ID03 frames the pedestal
   carries **98.5 %** of the centroid weight, so the naive first moment is **~67× too
   small**. Reproduce `darling` on **background-subtracted** frames; the two pipelines then
   agree to correlation 1.0, RMS ~1e-7°. This is required for *correctness*, not tidiness
   (Notebook §1).

   > **The 67× is the *median* estimator's number and is not what you will measure.** It is
   > `1/(1−f_ped)` with the floor at the median (`SURVEY_TEMPLATE.md`). The documented
   > *subtraction* recipe uses percentile 5 — deliberately conservative, because
   > over-subtracting eats signal — and a before/after comparison run that way gives
   > **~18.6×**, reproduced 2026-08-12. Both are correct for what they measure. A reader
   > who takes 67× as "what a before/after should show" will see 18.6× and wrongly conclude
   > the rule failed to reproduce. Quote the estimator with the number, always.

2. **Refraction is a gauge, not a per-pixel strain (§4).** The mean refraction shifts the
   Bragg peak by a constant $\varepsilon_{\mathrm{ref}}=\chi_{0r}/(2\sin^2\theta_B)$
   (≈ 144 µε for Cu 002 at 0.71 Å). It is constant across a grain for a fixed reflection
   and energy, so on a **relative** strain map it is absorbed into the lattice reference —
   do not subtract it as a field. It is a real bias only where refraction **varies**
   (thickness / perfection gradient), and that residual lives in the near-perfect matrix
   where dark-field contrast is weak, so it is bounded, localisable and maskable
   (Notebook §3).

3. **Orientation is centroid-exact; strain has a validity boundary at ~0.3 Λ (§4).** In
   symmetric-Laue geometry the rocking curve is even and absorption preserves that, so the
   per-pixel centroid orientation is exact by symmetry — not merely small. The kinematic
   defect/strain inverse holds only for crystals thinner than **~0.3 Λ**; by ~0.15 Λ the
   fit residual leaves the noise floor, by ~1.1 Λ the recovered amplitude is biased ~+38 %.
   This was verified **cross-model** (dynamical data, kinematic inverse), not by inverse
   crime (Notebook §4).

4. **Multi-reflection registration is the binding systematic, not photon statistics (§3).**
   Different reflections diffract at different 2θ → different magnification and field of
   view. Without co-registration metadata (fiducials, a shared sample frame) the
   per-reflection maps cannot be fused into a deformation-gradient tensor. On the real ID06
   111↔002 set the best content cross-correlation was 0.43 at the search edge; the deposit
   carried no co-registration metadata. **This is a property of the experiment to report,
   not a parameter to tune** (Notebook §2).

5. **Validate real data by injection-recovery, never round-trip.** A forward-then-inverse
   round-trip returns 1e-16 because it inverts its own generator — a software-consistency
   metric. For physical accuracy, resample a *known* orientation/strain shift into the
   measured raw frames and recover it against the real noise, background and detector
   (gain 0.9998–1.0000 on the four ID03 scans) (Notebook §1).

6. **Units: µm, degrees, Å** (Å for wavelength and lattice parameters only). Output Euler
   angles are radians; misorientation in `midas_stress` is radians, axis–angle in degrees.

7. **Do not reimplement what a `midas_*` package already does.** Orientation and
   misorientation → `midas_stress`; X-ray physics / structure factors / susceptibility →
   `midas_hkls` and `midas_dfxm.takagi_taupin.susceptibility_fourier`; image reading → the
   MIDAS readers, not `fabio`; anything DFXM-forward/inverse → `midas_dfxm`. The paper
   `packages/midas_dfxm/dev/paper/P_merged/` is the reference for every formula here.

The first rules distrust the *data*. These distrust your own run:

8. **Suspect success.** Every DFXM failure mode above reports success. "It ran" and "the
   map is smooth" are not evidence. Ask what the step would look like if it had silently
   done the wrong thing (kept the pedestal, fused unregistered reflections, absorbed the
   refraction gauge) and check that specific thing.

9. **Debug your own configuration before the physics.** Order: wrong pedestal/background →
   wrong reflection or geometry → a sign/unit convention → the kinematic-vs-dynamical
   regime → only then the sample. The Notebook §5 records claims that were **refuted** once
   the mundane cause was found — including a "world's-first" novelty claim the literature
   killed.

10. **Never take a number from a name.** Not the energy, the reflection, the 2θ, or the
    pixel size from a folder or filename — read it from the data/metadata and say which
    file in the report.

Rules 1–10 came from reducing our own scans. **Rules 11–20 came from a second campaign —
re-analysing an archived public deposit that had been reduced by another group's pipeline.**
They split the same way: 11–16 distrust the data and the upstream reduction, 17–20 distrust
your own analysis of it. Each is terse here; the measurement behind it is in Notebook §7.

11. **Read the producing pipeline's own scripts before comparing two channels, and before
    reporting any discrepancy with its output (§0b).** A preprocessing constant upstream of
    a comparison can be symmetric in pixels and wildly asymmetric in signal: one archived
    reduction zeroed a fixed block of detector rows holding **15× more** of one channel's
    in-aperture signal than of the other's, and toggling it moved **~20 %** of the published
    labels. Reading the scripts also stops you reporting a correction as missing when it is
    already there — an "uncorrected inter-reflection offset" we nearly reported was already
    zeroed by that pipeline's own peak-centring step (Notebook §7d).

12. **A published rocking-curve FWHM is the *integrated* width — measure the per-pixel width
    on the frames (§2).** The integrated curve is broadened by mosaic spread *across* pixels,
    so it is wider than what a per-pixel fit sees: **2.6–2.7×** on one archived set. Dividing
    the step into a published FWHM overstates points-per-FWHM by that factor, and it
    invalidated the premise of an entire preregistration. Take **argmax-local, contiguous**
    half-max crossings and check contiguity — global outermost crossings let one noise spike
    set the width, and a non-contiguous above-half-max set spans the gap between two islands
    (Notebook §7a).

13. **Measure the detector gain by photon transfer before quoting any absolute χ²/dof, σ, or
    "the model is rejected" (§2).** On one integrating sCMOS the measured gain was
    `var = 2.23·y + 149`, not `var = y`, inflating every absolute χ²/dof ~2.2× — enough to
    turn an adequate model (true χ²/dof 1.08) into an apparently rejected one (2.6) and
    invent a misspecification that was not there. **Remove the pedestal first:** a pedestal
    makes var/mean invalid as a gain estimate and can push it below 1. Photon-counting
    detectors do sit at var/mean ≈ 1 — verify which kind you have rather than assuming, and
    never carry one detector's gain onto another's frames (we did, and it flipped a result's
    sign). Ratio statistics — likelihood ratios, ROC/AUC — rescale together and are unaffected
    (Notebook §7b).

14. **A background that tracks the rocking curve distorts curve *shape* (§2).** Correlate the
    per-frame *level* of whatever you subtract against the integrated rocking curve: at
    r ≈ +0.92…+0.97 you are subtracting a θ-dependent scalar, and every per-pixel width or
    FWHM computed afterwards is biased — while the centroid is barely affected, so the map
    still looks right. The cause is mundane: a morphological-opening / rolling-ball background
    whose structuring element **exceeds the ROI**, on an ROI with no non-diffracting pixels to
    act as a common-mode reference, degenerates to exactly that scalar. Check kernel size
    against the downsampled ROI. Rule 1 says do not under-subtract; this says do not let the
    subtraction follow the signal (Notebook §7c).

15. **Never take the magnification — hence µm/px — from a script constant (§0b, §1).** On one
    archived reduction a hardcoded magnification set every published length and disagreed with
    four independent optical routes by about a factor of two — exactly the error a calibration
    makes if it counts a line pair as one feature. Derive it from the optical record and
    cross-check ≥ 2 independent routes. Two corollaries: a refractive objective specified at
    one energy has f ∝ E², so at a higher energy f can exceed the working distance and the
    geometry is not what the nominal number implies; and a **focus diagnostic tells you the
    image is sharp, not that the scale is right** (Notebook §7e).

16. **Before comparing intensities across separately-acquired groups, confirm a flux monitor
    exists and that the optics did not move (§0b).** One archive had no monitor column at all
    — the whole flux record was three hand-typed readings at ±11 %. And the **frame total is
    not a flux monitor on a rocking scan: it *is* the rocking curve** (use non-diffracting
    pixels; done that way flux was stable to 2.7 % on a public ID03 scan). Check the *optics*
    columns, not only the sample columns — an apparent inter-reflection image shift was fully
    explained by the objective having moved between groups, logged in the same motor file
    (Notebook §7d).

17. **Every control must be able to fail, and a correction must recover a planted bias.**
    Three "tests that could not fail" reached user-facing text in that campaign: a rule
    checked against a set containing no case that could break it (it printed `rule holds:
    True`), a null that injected the fitted model's own parameters back into the same pixels,
    and a "stability sweep" that varied two thresholds while leaving hardcoded the one that
    decided the answer. Before believing a control, state what result would have refuted it.
    For a correction, plant a known bias and read the residual: ours moved 39.5 % of labels,
    and on a planted 6.4 % bias left a **24.2 % residual — 3.8× worse than the bias it was
    meant to remove**. That is what killed it, at a cost of about twenty lines (Notebook §5j).

18. **A second script you also wrote is not an independent reproduction.** A projection number
    was "reproduced" by a second implementation and both were wrong: the prose had dropped a
    minus sign, and the script carried a 30° frame rotation (the reciprocal basis is
    Busing–Levy, so the real-space axis sits at −30°, and for a traceless in-plane deviator
    30° is exactly a sign flip). **Two errors cancelled and were logged as a reproduction.**
    Independence means a different route — a closed form, a frame-free invariant, or someone
    who has not seen the derivation (Notebook §5g).

19. **Map pixels are not independent samples.** An iid bootstrap over map pixels understates
    the variance, because the optical PSF and the microstructure both correlate neighbours: a
    "−3.2 σ" deficit over 6,812 pixels in a field autocorrelated to 0.90 at 48 px became
    **−0.73, p = 0.47** under a block bootstrap (σ 0.076) and a phase-randomised surrogate
    (σ 0.082), at n_eff ≈ 172 rather than 6,812. Quote the autocorrelation length; resample in
    blocks or use surrogates (Notebook §5h).

20. **To test whether a centroid is robust to lineshape, the alternative must be asymmetric.**
    A Gaussian and a symmetric pseudo-Voigt share a symmetry, so comparing their centres is
    blind to the one misspecification that moves a centre: skew. We reported a centre map
    robust at 0.045 mdeg; against an adequate **split** lineshape the centre moved a median
    **1.98 mdeg (p95 10.67)**, 12 % of the FWHM — a **44× error**, on the map the analysis
    leaned on most (Notebook §5i).

21. **One measured strain component plus compatibility does NOT give you the others — it is an
    identifiability limit, not a prediction (§4, Notebook §8b).** In-plane Saint-Venant
    compatibility is *one* equation in *three* unknown strain components, so a measured
    $\varepsilon_{xy}$ fixes only one combination of the normal strains. A companion field with
    **zero dilatation** is exactly as compatible as one with maximal dilatation; the partition is
    set by the elastic moduli, which single-peak data cannot access. Do not present a
    compatibility-derived companion (a "predicted dilatation wave", a filled-in $\varepsilon_{xx}$)
    as a measurement or as "demanded". The closure-free, falsifiable statement a multi-peak run
    must satisfy is the source term $\partial^2\varepsilon_{xx}/\partial y^2 +
    \partial^2\varepsilon_{yy}/\partial x^2 = 2\,\partial^2\varepsilon_{xy}/\partial x\partial y$.

22. **A reflection senses a given strain as EITHER a d-spacing change (θ,2θ) OR a lattice tilt
    (θ-rock / mosaicity), set by the reflection direction — not both equally (§4, Notebook §9;
    /verify ESTABLISHED across 4 lenses).** The shift is $\Delta\mathbf g=-\mathbf H^{\mathsf T}\mathbf g_0$
    (from $\mathbf g=\mathbf F^{-\mathsf T}\mathbf g_0$), with $\mathbf H$ the **symmetric** strain
    tensor here; a longitudinal part (‖$\mathbf g_0$ = strain) and a transverse part
    ($\perp\mathbf g_0$ = rotation). For a shear $\varepsilon_{xy}$, a reflection with $\mathbf g_0$
    along the shear \emph{diagonal} ($[110]_T$-type) sees pure strain; one along a \emph{cube axis}
    ($[100]_T$-type) sees pure tilt and is **blind in a θ,2θ scan**. **State the index frame** — the
    same Miller string flips channels between frames: orthorhombic 400 ($a_O\!\parallel\![110]_T$) is
    strain-sensing, tetragonal (400) is tilt-sensing. Before concluding "no signal", check the
    channel *and the frame*. In a fixed-θ **intensity** image the response at exact Bragg is even, so
    a period-λ wave images at **λ/2** (a generic rocking-curve corollary, not a named DFXM effect);
    the true λ returns off-Bragg on the **weak-beam flank** — but only for the **strain** channel (a
    tilt-channel intensity ∝ $|$deviation$|$ doubles at any operating point).

### Traps that silently corrupt results

| Trap | Symptom if missed | Where |
|---|---|---|
| first moment on raw (un-subtracted) frames | orientation amplitude ~67× too small, map still smooth | §2 |
| refraction offset subtracted as a strain field | a real uniform strain "removed"; gradient bias left in | §4 |
| two reflections overlaid without co-registration | plausible full-F tensor that means nothing | §3 |
| a compatibility-derived companion component quoted as a measured/"demanded" dilatation | a closure-dependent number (true value ranges 0→upper bound) read as physics | §4, Notebook §8b |
| a periodic strain wave judged real by a scalar coherence/SNR number | a non-periodic phase scores comparably; use autocorrelation periodicity + reflection rotation | Notebook §8c, DIAGNOSIS |
| kinematic strain inverse past ~0.3 Λ | converged, biased amplitude; looks like a real strain | §4 |
| a shear-strain reflection reported as "no contrast" | wrong channel — a cube-axis reflection sees the shear as tilt (θ-rock), not strain (θ,2θ) | §4, Notebook §9 |
| a fixed-θ intensity image read at exact Bragg | 2nd-order response images the wave at **λ/2**; go weak-beam for the true λ | Notebook §9 |
| a single DFXM frame used to call a wave 2D vs 3D | the inclined projection collapses sample x and z onto one detector axis; you need the scanned reconstruction | Notebook §9 |
| a Miller string ("the 400 reflection") quoted without its index frame | channel inverts: ortho 400 (=tet [110]) is strain-sensing, tet 400 (cube axis) is tilt-sensing | Notebook §9a |
| round-trip quoted as physical accuracy | 1e-16 "validation" that tests only the linear algebra | §2 |
| mosaicity read as intrinsic sample spread | it is the intrinsic spread **convolved with the instrument resolution** — deconvolve with `fit_orientation_mosaicity`, not `moment_orientation` | §2, §4 |
| resolution widths assumed isotropic | DFXM resolution is anisotropic (Poulsen); use `poulsen_resolution_widths` | §1 |
| Λ, θ_B, or χ taken from a template | they are material-, reflection- and energy-specific; compute from `susceptibility_fourier` / `extinction_length` | §1 |
| capability result (typing, defect model, full-F) reported as real-data-validated | those are simulation-grounded in this package as of writing — say so | §4 |
| a published/quoted rocking FWHM used as the per-pixel width | points-per-FWHM overstated ~2.6×, so a per-pixel model-selection test runs outside its validity | §2 |
| absolute χ²/dof quoted with gain assumed = 1 | adequate models look rejected; a misspecification is invented | §2 |
| var/mean used for gain without removing the pedestal | the gain estimate is invalid and can come out below 1 | §2 |
| background whose per-frame level tracks the rocking curve | per-pixel widths biased; the centroid barely moves, so the map still looks right | §2 |
| morphological / rolling-ball kernel larger than the ROI | the "spatial" background degenerates to a θ-dependent scalar | §2 |
| magnification taken from a script constant | every length wrong by that factor, and a focus check will not catch it | §1 |
| intensity comparison across separately-acquired groups with no flux monitor | an unnormalised comparison that reads as a physical difference | §0b |
| frame total used as a flux monitor on a rocking scan | you have normalised the rocking curve by itself | §2 |
| iid bootstrap on autocorrelated map pixels | significance inflated ~√(n/n_eff) — a nonexistent effect at several σ | §4 |
| centroid robustness tested against a *symmetric* alternative lineshape | the test always reassures; skew is the misspecification that moves a centre | §2 |
| an intensity ratio between two channels read as a phase fraction | a shared channel makes the neutral point neither 0.5 nor constant; single-phase voxels land in "mixed" | §4 |
| moment-based bimodality on a rocking curve below ~12 pts/FWHM | measures curve *broadness*; flagged and unflagged pixels come back indistinguishable | §4 |
| one fixed rocking window reused across a raster while θ_B drifts | truncated positions bias widths and integrals, and manufacture apparent two-population structure | §2 |
| Λ assumed similar for a weak satellite and a strong parent | Λ ∝ 1/\|F\|, so the dynamical boundary can bind for one and never for the other | §1 |
| rocking-width / Darwin-width used as a dynamical-relevance criterion | the criterion is t_coherent/Λ; a wide rocking curve does not bound dynamical effects | §1, §4 |

---

## 0. Environment and install gate — before anything else

`midas_dfxm` is a **Python library**, not a CLI. The workflow is script-based: import the
package and call its functions. There is no `midas-dfxm run` command.

```bash
pip install "midas-dfxm>=0.3.2"        # public on PyPI since 2026-07-29
# reductions of ESRF frames also use the community package:
pip install darling-pypi               # ID03/ID06 raw-frame moment reduction; IMPORTS as `darling`
                                       # (`pip install darling` fails -- no such distribution)
```

Then run the gate and read its output — `pip install` exiting 0 tells you nothing:

```bash
python - <<'PY'
import midas_dfxm as dx
print("midas_dfxm", dx.__version__)
# the three pillars the workflow leans on must import:
from midas_dfxm.mosaicity_fit import moment_orientation, fit_orientation_mosaicity
from midas_dfxm.field_inverse import recover_deformation_direct, deformation_identifiability
from midas_dfxm.takagi_taupin import susceptibility_fourier, extinction_length, solve_tt_laue
print("reduction + full-F + dynamical: import OK")
PY
```

On an APS beamline host use the shared env by full path
(`/home/beams12/S1IDUSER/opt/envs/midas/bin/python`); GPU prefix
`CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE`; on the Mac activate the project
env (`midas_env` in use). Outputs go in a project/gdata directory you own — **never
`/tmp`**. Long jobs get `setsid`/`nohup` + a redirect.

---

## 0a. THE ORDER — do these in this sequence

Two steps cannot be checked after the fact, so they come first.

1. **Classify the scan and confirm the pedestal path (§0, §2).** Is it a mosaicity (rock/
   roll) scan, a strain (θ) scan, or a multi-reflection set? Confirm you have the
   background/pedestal you will subtract *before* reducing. A first moment on raw frames
   cannot be un-done later.
2. **Configure material, reflection, geometry (§1).** Compute θ_B, Λ, χ from the material
   and energy — never a template. This fixes whether you are in the thin (kinematic-safe)
   or thick (dynamical) regime, which changes what §4 is allowed to claim.
3. **Reduce to per-pixel maps (§2).** Moment orientation on background-subtracted frames;
   validate by injection-recovery; reproduce `darling` as the arithmetic cross-check.
4. **Only if ≥2 co-registered reflections exist, attempt the tensor (§3).** Check the
   registration gate first; if it fails, stop at per-reflection maps and report the wall.
5. **Analyse past orientation (§4).** Mosaicity (deconvolved), typing, defect model — each
   labelled real vs simulation — and the kinematic validity boundary for any strain claim.
6. **Report with provenance (§5).**

---

## Phases

Open each as you reach it:

- **[phase-0-survey.md](phase-0-survey.md)** — §0b survey, §0c already-processed check
- **[phase-1-configure.md](phase-1-configure.md)** — §1 material/reflection/geometry/Λ/resolution
- **[phase-2-reduce.md](phase-2-reduce.md)** — §2 pedestal (and over-subtraction), §2a′ detector gain, moment reduction, injection-recovery, `darling` cross-check, §2f the per-pixel rocking width
- **[phase-3-multireflection.md](phase-3-multireflection.md)** — §3 the registration gate, full-F tensor, the rank-6 ceiling and the oblique-geometry prior art
- **[phase-4-analyse.md](phase-4-analyse.md)** — §4 mosaicity, typing, defect model, the ~0.3 Λ boundary, the refraction gauge, §4d′ shared-channel ratios, §4d″ statistics on maps
- **[phase-5-report.md](phase-5-report.md)** — §5 report and healthy ranges

When something looks wrong: **[DIAGNOSIS.md](DIAGNOSIS.md)**. Before re-opening a settled
question: **[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md)**. Before promising that this measurement can
answer the question at all — or proposing a different one — **[ENVELOPE.md](ENVELOPE.md)**.

# DFXM diagnosis reference

Symptom → discriminating test → cause → lever, for dark-field X-ray microscopy. Indexed by
*symptom*, not by step — the step that produced a symptom is rarely the step you are on.
Every entry carries a test that can come back the other way; an entry that cannot exonerate
the cause it names does not belong here.

Source of the content: this campaign's `LAB_NOTEBOOK.md` and the paper
`packages/midas_dfxm/dev/paper/P_merged/`. The entries from **"A quoted rocking width…"**
onward come from the second campaign — a re-analysis of an archived deposit reduced by another
group's pipeline — and cite Notebook §7 for what it established and Notebook §5f–§5l for what it had to
retract. All are stated instrument- and sample-independently.

## Local symptoms

These are emitted by **this technique's own procedure**, not by `beamreport`'s generic
diagnostics, which key off per-observation residuals against declared coordinates. A DFXM
pedestal dilution or an inter-reflection registration failure is real, useful, and nothing
generic will ever detect it — so it is declared here rather than renamed into the wrong
shape.

Every row names where the check lives. A symptom nothing produces is dead text that reads
as coverage, which is exactly what the generic vocabulary existed to prevent.

| symptom | emitted by |
|---|---|
| `registration_fail` | inter-reflection co-registration gate (Notebook §2) |
| `strain_inverse_biased` | amplitude bias check on the strain/defect inverse (Notebook §4b) |
| `width_scale_mismatch` | quoted rocking width against a per-pixel fit (Notebook §7a) |
| `background_theta_coupled` | correlation of background estimate with the rocking angle (Notebook §7c) |
| `intergroup_offset` | between-group offset check (Notebook §7d) |
| `scale_magnification_unverified` | magnification/scale verification (Notebook §7e) |
| `resolution_degraded` | measured resolution against the instrument's demonstrated best (Notebook §7f) |
| `ratio_threshold_invalid` | channel-ratio-to-phase-fraction validity check (Notebook §7i) |
| `map_significance_inflated` | map-statistic significance against its null (Notebook §5h) |
| `centroid_lineshape_untested` | lineshape-invariance test on the centroid (Notebook §5i) |
**Six entries migrated to the generic vocabulary** (2026-08-12) once `beamreport` grew the
detectors for them: the pedestal dilution and the too-broad mosaicity are `scale.suppressed`
/ `scale.inflated`, the uniform strain offset is `systematic.common_offset`, the doubled
rocking curve is `split.bimodal`, the inflated chi-squared is `uncertainty.miscalibrated`,
and the circular feature width is `floor.limited`. They are detected generically now, so the
entries below explain them rather than also having to find them. The ten that remain are
genuinely DFXM-specific.

---

## Orientation map is smooth but the amplitude is far too small

symptom: scale.suppressed
coord: orientation

**Test.** Recompute the per-pixel first moment on **background-subtracted** frames and
compare the intragranular spread to the raw-frame result. If the amplitude jumps by ~1–2
orders of magnitude, the raw run was pedestal-dominated. Confirm by measuring the pedestal's
share of the centroid weight: if the flat background carries ≳ 95 % of $\sum I$, the moment
is diluted. If subtracting the background changes the amplitude by < 10 %, the pedestal is
not the cause — look at the scan range instead (too few points across the rocking curve
flattens the moment for a real reason).

**Cause.** A large positive detector pedestal under every pixel pulls the intensity-weighted
mean toward the frame centre, underreporting the true centroid excursion by
$1/(1-f_{\text{ped}})$. On raw ID03 frames $f_{\text{ped}} = 0.985$ gave a ~67× dilution
(Notebook §1a).

**Lever.** Subtract the scalar background (`darling`'s own, or a measured dark) before the
moment. Verify against `darling` on the subtracted frames (correlation should reach 1.0).
Never quote an orientation amplitude from an un-subtracted moment.

## Two reflections will not co-register / the full-F tensor looks wrong

symptom: registration_fail
coord: multi_reflection

**Test.** Compare the two reflections' frame geometry *before* trying to fuse them: their
2θ, magnification and field of view. If the 2θ differ substantially (e.g. 67.5° vs 14.2°),
the magnifications and FOVs differ and the maps are not on a common grid — search the best
intensity cross-correlation over scale + shift and read where the maximum sits. A maximum at
the **search edge** with a low NCC (≈ 0.4) and no co-registration metadata in the deposit is
the wall. If instead the NCC peaks cleanly in the interior at a sensible scale, registration
is feasible and this entry does not apply.

**Cause.** Different reflections diffract at different 2θ → different objective magnification
and FOV; without fiducials or a shared sample frame the per-reflection maps cannot be placed
on a common voxel grid. Inter-reflection **registration**, not photon statistics, is the
binding systematic (Notebook §2).

**Lever.** There is nothing to tune — **halt** and report the wall as a property of the
experiment. If the field is structured, the differentiable substrate can self-register
(carry per-reflection shifts as parameters and minimise multi-reflection consistency —
removes about half), but do not present a fused tensor built without metadata as a
measurement.

## A clean, uniform strain offset of order 100s of µε across the whole grain

symptom: systematic.common_offset
coord: strain

**Test.** Map the offset spatially. If it is **constant** across the grain (flat within the
per-pixel scatter) it is a reference offset, not a field — toggle the refraction term in the
forward (χ₀ on/off) and see if the offset equals $\chi_{0r}/(2\sin^2\theta_B)$ for this
reflection/energy (≈ 144 µε for Cu 002 at 0.71 Å). If instead the offset **varies** across
the grain and correlates with a thickness or perfection gradient, it is (partly) a real
gradient-driven bias and this entry only partly applies.

**Cause.** The mean refraction shifts the Bragg peak by a constant; a kinematic reader books
it as a uniform apparent strain. On a relative intragranular map it is a **gauge** absorbed
into the lattice reference (Notebook §3).

**Lever.** Do **not** subtract it as a per-pixel field. For a *relative* strain map, leave
it — it cancels in the reference. Apply it only to set the **absolute** strain scale
(cross-reflection or absolute-d work). If it varies spatially, the varying part is real; its
absolute magnitude needs the Takagi–Taupin forward, and it is maskable in the near-perfect
matrix.

## The strain/defect inverse converged but the amplitude looks biased

symptom: strain_inverse_biased
coord: strain

**Test.** Compute the crystal thickness in extinction lengths, $t/\Lambda$
(`extinction_length` for this reflection/energy). If $t \gtrsim 0.3\,\Lambda$, invert the
same contrast with the dynamical forward and compare — a kinematic-vs-dynamical amplitude
gap that grows with thickness is the regime boundary. If $t \lesssim 0.15\,\Lambda$ and the
two inverses agree, the kinematic model is not the cause; look at registration or the
reference offset instead.

**Cause.** The kinematic (first-Born) inverse leaves the noise floor by ~0.15 Λ and biases
the recovered amplitude past ~0.3 Λ (+38 % by 1.1 Λ); the leading dynamical correction the
kinematic model cannot absorb grows as $(\pi t/\Lambda)^2$ (Notebook §4b). This is the
near-perfect/thick regime, where dark-field contrast is also weakest.

**Lever.** Use the dynamical (Takagi–Taupin) forward/inverse for the strain claim, or
restrict the quantitative strain claim to the thin part of the field. The geometric full-F
inverse is exact on clean thin-crystal data (round-trip 4.67e-20), so the fix is the forward
model, not the linear algebra.

## Measured mosaicity is larger than the physics expects

symptom: scale.inflated
coord: orientation

**Test.** Fit the physical forward (`fit_orientation_mosaicity`) — a local orientation plus
an intrinsic mosaic covariance **convolved with the instrument resolution** — and compare to
the raw moment/Gaussian spread (`moment_orientation`). If the fitted intrinsic spread is
markedly smaller than the moment spread, the excess was instrument resolution. If the two
agree, the spread is intrinsic and this entry does not apply.

**Cause.** Both the moment and a phenomenological Gaussian report the *measured* spread =
intrinsic mosaicity ⊛ (anisotropic) instrument resolution. Reporting the measured spread as
intrinsic overstates the sample mosaicity.

**Lever.** Deconvolve with the physics-forward fit; supply the anisotropic resolution from
`poulsen_resolution_widths`, not an isotropic assumption. Report the intrinsic spread with
its resolution kernel named.

## A quoted rocking width disagrees with what a per-pixel fit sees

symptom: width_scale_mismatch
coord: reduction

**Test.** From the same frames, compute both widths: the whole-image **integrated** rocking
width, and the **per-pixel** median width using argmax-local, contiguous half-max crossings.
If the integrated width is ~2–3× the per-pixel median and the quoted number matches the
integrated one, the quoted number is an integrated width. If the two agree to ~10 %, the
quoted width already describes the per-pixel curve and this entry does not apply — look at
the fit window or the step instead. Separately check **contiguity** of the above-half-max
index set: if it is non-contiguous, the reported width spans a gap between disjoint islands
and is meaningless regardless.

**Cause.** The integrated curve is broadened by mosaic spread *across* pixels, so it is
wider than the curve any single pixel presents; a per-pixel fit sees only the latter.
Measured 2.6–2.7× on one archived scan (Notebook §7a). Global outermost half-max crossings
inflate it further by letting one noise spike set the width.

**Lever.** Derive points-per-FWHM from the **per-pixel** width, never from a published or
integrated one — a per-pixel model-selection test needs ≳ 12 pts/FWHM and this error is what
puts it outside validity. Re-check any preregistration whose sampling premise came from a
published width.

## χ²/dof is far from 1 everywhere, or every candidate model is "rejected"

symptom: uncertainty.miscalibrated
coord: reduction

**Test.** Measure the gain by photon transfer: regress the variance of nearest-neighbour
differences against local mean, **after removing the pedestal**. A slope ≈ 1 with intercept
≈ 0 means photon-counting statistics and the gain is *not* your problem — then look at σ
double-counting (a MAD noise estimate already contains dark and read) or at the background.
A slope ≳ 2 means the counts are ADU, not photons. If var/mean comes back **below 1**, you
have not removed the pedestal — the estimate is invalid, not informative.

**Cause.** Absolute χ² scales as 1/gain. On one integrating sCMOS the measured gain was
`var = 2.23·y + 149`, inflating every absolute χ²/dof ~2.2× — enough to make an adequate
model (true 1.08) look rejected (2.6) and invent a misspecification (Notebook §7b).

**Lever.** Re-quote all absolute χ²/dof and error bars at the measured gain, per detector —
never carry one detector's gain onto another's frames. Ratio statistics (likelihood ratios,
ROC/AUC) rescale together and need no correction, so a relative conclusion built on them
survives.

## The background subtraction tracks the rocking curve

symptom: background_theta_coupled
coord: reduction

**Test.** Record the per-frame **level** of whatever you subtract and correlate it against
the integrated rocking curve. At r ≳ 0.9, with the level swinging a non-trivial fraction of
the peak, the "background" is following the signal. Then check the filter geometry: compare
the structuring-element size against the **downsampled** ROI — if the element exceeds the
image, the spatial filter has degenerated to a scalar. If r is small and the kernel fits
inside the ROI, the filter is a genuine spatial background and this entry does not apply.

**Cause.** A morphological-opening / rolling-ball background on an ROI with no
non-diffracting pixels has no legitimate common-mode reference, so it returns a θ-dependent
scalar (r = +0.919 in the degenerate case, and still +0.966 where the kernel fitted).
Subtracting a θ-dependent scalar distorts rocking-curve **shape** (Notebook §7c).

**Lever.** Use a dark-only background. Any per-pixel **width/FWHM** derived from data cleaned
this way is biased and must be recomputed; a **centroid** is far less sensitive, so
orientation maps usually survive — which is exactly why the fault is invisible. If a cleaning
recipe was validated by injection-recovery, check whether it tested *feature* recovery or
*shape* preservation; they are different tests.

## An intensity ratio between two channels is being read as a phase fraction

symptom: ratio_threshold_invalid
coord: segmentation

**Test.** Compute |F|² at the **measured Q** for every candidate structure *and* every
twin/domain variant. If exactly one structure contributes at each channel's Q, the ratio is a
genuine two-state contrast and this entry does not apply. If both contribute at one of them,
compute the ratio a **100 % single-phase** region would produce as a function of local variant
population, and compare it against your threshold band. Commensurate periods are the thing to
look for: a supercell of period 2n·c reproduces every reflection of one at n·c.

**Cause.** A shared channel. Twinning compounds it — a symmetry rotation permutes which
modulation arm each variant contributes to the same lab-frame **Q**, and systematic
extinctions can forbid one variant outright — so the ratio's neutral point is neither 0.5 nor
constant. Measured spread across variants: 13.9× (Notebook §7i).

**Lever.** Prefer a channel that is **forbidden** in the competing structure. Where none
exists, replace the hard deadband with a per-pixel class probability plus an explicit
"undecidable at this dose" class. Do not attack this with a better feature extractor: a better
extractor on a mis-specified contrast returns a mis-specified answer.

## A map statistic comes back at many σ

symptom: map_significance_inflated
coord: statistics

**Test.** Measure the field's spatial autocorrelation length, then recompute the significance
two ways: a **block bootstrap** with blocks larger than that length, and **phase-randomised
surrogates** that preserve the power spectrum. If σ collapses, the iid resampling was the
cause. If the effect survives both, autocorrelation is not the explanation — then check
whether the null is matched on the right nuisance variables (matching intensity alone is not
matching; adding a |∇I| match flipped one sign) and whether the null has the same hierarchy as
the data (a single global mean/sd gives **zero** between-scan degrees of freedom when most
variance is between scans).

**Cause.** Map pixels are not independent samples — the optical PSF and the microstructure both
correlate neighbours. A "−3.2 σ" over 6,812 pixels in a field autocorrelated to 0.90 at 48 px
became −0.73, p = 0.47 at n_eff ≈ 172 (Notebook §5h).

**Lever.** Quote n_eff and the autocorrelation length alongside any map significance. Also
sanity-check the statistic's admissible range before believing it: a "dip" of 0.74 cannot be
Hartigan's dip, which is bounded by 0.25.

## Measured spatial resolution is worse than the instrument's demonstrated best

symptom: resolution_degraded
coord: resolution

**Test.** Separate **within-frame** blur from **between-frame** drift: compare an edge width
measured in a single short frame against the same edge in the summed stack. If the short frame
is already blurred, the cause is within-exposure vibration and re-registration cannot help. If
only the sum is blurred, it is drift and registration recovers it — this entry then does not
apply. If it is vibration, the next question needs the **frequency spectrum**: shortening the
exposure only helps for power faster than the frame time, so if the power sits below ~1/exposure
the short-frame lever fails and shortening exposures is wasted beamtime.

**Cause.** Sample/objective vibration in the ~1–100 Hz band; the resolution penalty **saturates**
once the exposure exceeds the slowest period, so long exposures sit in a regime where amplitude
alone predicts nothing about recoverability (Notebook §7f).

**Lever.** Ask for a vibration **spectrum**, not an rms amplitude — an archive of images cannot
supply it (README STOP table). Note that whole-ROI cross-correlation "registration" of a rocking
stack addresses only drift, and on a rocking stack the content itself changes with θ, so it is
not a fix for within-frame blur.

## A fitted feature width sits at the pipeline's own resolution floor

symptom: floor.limited
coord: resolution

**Test.** Refit the same feature with a deliberately different fit window / ROI half-width. If
the recovered width tracks the window, it is measuring the window, not the feature (one erf fit
ran 0.95 → 1.53 µm this way). Then compute the pipeline floor independently — PSF ⊛ step ⊛ any
projection through the illuminated thickness — and compare: if the fitted width lies within the
floor, the feature is unresolved. If the width is stable across windows and comfortably above
the floor, it is genuinely resolved and this entry does not apply.

**Cause.** An erf/step fit on an unresolved edge returns a width set by the window and the
resolution floor, not by the sample. Reporting it as resolved and then reasoning from it is
circular (Notebook §5h).

**Lever.** Report the floor alongside the number and call the width an **upper bound**. A
projection through a finite illuminated thickness alone forbids a resolved boundary width in
DFXM projection images — do not claim one.

## Some rocking curves look doubled or bimodal

symptom: split.bimodal
coord: statistics

**Test.** First count **points per FWHM** using the per-pixel width (see width_scale_mismatch):
below ~12, a moment-based bimodality coefficient measures curve *broadness*, not bimodality, and
has no discriminating power. Then run the enrichment test that can exonerate: count multi-peak
pixels among flagged **and unflagged** populations at matched SNR. If both come back at the same
rate (40 % vs 40 % in one case; enrichment 0.96× in a better-sampled revival), the statistic is
describing the estimator, not the sample. Also check whether the flagged pixels' scan windows
**bracket their own maxima**. If enrichment is well above 1 with adequate sampling and interior
maxima, the structure is real.

**Cause.** Three compounding artifacts (Notebook §5l): undersampling; a fixed rocking window
reused across a raster while θ_B drifts, so the statistic maps *which scans caught their peak*;
and a peak-finder whose prominence is judged against global side-minima rather than local photon
noise. Check the pipeline's design first, though — edge frames are sometimes the dark reference
by intent, in which case a peak "never in the last frames" is the design, not a defect.

**Lever.** Use a real two-component fit with a parametric-bootstrap null and multi-start, judged
on **raw likelihood** (χ²/dof penalises the nested model by the dof ratio at an exact tie). Match
populations by 1:1 nearest-SNR pairing rather than quantile bins, which fail silently when one
class dominates. Gate on enrichment, never on the statistic's own p-value.

## A centroid / orientation map is reported as robust to lineshape choice

symptom: centroid_lineshape_untested
coord: orientation

**Test.** Refit with an **asymmetric** lineshape (split pseudo-Voigt, or any skewed profile) and
compare the fitted centres. A *symmetric* alternative — pseudo-Voigt, Lorentzian, Pearson VII —
shares a symmetry with the Gaussian and cannot move a centre by construction, so agreement with
one proves nothing. Measure the curves' asymmetry against an exactly-symmetric control generated
at the measured noise: if |asymmetry| is at the control level and the asymmetric fit moves centres
by far less than your per-pixel precision, the centroid genuinely is robust and this entry does
not apply.

**Cause.** Skew is the one misspecification that shifts a centroid. Testing against a symmetric
model is a test that cannot fail. Doing it properly turned a claimed 0.045 mdeg into a median
1.98 mdeg (p95 10.67), 12 % of the FWHM — a 44× error (Notebook §5i).

**Lever.** Quote the centre shift under an adequate asymmetric lineshape as the robustness
figure. Beware the optimiser too: a bounded `trf` can silently return the initial width on these
curves — use `method="lm"` on a log-width parameterisation with multi-start and verify against a
second optimiser.

## Lengths are self-consistent but the absolute scale is suspect

symptom: scale_magnification_unverified
coord: geometry

**Test.** Derive the magnification from **≥ 2 independent optical routes** — objective/image
distances in the parameter record, focal length and working distance, any physical mask or
knife-edge calibration — and compare against the constant the analysis actually uses. Agreement
across routes exonerates the constant. Disagreement clustered on one side of it, especially by a
factor near 2, does not. Then check f(E): a refractive objective specified at one energy has
f ∝ E², so recompute f at the energy actually used and compare against the working distance — if
f exceeds it, the nominal geometry is not what was realised.

**Cause.** A hardcoded magnification propagating into every length. A factor of exactly 2 is the
signature of a calibration that counted a line pair as one feature (Notebook §7e).

**Lever.** Read the scale from the optical record and name the file in the report (rule 10, rule
15). Do not accept a **focus** diagnostic as evidence of scale — a perfectly focused image can
carry the wrong µm/px.

## An offset or intensity difference appears between two scan groups

symptom: intergroup_offset
coord: geometry

**Test.** Build a **zero-displacement null**: identical footprints, no real shift, the same
photon counts and pedestal, and see whether it reproduces the observed offset, its scatter, and
its spatial trend. If the null reproduces them, the effect is an estimator artifact. If the
observed effect clearly exceeds the null, look next at the *optics* columns of the motor
metadata — not just the sample columns — and at r(offset, log counts): a strong count dependence
(+0.92…+0.97) with the partial correlation on position going **negative** means you are measuring
counts, not position. A genuine displacement survives the null and shows no count dependence.

**Cause.** Several, all mundane (Notebook §7d): an intensity centroid on a mostly-pedestal frame
**measures the pedestal**; the objective moved between groups, so the *image* shifted by
δ(1+M) without the sample moving; there was no flux monitor to normalise a between-group
intensity comparison; and a correlation with the raster's **fast axis** is inseparable from a
per-cycle mechanical effect such as stage settling after fly-back.

**Lever.** Subtract the pedestal before any centroid. Read the optics motor columns. For an
intensity comparison, find the monitor column — and if none exists, halt and ask (README STOP
table), because the normalisation cannot be recovered from the archive. Test a fast-axis
correlation by dropping the scans immediately after each fly-back.

## A claimed strain wave must be told from a non-periodic phase

symptom: periodicity_vs_scalar
coord: strain

**Test.** Do **not** decide it with a scalar direction-coherence or band-SNR number — those do
not collapse on a non-periodic field. Use two structural tests instead. (i) **Real-space
autocorrelation** of the field in a clean window: a genuine wave has repeating side-lobes at ±λ
(and ±2λ…); a non-periodic phase (smooth strain blobs) gives a single central peak decaying
monotonically with no side-lobes. (ii) **Reflection rotation**: measure the wavevector direction
from two different reflections — a real strain wave's wavevector rotates *with the crystal axes*
(e.g. ~90° between orthogonal reflections), whereas a lab-frame processing/optics artifact gives
the *same* direction for both. If the autocorrelation shows side-lobes **and** the direction
rotates with the reflection, it is a real wave; if either fails, it is not.

**Cause.** A per-window band-SNR / coherence gate measures whether a window has *a* dominant
Fourier mode, which a smooth aperiodic gradient also has (its power piles up at the low-frequency
band edge). On the tetragonal null the band-SNR was *higher* than on the real wave and coherence
R = 0.63 was non-trivial (Notebook §8c), so the scalar is permissive, not diagnostic.

**Lever.** Report the autocorrelation periodicity and the reflection-rotation as the evidence, not
the coherence scalar. Keep the scalar only as a coarse pre-filter. If only one reflection exists,
the rotation test is unavailable and the autocorrelation carries the whole claim — say so.

## A reflection that should see the strain shows no contrast, or the wave period looks halved

symptom: wrong_channel_or_half_period
coord: strain

**Test.** Two distinct causes, each with a clean check. (i) **No contrast in a θ,2θ map** — compute the
shift direction for this reflection: $\Delta\mathbf Q=\mathbf H\,\mathbf G_0$. If $\Delta\mathbf Q$ is
**transverse** to $\mathbf Q$ (a cube-axis reflection viewing a shear), the strain (longitudinal)
channel is genuinely zero and the signal is all **tilt** — re-image as a **θ-rock (mosaicity)** map and
it appears. If $\Delta\mathbf Q$ has a longitudinal component and there is still nothing, the cause is
elsewhere (pedestal, alignment). (ii) **Period looks like λ/2** — check whether the intensity image was
taken at the **exact Bragg peak**: the response there is 2nd-order, so a wave of period λ images at λ/2.
Step onto the **weak-beam flank** (off-Bragg) and the true λ returns. A θ,2θ *strain map* (a COM, linear)
does not have this doubling; only fixed-θ intensity does.

**Cause.** (i) A single reflection senses a strain as **d-spacing (θ,2θ) or tilt (θ-rock) depending on
its direction** relative to the strain axes; a reflection can be blind in one channel (Notebook §9a).
(ii) At exact Bragg the fixed-θ intensity is an even function of the deviation → frequency doubling
(Notebook §9b).

**Lever.** Match the channel to the reflection (diagonal reflection → θ,2θ strain; cube-axis → θ-rock
tilt), and read the intensity wavelength on the weak-beam flank, not at the peak. If you need the true
strain field, use the θ,2θ COM (linear) rather than a fixed-θ intensity.

# XRD-CT diagnosis reference

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

Symptom → discriminating test → cause → lever. Keyed by *symptom*, not by step — the step that
produced a symptom is rarely the step you are on.

**Every entry carries a test that can come back the other way.** Before re-investigating, read
[`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) §5: four results are recorded there as refuted or invalid,
one as **downgraded**, and three inferences as withdrawn. Three of the entries below exist
because an attractive hypothesis was wrong.

---

## Local symptoms

Emitted by **this technique's own procedure**, not by `beamreport`'s generic diagnostics,
which key off per-observation residuals against declared coordinates. A ring matched as a
singlet when it is a doublet, a background estimator riding up the peak flanks, or a
point-group table with the wrong order are real and useful, and nothing generic will ever
detect them — so they are declared here rather than renamed into the wrong shape.

Every row names where the check lives. A symptom nothing produces is dead text that reads
as coverage, which is exactly what the generic vocabulary existed to prevent.

| symptom | emitted by |
|---|---|
| `detection.unverified` | `scripts/odf_positive_control.py` — plants discrete crystallites at the measured contrast and asks whether the fit recovers them; a null with no positive control is unverified, not negative |
| `pattern.static_in_omega` | the ω-dependence check in that entry: `n_s·ẑ = cos θ_B sin η` carries no ω term, so an axial fibre is *necessarily* static in ω and cannot be called instrumental on that basis |
| `ring.doublet_as_singlet` | `count_maxima` on the background-subtracted azimuthal mean of the window, which requires a *dip* between maxima rather than two high points |
| `background.rides_peaks` | the background estimate at ring centres against a straight interpolation across the peak, or against truth on a synthetic. **Not `scale.suppressed`**: that is "a recovered magnitude far below an independent expectation", and the suppressed *area* is the consequence here, not the symptom — the entry keys off the estimator's own behaviour, which nothing generic observes, and its discriminating test is a background comparison, not an area one |
| `field.explained_by_polynomial` | `explained_by_polynomial` on the recovered field, plus whether it is centred on the rotation axis |
| `symmetry.wrong_order` | assertion against `midas_hkls.point_group.EXPECTED_ORDER` plus a closure-under-composition check |
| `polefigure.metric_convention` | Monte-Carlo pole figure from sampled orientations against the closed-form operator (`tests/test_texture_kernel.py`) — symmetry tests cannot catch it |
| `compute.thread_oversubscribed` | host `uptime` load against the core count |

---

## A per-voxel texture map looks structured and plausible

symptom: null.not_cleared
coord: texture

**This is the most dangerous symptom in the doc set**, because the failure mode looks exactly
like success.

**Test.** Three of them, all cheap, all able to come back the other way.

1. **The ladder.** Fit a *uniform null*, then **one globally shared** texture, then per-voxel
   (`fit_uniaxial_ladder`). If the global rung already captures most of the improvement, you
   have a sample-average texture, not a map. If the null is within a few percent of per-voxel,
   you have nothing.
2. **The polynomial `r²`** (`explained_by_polynomial`). Above ~0.5 the "map" is a smooth field
   — absorption, an illumination gradient, a geometry error. This has retracted a result in
   this project **three times**.
3. **Cross-ring agreement.** The same physical texture must be seen by every vetted ring.
   Pairwise correlation below ~0.3 between per-ring maps means whatever is being fitted is not
   a single ODF.

**Cause.** At low peak-to-background the extracted **area** is a small difference of large
numbers and inherits the background model's error, which is independent per frame. A model
with 23–61 parameters per voxel will absorb that error as texture and produce a structured
map. Measured: 36 % area scatter at 2 % contrast with *no* planted structure.

**Lever.** Measure peak/background (phase 2). Drop to the 4-parameter uniaxial model. Run
`scripts/odf_positive_control.py` at the measured contrast. If the control says recovery fails
there, the map is not reportable at any spatial scale — say so and report strain instead.

---

## The fit returns a null. Is there no texture, or can we not see it?

symptom: detection.unverified

**Test.** `scripts/odf_positive_control.py --contrasts <your measured contrast>`. It plants
**discrete crystallites** (no Legendre polynomials, no squared modulus — nothing the fit uses),
lays real peaks on a background at that contrast, and runs the same extraction and fit.

It scores **two separate claims**, and they support different conclusions:

* **Detect** (global rung finds planted texture) holds at your contrast → a null on the
  *global* rung is a statement about the **sample**.
* **Resolve** (per-voxel `S` correlates with the plant) fails → a null on the *per-voxel* rung
  is **not** interpretable; it is consistent with texture the reconstruction cannot localise.
  Report a sample-average bound, not a map.
* Neither holds at your contrast → the null is an **SNR limit** and says nothing at all.

Measured on the DAC Ti geometry: **detect YES (23–31 %), resolve NO (|corr| ≤ 0.35)**.

**Check the control's own plant quality first.** It prints normals per azimuthal bin and the
implied Poisson noise on the *planted* pole figure. Only ~6 % of normals land near the
diffraction condition, so a modest crystallite count leaves single-digit counts per bin and a
plant that is 40 % noise. A failure at that plant quality says nothing about the pipeline —
the script warns and reports INCONCLUSIVE rather than REFUTE.

**Also compare the two background modes.** `--background both` runs an exactly-known pedestal
and the real estimated-background chain. The **gap between them is the background-model error**
(measured: best |corr| 0.67 → 0.35), and it does not average down with more frames the way
Poisson noise does. A bound quoted from the `known` arm alone is optimistic.

**And check the refute line against the control.** With a realistic background, planted texture
at peak/bg = 0.02 yields only **2.6 %** improvement — *below* a 5 % refute line. At low contrast
a fixed 5 % line would reject real texture.

**Cause of a genuine null, on the one real dataset taken this far:** the sample really had no
coherent azimuthal texture in the vetted phases. The global rung bought 0.11 % against 24–31 %
for planted texture at comparable contrast.

**Lever.** If detection works and the fit still nulls, report the **bound** with its softener,
and move to strain.

---

## Strain magnitudes are absurd — tens of thousands of microstrain

symptom: scale.inflated
coord: strain

**Test.** Count the **live azimuths** per ring: `RingExtraction.live_mask(min_snr=3)`. Then
recompute the spread as a **5–95 % inter-percentile range and a MAD**, not as a peak-to-peak.

**Cause.** Gating on the ring's **median** SNR lets dead azimuthal bins through, and
peak-to-peak over η is a max-minus-min that a handful of them dominate. Measured: one
reflection read **107,410 µε** and another **638,282 µε** (11 % and 64 % strain) from exactly
this.

**Lever.** Apply the per-bin mask (`live_mask`), then report inter-percentile range and MAD.
Real numbers on that dataset after the fix: 3352–7324 µε with MAD 1228–3354 µε, and six
reflections agreeing — which is what made them credible.

---

## Azimuthal intensity varies strongly, and it looks like a pole figure

symptom: trend.periodic
coord: eta_deg

**Test.** Two of them.

1. **Window width.** Extract the area in a narrow window and again in a much wider one. If the
   azimuthal variation collapses when the window widens, you are seeing the peak **move**, not
   change amplitude.
2. **Radial halves.** Correlate the inner and outer half of the ring. **Anti-correlation is
   the signature of movement**; a genuine intensity change correlates positively. Measured
   −0.72 on the CeO₂ scan.

**Cause.** Under differential stress the peak position varies with azimuth,
`d(ψ) = d₀[1 + (1 − 3cos²ψ)Q(hkl)]` (Singh's lattice-strain relation — it is how DAC
differential stress is *measured*). A fixed radial window converts that movement into
azimuth-dependent intensity: **fake texture**. In a DAC this is physics, not an artefact.

**Lever.** Widen the window until the half-correlation is no longer negative, or switch to
**peak-fitted** areas.
And read the movement as the signal it is — that is your strain.

---

## The CeO₂ (or any powder-standard) null shows structured texture

symptom: null.not_cleared
coord: texture

**Test.** Sweep `L`. If the residual is **flat in `L`**, truncation is not the cause. Then
predict the absorption effect quantitatively before invoking it.

**Cause — identified 2026-08-20, provisional.** On the 11-ID-C CeO₂ scan it is **finite
crystallite counting**: a 3.7 % random-per-ω component plus a 0.9 % ω-locked floor whose amplitude
scales as chord^(−0.572) against the 1/√N_grains prediction of −0.50. Residual geometry, background
model, flat-field, shot noise, capillary absorption **and peak movement** were each excluded by a
test that could have come back the other way (`LAB_NOTEBOOK.md` §5b-ter).

**Peak movement was the earlier suspect and is wrong** — a window wider than ~2× FWHM has an area
invariant to sub-pixel movement, so peak-fitting cannot fix an area structure.

**Lever.** Check the grain-count gate (`ENVELOPE.md` §0a) before blaming the pipeline. If the
powder null fails for *counting* reasons, there is nothing in the analysis to fix — the levers are
finer powder, a larger gauge volume, or more ω (and more ω removes only the random part, never the
ω-locked floor). If it fails for any other reason, fix that before real-sample texture.

---

## An azimuthal pattern does not vary with ω. Is it instrumental?

symptom: pattern.static_in_omega
coord: eta_deg

**You cannot tell from that.** This entry exists because the inference was made and had to be
withdrawn.

**Test.** Not the ω behaviour. Use the ladder and the polynomial `r²` instead (first entry).

**Cause of the confusion.** `n_s·ẑ = cos θ_B sin η` carries **no ω dependence**, so a fibre
about the rotation axis produces an azimuthal pattern that is *necessarily static in ω*.
Static in ω is what an axial fibre looks like. The converse limit is real though: if the
sample's unique axis is **not** the rotation axis, the texture *does* vary with ω and the
uniaxial model cannot fit it by construction.

**Lever.** Establish the loading / unique-axis geometry from the experiment. If it is not
axial, either free the axis (the forward model supports it) or state that the null is scoped
to an axial fibre only.

---

## Absolute strains are all offset by a similar amount

symptom: systematic.common_offset
coord: strain

**Test.** Refine the sample-to-detector distance **against the data** — ring positions across
many rings over-determine it. Compare against the metadata and against the beamline
calibration *separately*; do not assume they agree.

**Cause.** A wrong distance is an absolute strain-scale error. On the 11-ID-C CeO₂ scan the
metadata said 1600 mm, the beamline calibration 1579.5 mm, and the data required **1632 mm** —
both stored values wrong.

**Lever.** Use the measured distance. Or report **relative** strain referenced to the median
over azimuths, which is `strain_from_centroid`'s default precisely because of this.

---

## A "clean singlet" ring gives unstable areas and centroids

symptom: ring.doublet_as_singlet

**Test.** `count_maxima` on the background-subtracted azimuthal mean of the window. It requires
a **dip** between maxima (prominence), not just two high points.

**Cause.** A sub-pixel ring assignment matches a **doublet** as one line. hcp Ti α(101) held
maxima at 381.6 and 393.6 px with a dip between and was assigned as a single ring at 393.6 px
— through four analyses.

**Note on a height-only test.** Counting local maxima above half height alone calls a clean
singlet a *triplet* at low contrast, because photon noise on the peak plateau produces several
samples that each clear half height and each exceed their neighbours. A genuine doublet's dip
sits ~60 % below its lobes; noise ripple is a few percent. That gap is wide, so the prominence
criterion is robust rather than tuned.

**Lever.** Exclude multiplets. Do not fit them; report which rings were dropped and why.

---

## The background estimate rides up under every peak

symptom: background.rides_peaks

**Test.** On a synthetic with a known background, compare the estimate at the ring centres
against truth. Or on real data, check whether the fitted background at a ring centre exceeds a
straight interpolation from the ring-free radii on either side.

**Cause.** A rolling low percentile over blocks comparable to the peak width sits on the peak
**flank**. With an 8-px FWHM peak in a 30-px block that is ~27 % of the block.

**Lever.** `background_from_ring_free` — mask the known rings, take a low percentile of the
*ring-free* samples per azimuth, interpolate across the peaks. Not `rings.rolling_baseline`,
which is for ring *finding*, where the bias does not matter.

---

## The texture map is smooth and radial

symptom: field.explained_by_polynomial
coord: texture

**Test.** `explained_by_polynomial`. And ask whether the field is centred on the rotation axis.

**Cause.** Absorption, illumination, or a centring error. A radially smooth "texture" field is
the classic instrumental signature. On the DAC Ti scan a global cubic explained only 5.8 % of
the recovered `S`, which ruled *this* cause out — the field there was scatter, not a smooth
artefact. Both outcomes are informative; the test distinguishes them.

**Lever.** If smooth: fix the geometry or the absorption correction; do not report it as
texture. If scatter: the data does not support a per-voxel map (first entry).

---

## A symmetry group has the wrong number of elements

symptom: symmetry.wrong_order

**Test.** Assert the order against `midas_hkls.point_group.EXPECTED_ORDER`, and check the group
closes under composition.

**Causes, both real.**

* **Keying on quaternions.** `q` and `-q` are the same rotation and the sign tie-break is
  unstable exactly at `w = 0` — the 180° elements, of which every one of these groups has
  several. 432 closed to **28**. Key on the **matrix**.
* **Discarding improper operations.** Friedel makes the measurement centrosymmetric, so the
  recoverable symmetry is the **Laue** group and improper operations map to `-R`. Discarding
  them under-symmetrises the **73** space groups that have improper operations but no
  inversion centre: Pm (#6) → order 1 instead of 2, Pmm2 (#25) → 2 instead of 4.

**Lever.** `midas_hkls.proper_rotations_from_space_group(spec, lattice)`. It raises rather than
returning a plausible non-group if the lattice is inconsistent with the space group.

---

## A hexagonal or non-cubic pole figure is subtly wrong

symptom: polefigure.metric_convention

**Test.** Compare a Monte-Carlo pole figure from **sampled orientations** against the
closed-form operator (`tests/test_texture_kernel.py`). Symmetry tests cannot catch this: a
convention error leaves the result symmetric and smooth.

**Cause.** In cubic, `(hkl)` doubles as a Cartesian direction. **In every other system it does
not** — the reciprocal metric matters, and in hcp the angle between (10-10) and (0001) is not
what the raw index triple suggests. A cubic shortcut applied to hcp is silently wrong.

**Lever.** `SymGSH.families(hkl)` with the lattice set, or
`midas_hkls.plane_normals(hkl, group, lattice)`. `gsh.hkl_family` is the **cubic-only**
shortcut.

**And when you build that MC comparison:** the Monte-Carlo side counts normals inside a finite
cap, so it is cap-**averaged**. Evaluating the model **pointwise** compares two different
quantities and biases the fitted slope low on sharp features — measured 0.969 pointwise
against 1.008 cap-averaged. Smooth the model over the same cap. A slope near 0.97 with
corr ~0.999 is this, not a broken operator.

---

## A batch of per-voxel fits never finishes

symptom: compute.thread_oversubscribed

**Test.** `uptime` on the host. Compare against the core count.

**Cause.** Unpinned BLAS threads. Each process spawns as many threads as there are cores;
15 processes on a 96-core host drove **load 437** with nothing finishing in 40 minutes.

**Lever.** `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on every worker.

**And when you clean up:** `pkill -f "foo.py"` matches its own ssh command line and kills the
session (exit 255). Use `pkill -f "[f]oo"`.

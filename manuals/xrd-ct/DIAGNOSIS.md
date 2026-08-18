# XRD-CT diagnosis reference

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

Symptom → discriminating test → cause → lever. Keyed by *symptom*, not by step — the step that
produced a symptom is rarely the step you are on.

**Every entry carries a test that can come back the other way.** Before re-investigating, read
[`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) §5: four results are recorded there as refuted or invalid,
one as **downgraded**, and three inferences as withdrawn. Three of the entries below exist
because an attractive hypothesis was wrong.

---

## A per-voxel texture map looks structured and plausible

**This is the most dangerous symptom in the doc set**, because the failure mode looks exactly
like success.

**Test — three of them, all cheap, all able to come back the other way.**

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

**Test.** Sweep `L`. If the residual is **flat in `L`**, truncation is not the cause. Then
predict the absorption effect quantitatively before invoking it.

**Cause — diagnosed, NOT proven.** On the 11-ID-C CeO₂ scan the residual was flat from L=6 to
L=10, absorption was excluded (predicted 0.000 %), and the two radial halves were
anti-correlated at −0.72. **Leading suspect: peak movement in a fixed radial window** (see
the entry above). The fix — peak-fitted areas — has **not yet been run**, so this remains a
diagnosis rather than a conclusion.

**Lever.** Do not proceed to real-sample texture on a pipeline whose powder null fails. Fix
the null first, or restrict the claim to strain.

---

## An azimuthal pattern does not vary with ω. Is it instrumental?

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

**Test.** `explained_by_polynomial`. And ask whether the field is centred on the rotation axis.

**Cause.** Absorption, illumination, or a centring error. A radially smooth "texture" field is
the classic instrumental signature. On the DAC Ti scan a global cubic explained only 5.8 % of
the recovered `S`, which ruled *this* cause out — the field there was scatter, not a smooth
artefact. Both outcomes are informative; the test distinguishes them.

**Lever.** If smooth: fix the geometry or the absorption correction; do not report it as
texture. If scatter: the data does not support a per-voxel map (first entry).

---

## A symmetry group has the wrong number of elements

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

**Test.** `uptime` on the host. Compare against the core count.

**Cause.** Unpinned BLAS threads. Each process spawns as many threads as there are cores;
15 processes on a 96-core host drove **load 437** with nothing finishing in 40 minutes.

**Lever.** `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on every worker.

**And when you clean up:** `pkill -f "foo.py"` matches its own ssh command line and kills the
session (exit 255). Use `pkill -f "[f]oo"`.

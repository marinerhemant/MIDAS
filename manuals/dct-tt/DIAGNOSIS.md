# Diagnosis — symptom → discriminating test → cause → lever

Keyed by **symptom**, not by pipeline step, because that is how trouble arrives. Every entry
carries a test that can come back the other way.

---

## 1. The grain map looks clean and space-filling

**The most dangerous symptom in the technique**, because it is exactly what dilation of a
sparse measurement produces — and it looks like success.

**Test.** Compute the fraction of labelled voxels that came from a measured core versus from
dilation, and the fraction claimed by exactly one grain.

**Expected on real data.** In the adopted map here, **86 % of labelled volume was dilation**
and only **~22 %** of the domain was uncontested at *any* threshold.

**Lever.** Report both fractions. If you need more measured volume, note that lowering the
threshold converts uncontested → contested, not uncontested → measured: contested-voxel
assignment does not separate from a null.

---

## 2. Indexing yields lots of grains and they look plausible

**Test.** ω-scrambled null: permute ω across the spot table, re-run the identical pipeline.

**Discriminating outcome.** At a 4× looser margin, real and null indexed **identically**
(2761/2902 seeds, completeness 0.250 both). At the adopted margin, real 488–814 seeds, null
**0**.

**Cause.** Tolerance set by hand rather than from the null.

**Lever.** Set the acceptance above the null's *maximum* completeness (here 0.09 vs a null max
of 0.069). Remember MIDAS FF margins are **µm against the ring radius** — at an 880 µm radius,
150 µm is a ~10° window.

---

## 3. Seeds refuse to cluster into grains

**Test.** Cross-check your misorientation against `midas_stress.misorientation` on a pair you
believe is nearly coincident.

**Discriminating outcome.** 29.8° reported for a pair genuinely 0.33° apart.

**Cause.** Symmetry operator in the wrong position. With `v_sample = U v_crystal`, equivalents
are `U·S`, so the misorientation is `Uaᵀ Ub S` — **S on the right**.

**Lever.** Fix the frame. Here: 367 clusters from 488 seeds → 205 after the fix.

---

## 4. Grain centres form a smeared cloud with no sample shape

**Test.** Does the centre cloud reproduce the sample cross-section — compact, bounded by the
beam height?

**Cause.** Solving for one shared **lab** position across a grain's pairs. Each Friedel pair
flashes at its own ω, so that is the wrong model.

**Lever.** Include `Rz(σω)` in the design matrix. Residual 52 → 41 µm here, and the cloud
became a recognisable cross-section — a physical check, not a numerical one.

---

## 5. Forward-simulated spots are "nearly right" but never quite match

**Test.** Overlay predicted and observed spots for one grain you trust, and check
(a) whether the pattern is the exact **inversion** about the beam centre, and (b) whether
predictions exist for the *whole* 360° or only half.

**Cause.** Either the antipode convention (`flip_y` **cannot** express it, so no flip flag
fixes it) or an ω→frame conversion without wrapping, which silently discards half a 360° scan.

**Lever.** Fix both before trusting any assignment. Three retractions here rest on these two.

---

## 6. Reconstructed grains come out absurdly small

**Test.** Histogram the reconstructed volume. Are a few voxels far above the grain's own
level?

**Cause.** Thresholding at a fraction of the maximum, with streak artefacts setting the
maximum.

**Lever.** Otsu. `0.5 × max` reported **60 µm grains as 6 µm**.

---

## 7. Grains fragment into multiple bodies

**Test.** Inject candidate causes into a validated phantom **one at a time**, at the real
measured value, holding everything else fixed.

**Discriminating outcome here.** Spot count (29 spots) → 2.0 % fragmentation; position error
(12 µm) → 1.4 %; silhouette roughness → ruled out; **ray-direction scatter (0.10°) → 20.0 %**,
against 44 % observed.

**Cause.** Intragranular orientation spread — the grain is not one coherent diffracting
object.

**Lever.** None, and check this before building one: the **coherent** (refinable) part of the
residual is 0.031° against **0.173°** incoherent, ratio 0.18. Refining orientations will not
help. Report the fragmented grains as position+orientation only.

---

## 8. Nothing finishes, and the machine is loaded

**Test.** `torch.get_num_threads()` inside a worker.

**Cause.** Every worker grabbing every core.

**Lever.** `torch.set_num_threads(1)` and `OMP_NUM_THREADS=1`. Here: 20 workers × 64 threads
finished **zero** grains in 32 min; one thread each finished all 121 in ~3 min.

---

## 9. A TT intragranular field fits beautifully — on the mask you gave it

**Test.** Re-run on a **deliberately wrong support** and compare.

**Discriminating outcome.** Wrong support scored **0.810** against the true support's
**0.860**, and the two fields agreed at NCC **+0.940** on 79 % overlap.

**Cause.** A wrong support scores well largely because it *contains* the sampled region. The
data determine the field; the domain constrains it far less than it feels.

**Lever.** Never present a field without this control. It does not invalidate the field — it
bounds what the field is evidence *for*.

---

## 10. A TT field looks smooth and convincing

**Test.** Compare against a **polynomial ceiling** — the best a smooth global function of
stated order achieves on the same planted residual while carrying no per-voxel information.

**Discriminating outcome.** Recovery beat the ceiling only over ~one octave: **1.2–2.0 µm**.
Above ~2.8 µm a low-order polynomial did *better*.

**Cause.** Smooth structure is not per-voxel information.

**Lever.** Report the transfer function with the field. A field without it is unbounded.

---

## 11. A recovered field's amplitude matches truth suspiciously well

**Test.** Inject sub-pixel registration jitter and watch amplitude, not just correlation.

**Discriminating outcome.** At 0 / 0.1 / 0.3 px, correlation fell **0.246 / 0.176 / 0.042**
while `|H|/|H_true|` *rose* **0.24 → 0.39 → 1.06**.

**Cause.** Misregistration manufactures magnitude; the fit compensates by inventing amplitude.

**Lever.** Register deliberately, and treat suspiciously-correct amplitude as an artefact
until shown otherwise.

---

## 12. A model-selection sweep gives an incoherent answer across λ

**Test.** Does the loss at the returned iterate equal the best loss actually evaluated?

**Cause.** Adam without an annealed schedule ends *above* its own best iterate. The effect on
one reconstruction is < 1 % — **but the argmin then moves with the hyperparameter**, so
cross-λ comparison is meaningless.

**Lever.** `lr_schedule="cosine"` with `eta_min=0` (a floor of `lr/100` still leaves the final
steps at full rate) and `return_best=True`. Note `info["settled"]` is budget-relative and is
**not** a convergence certificate.

---

## 13. Every length looks wrong by a constant factor

**Test.** Re-derive the effective pixel from a known physical length, on **two independent
axes**.

**Cause.** The header pixel is the **sensor** pixel, not the imaging pixel.

**Lever.** Measured here: header off by **6.65×** on one dataset, and a metadata pixel off by
**2×** on another. The two-axis agreement (1.4 % here) is what makes the new value
trustworthy.

---

## 14. Orientations read from a reference grain map are wrong at large misorientation

**Test.** Convert a known rotation and check the **absolute angle**: `θ = 2·atan(|r|)`.

**Discriminating outcome.** **Below midas-stress 0.9.0**, `rodrigues_to_orient_mat` gave
60°→**80°**, 90°→**180°** — right axis, wrong angle, inflated by `1/cos²(θ/2)`. Small angles
look fine (5°→5.010°), which is why it survived casual checks. Fixed in 0.9.0.

**Lever.** Check the installed version first; substituting the old converter moved a 74-scan
tilt residual from 0.043° to 26.5°. Then check the separate **convention** question — read
deposits with `midas_dct_tt.rodrigues_to_crystal_to_sample`.

---

## 15. A count against a threshold changes when nothing physical changed

**Test.** Recount with a small tolerance (`>= x - 1e-9`).

**Discriminating outcome.** Counting `γ ≥ 90°` exactly gave **39** and **34** of 55 for two
algebraically identical bases; with a tolerance, **40** and **42**.

**Cause.** At a saturation point the quantity piles up *at* the threshold, so a bare `>=`
counts bit-exact rounding.

**Lever.** Always compare with a tolerance, and say what it was.

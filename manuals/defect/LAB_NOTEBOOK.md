# Lab notebook — evidence, retractions, and open questions

One notebook per campaign, started on day one. Retractions decay fastest and backfilling them
does not work.

**Read this before re-investigating anything.** Several of the entries below are results that
were believed, written down, and then killed — none of them by new physics.

---

## The reference sample

Deformed **Cu-9at%Al single crystal**, FCC SG 225, a = 3.6356 Å (~0.6 % expanded over pure
Cu; Al in solid solution). FF-HEDM, 10 Y-layers. Deformed into **two Σ3-twin orientation
families** — parent and deformation twin — each a ~18° mosaic.

This is the only material family with real-data anchors in the suite. `ENVELOPE.md` §9 says
how far that licenses you.

---

## E — Established (survived a test that could have failed)

**E1 — The 9R polytype is real, discrete, and sits on the SHARED {111}.**
A discrete `n·G/3` satellite ladder along ⟨111⟩, n = ±1..±6: `G/3, 2G/3, 4G/3, 5G/3` in
FCC-forbidden gaps, bracketed by `111` (n±3) and `222` (n±6) fundamentals.
Three independent methods agree that the axis is the ⟨111⟩ **shared** by parent and twin:
* assumption-free cross-tabulation, null-baselined: **55.8 % / 56.9 %** of satellite
  intensity within 5° of *both* a parent and a twin ⟨111⟩, against a **1.9 %** null — ~30×.
  Parent-only 5 % (= null), twin-only 3 % (**below** null).
* shared-⟨111⟩ check: **99/99** Σ3 pairs, max-satellite axis == shared axis, 0.0°.
* full sample, all 10 layers: **1018/1025** Σ3 pairs at 0.0°; per-layer "both(shared)"
  55.6–56.0 %, 28–31× enrichment. Uniform through the bulk.

**E2 — The satellites are reflections, not Ewald-sampled relrods.**
Each is compact in ω (σ ≈ **0.6°**) with a Friedel mate at ω+180.

**E3 — `5G/3` is a satellite, not a `220`.**
Nearest `220` of any of 232 grains is **0.27 Å⁻¹** away in full 3-D. Radial coincidence
alone is worthless here. Corroborated by the forbidden shells (`G/3`, `2G/3`, `4G/3`) being
**empty** across all grains.

**E4 — Two Ewald crossings per reflection, verified.**
Predicted diffraction ω from `q_lab·x̂ = −|q|²/2k` matches observed ω to **<1°**. A pipeline
assuming one crossing silently discards half a 360° scan.

**E5 — The ω-doublet is real: two 9R variants.**
Same-rung, same-side pairs separated in **ω**. See R3 — this one was wrongly retracted first.

**E6 — HCP at grain boundaries: NOT DETECTED.**
Two independent tests. A self-consistent-lattice scan over 232 grains × 4 c-variants found at
most **1** HCP reflection family per orientation where the null reaches **2**. Single-`|q|`
coincidences exist and mean nothing. A clean negative, and it stands.

**E7 — Complete attribution of the scattered intensity, with the test that it is complete.**
Per-voxel classification, ω-summed: Bragg **64.8 %**, asterism **31.5 %** (~72 % lattice
rotation / 28 % strain), 9R satellites **0.04 %**, inter-Bragg **~0.6 %**, smooth
background **~1.6 %**, unexplained floor **≤ 0.2 %** and featureless. The
discriminating test: the off-ring residual collapsed monotonically 1.24 → 0.12 % as the
per-spot asterism cutoff widened — a real extra phase would not shrink under that.

> **AMENDED 2026-09-01.** The ~0.6 % bin was originally labelled "⟨111⟩ SF relrod". That
> label is withdrawn: it is a **distance-bin occupancy**, not an attribution. The bins are
> real and the closure test stands; naming the inter-Bragg bin after the fault rod does not,
> because rod and asterism are **not separable** in this sample (`ENVELOPE.md` §1a, R12).
> Quote the bin, not the defect.

**E8 — No parent/twin dislocation asymmetry. A null that is a result.**
Asterism rotation and strain widths identical parent ≈ twin to **1–3 %**. The twin's higher
stress and stored energy is an **elastic-anisotropy projection** of equal strain, and was
deliberately not headlined. `ENVELOPE.md` §12.

**E9 — No FCC-forbidden intensity.** With the correct FCC rule, forbidden/allowed ≈ **0.000**
= the random off-lattice control, on **0 of 248 grains**. No anti-phase boundary, no
selection-rule-breaking defect. (Contrast R8.)

**E10 — The ⟨111⟩ relrod seen directly in raw frames.** Back-projected bright pixels land on
integer hkl at median **0.11**; PCA direction **3.7° from ⟨111⟩**, 1-D dominant (singular
values [49.4, 9.6, 0.6]), threading 111→200→220→311→400; bridge pixels median **16.6°** from
⟨111⟩ against 28° random. One deformed Σ3-twinned grain that the indexer had fragmented into
~21 "grains" (14 sub-grains within 12° + 7 at 57–60°).

**E11 — Microstructure and depth uniformity.** 248 grains, median diameter 65 µm. Over 10
layers: grain count 252 ± 19 (CV 7 %), volumetric strain 0.148 ± 0.023 %, {111} texture MRD
4.9 ± 0.4, median misorientation 14.9 ± 1.2°. Genuinely uniform — see R7 for the bug that
nearly manufactured this.

**E12 — The ingest front end reproduces a hand-built chain exactly.**
On an independent La₃Ni₂O₇ DAC dataset: mask **byte-identical** (0 differing px of 2.48 M),
polar background **bit-identical**, spot list **615 of 615** sub-peaks to **0.0008 px**.
Not in the test suite; recorded in that project's `PORT_TO_MIDAS.md`.

---

## P — Provisional (do not let these become facts)

**P1 — Lateral coherence L ≈ 5–10 nm.** From satellite FWHM 0.075–0.12 Å⁻¹. **Lower bound**,
mosaic-contaminated. `ENVELOPE.md` §2.

**P2 — Defect volume fraction f ≳ 1 %.** From gap-free labels only (311 sat/Bragg = 0.7 %,
511 = 1.7 %). **Lower bound**; gaps remove signal asymmetrically. `ENVELOPE.md` §7.

**P3 — Modulation type = displacement / spacing relaxation.** Order pattern rises with n
(ratio-to-n1 ≈ [1.00, 0.45, 1.92, 4.42]), reproduced across two layers. Single method.

**P4 — Modulation tilt β ≈ 3°, β ≈ 4–8 %/layer.** Single method, not null-tested.

**P5 — Dislocation densities are RELATIVE.** The package's cubic-`H²` anisotropy correction
disagrees in absolute terms with the reference re-analysis's per-grain radial-breadth fit.
Ratios and orderings are supported; absolute m⁻² is not. `ENVELOPE.md` §8.

---

## R — Retracted (must not reappear)

**R6 — The whole v0.2 deck. RETRACTED — wrong phase.**
The sample was analysed as θ-Al₂Cu single crystal; it is **FCC Cu, a = 3.6356 Å**, a deformed
sub-grain mosaic. Everything downstream inherited it:

| claim | retracted value | corrected |
|---|---|---|
| phase | θ-Al₂Cu single crystal | FCC Cu, a = 3.6356 Å |
| microstructure | single crystal | 248 low-misorientation sub-grains, MRD 5 |
| residual elastic strain | −5 %…+11 % volumetric | ~0.15 % vol median, ±1 % |
| dominant fault plane | (103)/(114)/(411) | ⟨111⟩ rods in ~40 % of grains, not dominant |
| Burgers vector | 4.29 Å | **2.571 Å** |
| ρ_disloc | 5 × 10¹¹ m⁻² | **~10¹²–10¹³ m⁻²** |
| fault probability α | 0.16 | ~0.005 typical, ~0.022 faulted |
| forbidden/allowed | 0.46 "smoking gun" | ≈ 0 — see R8 |

*The lesson is not "check the phase". It is that a phase error is not one wrong number: it
propagates into the Burgers vector, the selection rules, the fault plane and the strain, and
each of those looked independently plausible.*

**R7 — "Uniform with depth" was nearly an artifact of reused data.**
The FF pipeline's zip-convert stage (historical path
`midas_ff_pipeline.stages.zip_convert`; the tree has since been reorganised and that path no
longer exists) reused the **first layer's** converted data for every subsequent layer in a
multi-layer batch — the shared detector object's `zarr_path` persisted and the skip-check
matched a stale path. Ten layers would have looked perfectly uniform
because they were the same layer. Found, fixed (ff-pipeline 0.3.2), and the depth profile
re-run with a verified unique md5 per layer. Uniformity across depth is exactly the claim a
data-reuse bug fabricates.

**R8 — "forbidden/allowed = 0.46, a smoking gun." RETRACTED — wrong space-group rule.**
It applied an **I4/mcm c-glide** rule, from the phase the sample does not have, whose
"forbidden" positions **include FCC-allowed reflections**. It was measuring ordinary Bragg
intensity. With the correct rule the forbidden positions are empty (E9).

**R9 — "α = 0.16." Self-retracted — a Hendricks–Teller local-minimum artifact.**

**R10 — "Parent-specific 9R", "L_9R 41 vs 31", "ΔPDF parent 1.95×". RETRACTED — projection
geometry.** All three were artifacts of the same OM@⟨111⟩ projection; the demonstration
showed corr(θ, L) = **−0.62**. Per-variant ΔPDF stays retracted.

**R11 — "The ⟨111⟩ relrod is not there." RETRACTED — frame-of-reference error.**
Judged on the ω-**sum**, which superimposes every grain's rod at every orientation and smears
it into a ring. Per-frame, per-cluster back-projection reinstated it (E10). *A second
retraction that was itself wrong — see R3.*


**R1 — "99.8 % intensity-budget closure" from the auto-classifier. RETRACTED.**
The classifier database was **~18 % wrong** against ground truth. Root cause: at high `|q|`
the 18°-mosaic node cloud of 232 grains makes `5G/3`, `220`, asterism and fault-rod tails
overlap in **every scalar feature**, so a scalar decision tree cannot classify them. Closure
was arithmetic, not attribution. *Everything computed before the classifier is unaffected.*

**R2 — "No 9R present." RETRACTED — it was a method bug.**
A 15° cone diluted the discrete (<5°) satellites, and the "random" isotropic baseline was
contaminated by the 992 textured grain ⟨111⟩ — signal in the denominator. Both faults push
the same way: false negative. See B1: the same flaw is still live in the package.

**R3 — "The doublet is mosaic, not two variants." RETRACTED — wrong diagnostic.**
The retraction looked at perpendicular/azimuthal q, saw mosaic, and concluded smear. The
signature is an **ω-split**. The doublet is real (E5). *A retraction can itself be wrong;
this one cost a real result for several days.*

**R4 — "The parent has an independent 9R population, 9.2× enriched." RETRACTED.**
Superseded by the clean attribution in E1: parent-only is at the null (0.9×) and twin-only is
below it (0.5×). The 9R is on the shared direction, not on either family's own planes.

**R12 — "The fault relrods carry 18.7 % of the diffracted intensity." RETRACTED — a
connected-component label sum is not a budget.**
The 45 hand-curated `FAULT_RELROD` cc3d components do carry 18.67 % of the thresholded
intensity, but each is a **reflection merged with its streak**, not a rod: **97.1 %** of that
intensity lies within 0.15 Å⁻¹ of an allowed FCC node, and **all 45** components have their
brightest voxel within **0.02 Å⁻¹ of an allowed shell**, with bimodal |q| histograms across two
adjacent shells. The "streak |q|" in the catalog is a centroid *between* two Bragg cores — a
position nothing occupies. *3-D connectivity at threshold merges a peak, its asterism and any
streak leaving it into one label, and the label is then named after whichever feature the
analyst noticed. This is a different failure from R1: not a bad classifier, a segmentation
unit that is not a physical object.*

**R13 — "Asterism (~31 %) is the largest non-Bragg component, not the fault rods." RETRACTED —
bin substitution, and it was the replacement offered for R12.**
The relrods were scored **far-only** (0.53 %) and asterism **whole-shell** (30.87 %), then
ranked against each other. Scored the same way at the same tolerance: relrod non-Bragg
**10.111 %** against all asterism labels' **10.119 %** — a tie, with relrod *larger* at every
coarser tolerance (2.34× at 0.20 Å⁻¹). And the "asterism" shell is 31.5 % asterism-labelled,
31.0 % relrod, 32.7 % Bragg. *The correction reproduced the error it was correcting: a mixed
bin renamed after the feature in mind. **Two populations must be scored with the same
accounting before they may be compared.*** Killed by `/verify` (statistics + artifact lenses)
before it reached the manuscript.

**R14 — "The near-Bragg halo is not directionally ⟨111⟩." NOT a retraction of a positive — a
retraction of a NULL that had no power.**
An I-weighted median of **29.1°** to the nearest ⟨111⟩ against **28.95°** isotropic read as a
clean absence. It was zero power: a **planted** ⟨111⟩ rod at a real node, through the identical
machinery, scores **29.02°** against **28.34°** for a planted isotropic blob. Cause in §1a.
*A null is only evidence of absence once a positive control shows the test can see the thing
present.* See O5.

**R5 — "The 9R is on a secondary {111}, 60–77° off the twin plane." RETRACTED.**
A mislabel of the ~70°-from-loading-z angle as "off the twin plane". Confirmed wrong by three
methods. It is the standard ITB-9R on the composition plane.

---

## D — Deprecated, with replacements (checked in the code 2026-09-01)

**D1 — `polytype.satellite_intensity.polytype_satellite_enhancement` — DEPRECATED, raises.**
Along-axis-sum over mean-random-direction-sum. It fails in **two opposite directions**, which
is why no threshold rescues it:

* **Inflation (false positive).** The ratio is dominated by **voxel count**, not intensity:
  the satellite tube is densely populated while random directions at `|q| = G/3` hit
  near-empty space. On the reference sample this turned a real ~5× per-voxel excess into
  **700–1600×** — a ~140× inflation.
* **Suppression (false negative).** In a *textured* sample the "random" directions land on
  real ⟨hkl⟩ of indexed grains, putting signal in the denominator. That is the mechanism of
  R2.

It now **raises by default**; `allow_deprecated=True` reaches the historical path for
reproduction only, and `tests/test_polytype.py` pins that it refuses.

**Replacement: `polytype.satellite_excess.satellite_radial_excess`.** A **count-normalised**
per-voxel mean-intensity ratio between on-axis and off-axis voxels **at the same `|q|`**, so
detector sampling density and texture cancel. It also carries the null the old metric lacked:
a real periodic polytype is **peaked at the thirds** (`G/3`, `2G/3`) and **dips to background
at the half-integers** (`G/6`, `G/2`, `5G/6`), whereas a continuous ISF relrod rises
monotonically inward from the Bragg. The verdict comes back with the numbers.

**D2 — `polytype.lamella_thickness.per_grain_lamella_thickness` — DEPRECATED, raises.**
Same escape hatch, same reason class. Pinned by a test.

*Note for the ledger: an earlier version of this notebook recorded D1 as a live unfixed bug,
on the strength of a project note. The code says otherwise — the guard and the replacement
both exist. Verify a "known bug" against the tree before repeating it.*

---

## O — Open questions

**O1 — Bulk or interfacial film?** Is the 9R distributed through the bulk, or is it a thin
(1–3 nm) interfacial film at the twin boundary? **Not decidable by FF-HEDM** — parent bulk,
twin bulk and the boundary all scatter to the same shared ⟨111⟩. Needs pf-HEDM or DFXM.
`ENVELOPE.md` §4. *This is the real open question of the campaign.*

**O2 — Absolute dislocation density.** Blocked on pinning the contrast factor and the
anisotropy convention against an independent method (P5).

**O3 — Does any of this transfer off FCC-with-planar-faults?** One material family is the
entire real-data base. Hexagonal polytypes and incommensurate modulations are plausible and
unproven.

**O5 — Can a fault-rod fraction be measured on this sample at all?** Not by any route tried:
distance fails (every group is 94–100 % near a node) and direction fails (no per-voxel node
attribution is possible — `ENVELOPE.md` §1a). It is **not** established that the rods are weak
or absent; it is established that the fraction is **not obtainable here**. Would need fewer or
better-resolved orientations, i.e. a different measurement (pf-HEDM / DFXM), or a sample whose
mosaic the indexer does not fragment ~100×.

**O6 — Three limits, or one?** Bulk-vs-boundary on a shared direction (§4), the rod fraction
(§1a), and the retracted per-variant dislocation densities are plausibly **one** limit:
*far-field cannot attribute a diffuse feature to one member of an over-fragmented mosaic.*
If that is right it should be statable as a single precondition — a test on grain count,
mosaic width and node spacing — and checked once at phase 0 instead of discovered three times.
Worth an hour before the next campaign.

**O4 — The new front-end modules have no real-data test in the suite.** `ingest`,
`completeness`, `rod_profile` and `residual_decomposition` are synthetic-tested here and
validated by hand on one external dataset (E7). Fold that into the suite as a fixture.

**R15 — "The per-grain `c` ordering reproduces under independent re-analysis." RETRACTED —
four lenses, four refutations, and the fallback went with it.**

Six orientation groups from a 621-position raster gave `c` medians spanning ~0.5 %, and the
ordering was stable when the pipeline was re-run with different gating (978 → 1281 domains).
Presented as an independent robustness check. It was not one, on three counts:

* **Not independent.** 933 of the 1281 second-run domains were the same crystal at the same
  position as a first-run domain; 95.4 % of the first run survived, 766 with a byte-identical
  `c`. An overlap-matched null predicted 1.20 discordant pairs and the comparison scored 1 —
  zero independent information. (`ENVELOPE.md` §14.)
* **Matched by rank, not identity.** Groups were paired by size rank. Matched by orientation,
  two of six swapped and the ordering inverted. Four of six values were quoted with the other
  two marked `--`; those two were exactly the pair that broke it. Same failure mode as an
  earlier a/b episode — dropping the inconvenient half of a list.
* **An artifact reproduces just as well.** One identical cell planted on every group's real `U`
  and real hkl list, with only the measured origin offset, reproduced at Kendall 0.73 with a
  span of 0.21–0.27 % and was *more* stable across the two runs than the real data.
  (`ENVELOPE.md` §13.)

**The fallback was withdrawn too.** A coarser two-level split — `{g1,g4}` vs `{g2,g3,g5}`, 35 mÅ
apart, holding independently in both halves at p = 5e-43 and p = 2e-5 — looked like the safe
statement and was reported as such. Its 0.18 % separation sits **inside** the planted control's
0.21–0.27 %. A split-half cannot see an orientation-locked bias, because the groups *are* the
orientations. Kruskal–Wallis H = 233.5, p = 1.9e-48 across the six groups does not help: it
tests whether they differ, and the artifact makes them differ.

**What is left.** Nothing quotable about per-group `c` on this sample. Excluded as mechanisms,
which narrows the alternative usefully: origin offset (a free-origin refit makes the observed
span *grow*, 0.0884 → 0.0969 Å), detector roll 0.25–1°, ω-step error 1 %, isotropic Lsd/λ, and
2θ-quadratic distortion — all ≤ 0.006 Å. No linear q-space systematic reaches the extreme group:
χ² = 146 (5 dof) for identical crystals, 127 for a free-axis uniaxial deviatoric, 129 for a
general symmetric-traceless ε.

**Two real bugs fell out of the refutation**, both now fixed and both pinned by tests
(`test_refine_to_convergence_*`): seed-cell anchoring, and a reported cell that was never fitted
to the reported reflections. `DIAGNOSIS.md` 19, `phase-2-index.md`. Note where they were found —
neither was caught by 920 passing tests, because both were in the *composition* of correct
primitives by an out-of-tree script. `rows.refine_to_convergence` exists so the composition is
in the package with the postcondition asserted.

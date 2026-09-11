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

*Every entry below names its generator (provenance audit, 2026-09-10). Recovered scripts are in
`~/Desktop/analysis/demk/fcc_reanalysis/recovered_from_transcripts/`; `scripts/` means `~/Desktop/analysis/demk/fcc_reanalysis/scripts/`;
gdata paths are under `/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/`, mounted on copland and
chutoro only. Where a recorded number disagreed with its own output, the output wins and the
entry says so.*

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
*Source:* `scripts/cross_tab_9r.py`, `scripts/cross_tab_9r_alllayers.py`, `twin_plane_check.py`,
`twin_plane_check_alllayers.py`; logs `demk_cc3d/cross_tab_rerun.log`, `cross_tab_alllayers.log`
and `demk_9r_phaseunique/cross_tab_9r_L2346.json` on gdata; transcript 34738a30, 2026-06-29.
1018/1025 re-run 2026-09-10, identical. **Read the pair counts as n = 1, not 99 or 1018:** the
crystal has one twin variant, so every pair measures the same orientation relation (R21).
**Double diffraction, checked 2026-09-10 — it does not explain the 9R.** Parent/twin double
diffraction does put reflections at exactly ±1/3, ±2/3, ±4/3 and ±5/3 of G111 on the shared ⟨111⟩
(`midas_defect/dev/paper/verify_cc54aa5246ae/own_checks/dd_sumlattice.py`), so on-axis ladder
*positions* alone could not rule it out — and the older argument that a clean G/2 control excludes
it does not follow, because double diffraction makes no half-order spots. What rules it out is the
9R-unique **off-axis** reflections. Every parent and twin reciprocal-lattice point sits at a multiple
of G111/3 along the shared axis, and so does every double-diffraction sum (5,891 points, deviation
7e-15; `dd_cannot_make_ninths.py`), so the 9R (0 1̄ 4) at 4/9 (|q| = 3.12 Å⁻¹) and (1̄ 0 5) at 5/9
(|q| = 3.28 Å⁻¹) are unreachable by it. Both are present: they are in the manuscript's off-axis 9R
table, and the 2026-09-10 verification lenses found rod intensity at 4/9–5/9 on the shared-(111)
rods (55 % of all fcc-forbidden gap intensity, none on rods rotated 30°). **The 9R is established,
and so are the ⟨111⟩ fault rods (E10).** Double diffraction can still add intensity at the on-axis
ladder positions, so it bears on on-axis intensity ratios, not on whether the 9R is there.
*Correction, 2026-09-10:* an earlier version of this paragraph said ordered 9R was not shown. That
misread the rod data and ignored the off-axis table.

**E2 — The satellites are reflections, not Ewald-sampled relrods.**
Each is compact in ω (σ ≈ **0.6°**; 0.57–0.72° over five labels) with a Friedel mate at ω+180.
*Source:* `recovered_from_transcripts/E2_omega_kabra.py` (originally a /tmp script; figure
`figures/omega_kabra.png`), transcript 34738a30, 2026-06-25.

**E3 — `5G/3` is a satellite, not a `220`.**
The nearest `220` of any of 232 grains is **0.22 and 0.27 Å⁻¹** away in full 3-D for the two
`5G/3` labels (243 and 326). Radial coincidence alone is worthless here. Corroborated by the
forbidden shells (`G/3`, `2G/3`, `4G/3`) being **empty** across all grains.
*Source:* `recovered_from_transcripts/E3_braggprob14.py` (ran on copland), transcript 34738a30,
2026-06-24. *Correction, 2026-09-10:* this entry said 0.27 Å⁻¹, which is label 326 only; label
243 sits at 0.2158.

**E4 — Two Ewald crossings per reflection, verified.**
Predicted diffraction ω from `q_lab·x̂ = −|q|²/2k` matches observed ω to **within 1.0°** (worst
of eight crossings: predicted −166.0°, observed −165.0°). A pipeline assuming one crossing
silently discards half a 360° scan.
*Source:* `recovered_from_transcripts/E4_ewald_two_crossings.py` (an inline `python -c`),
transcript 34738a30, 2026-06-30. *Correction, 2026-09-10:* this entry said <1°.

**E5 — The ω-doublet is real.**
Same-rung, same-side pairs separated in **ω**, fully resolved. See R3 — this one was wrongly
retracted first. *That the two peaks are two orientation variants is supported but NOT
established:* the preregistered test refuted the fixed-offset alternative and was read
INCONCLUSIVE on the variant claim itself (R18).
*Source:* `scripts/doublet_user.py`, transcript 34738a30, 2026-06-29. *Correction, 2026-09-10:*
the heading said "two 9R variants" as established.

**E6 — HCP at grain boundaries: NOT DETECTED.**
Two independent tests. A self-consistent-lattice scan over 232 grains × 4 c-variants found at
most **1** HCP reflection family per orientation where the null reaches **2**. Single-`|q|`
coincidences exist and mean nothing. A clean negative, and it stands.
*Source:* `recovered_from_transcripts/E6_hcp_scan.py` (the lattice scan) and `hcp_weak2.py`
(z = −1.09 / −0.36 at the 10-10 / 10-11 positions), transcript 34738a30, 2026-06-25. The scan
**cannot be re-run as recovered** — its input `/tmp/fammodel.npz` is gone — and the
`hcp_test.py` recovered alongside it is not what produced these numbers.

**E7 — Complete attribution of the scattered intensity, with the test that it is complete.**
Per-voxel classification, ω-summed: Bragg **64.78 %**, asterism **31.54 %** (~72 % lattice
rotation / 28 % strain), 9R satellites **0.04 %**, inter-Bragg **3.64 %** — four channels that
sum to 100.00 %. Of the inter-Bragg channel, **1.49 %** of the total is structured residual after
a radial-background subtraction. The discriminating test: the off-ring structured residual
collapsed monotonically **1.22 → 0.20 → 0.12 %** of the total as the `distB` cutoff (distance
from Bragg) widened from 0.20 to 0.70 to 1.00 — a real extra phase would not shrink under that —
leaving a far-field floor of **0.204 %**.
*Source:* `scripts/whats_unknown.py`, `asterism_per_grain.py` (CORE = 0.025, OUTER = 0.10),
`asterism_floor.py`, `onring_offring_residual.py`; logs `whats_unknown.log`, `floor.log` and
`onoff.log` in gdata `demk_9r_phaseunique/`; transcript 34738a30, 2026-06-23. *Correction,
2026-09-10:* this entry gave inter-Bragg as ~0.6 % and a smooth background of ~1.6 %, bins that
summed to 98.7 %. The log's inter-Bragg channel is 3.64 %, and no run printed 1.6 %. The collapse
was quoted as 1.24 → 0.12 %, splicing `onoff.log` (1.243 % at its default cutoff) onto the
`floor.log` sweep.

> **AMENDED 2026-09-01.** The inter-Bragg bin (recorded here as ~0.6 %; the log gives 3.64 %) was
> originally labelled "⟨111⟩ SF relrod". That
> label is withdrawn: it is a **distance-bin occupancy**, not an attribution. The bins are
> real and the closure test stands; naming the inter-Bragg bin after the fault rod does not,
> because rod and asterism are **not separable** in this sample (`ENVELOPE.md` §1a, R12).
> Quote the bin, not the defect.

**E8 — No parent/twin dislocation asymmetry. A null that is a result.**
Asterism widths agree parent ≈ twin: strain (radial) to **0.3 %** (σ_r 0.03348 vs 0.03359) and
rotation (tangential) to **3.1 %** (σ_t 0.03264 vs 0.03165), over 106 parent and 142 twin grains. The twin's higher
stress and stored energy is an **elastic-anisotropy projection** of equal strain, and was
deliberately not headlined. `ENVELOPE.md` §12.
*Source:* `scripts/asterism_per_grain.py` (tight run) → gdata `demk_9r_phaseunique/asterism_tight.log`,
transcript 34738a30, 2026-06-23. *Correction, 2026-09-10:* recorded as "1–3 %".

**E9 — No FCC-forbidden intensity.** With the correct FCC rule, forbidden/allowed ≈ **0.000**
= the random off-lattice control, on **0 of 248 grains**. No anti-phase boundary, no
selection-rule-breaking defect. (Contrast R8.)
*Source:* `scripts/c3_forbidden.py` → gdata `demk_diffuse/c3.log` and `c3_forbidden_L2346.npz`
(2026-05-20). The script was refactored on 2026-05-28 and the current version has **not** been
re-checked against 0.000 and 0/248.

**E10 — The ⟨111⟩ relrod seen directly in raw frames.** Back-projected bright pixels land on
integer hkl at median **0.11**; PCA direction **3.7° from ⟨111⟩**, 1-D dominant (singular
values [49.4, 9.6, 0.6]), threading 111→200→220→311→400; bridge pixels median **16.6°** from
⟨111⟩ against 28° random. One deformed Σ3-twinned grain that the indexer had fragmented into
~21 "grains": 12/21 within 10° and 14/21 within 15° of one reference, the other seven at
56.6–60.0°.
*Source:* `scripts/measure_streak_q.py`, `measure_bridge.py`, `streak_chord_test.py` (figures in
`figures/`), transcript 34738a30, 2026-06-23. *Correction, 2026-09-10:* the grouping was
recorded as "14 sub-grains within 12° + 7 at 57–60°"; the output reads as above.

**E11 — Microstructure and depth uniformity.** 248 grains, median diameter 65 µm. Over 10
layers: grain count 252 ± 19 (CV 7 %), volumetric strain 0.148 ± 0.023 %, {111} texture MRD
4.9 ± 0.4, median misorientation 14.9 ± 1.2°. Genuinely uniform — see R7 for the bug that
nearly manufactured this.
*Source:* `scripts/d_depth_profile.py`, `scripts/a_microstructure.py`; re-run 2026-09-10,
identical. The original stdout survives in no transcript.

**E12 — The ingest front end reproduces a hand-built chain exactly.**
On an independent La₃Ni₂O₇ DAC dataset: mask **byte-identical** (0 differing px of 2.48 M),
polar background **bit-identical**, spot list **615 of 615** sub-peaks to **0.0008 px**.
Not in the test suite; recorded in that project's `PORT_TO_MIDAS.md`.
*Source:* `recovered_from_transcripts/E12_port_mask_and_full_chain.py`, `E12_port_background.py`,
`E12_port_spots_615.py`, Argo transcript ef79a90a, 2026-09-01. "2.48 M" is 1679 × 1475 pixels,
computed, not printed by any run.

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

---

**R16 — "The Friedel intensity asymmetry in L5 is physical." RETRACTED — it is segmentation
volume, and a detector edge-response invented to explain the residual was refuted twice.**

*2026-09-08.* Many Friedel pairs in the gap-fixed components had unequal integrated intensity,
which Friedel's law forbids. The asymmetry is real in the label sums (0.32–1.82) and absent in
the physics: **intensity per voxel sits at median 0.965**, within ±10 % for nearly every pair.
A segmentation-free re-integration moved the median volume ratio 0.880 → **0.991** and cut the
log-spread **5.6×** (sd 0.237 → 0.042), stable over 8 support variants. Mechanism identified
and predictive: at a fixed absolute threshold `V_A/V_B = (amplitude ratio)^|s|` with local
slope |s| = **0.8–5.4**; it predicts 1.86 against 1.75 observed for pair 315/328.
`ENVELOPE.md` §16, `known-limits.md`.

Dead-strip truncation is real for 4 of 11 pairs (266, 267, 263, 258 all begin at row 848, one
past the 831–847 strip); symmetric censoring moves those 0.80–0.88 → 0.94–1.04. The gapfix
product is innocent — **0 labelled voxels inside any dead strip**.

**What died twice:** a claimed 10-row detector edge zone, then its 2-row replacement. The
row-profile estimator behind both is refuted by a comb-phase null — slide the same 7-tooth
17-row comb onto gap-free rows and it returns the same value or lower **one time in five**
(p = 0.197, spanning 0.69–2.39) — and by applying it to the same detector's column gaps, where
it returns **5.1**, which is impossible. It is also internally inconsistent: at distance 1 the
summed ratio is 0.864 while the above-threshold count ratio is 1.093 and the mean per pixel is
0.995. **There is no measured detector edge response on this instrument.**

**R17 — "There are four ladders." Then "there are two." BOTH REFUTED — and the search cone was
manufacturing the members.**

> **VOID NOTE (2026-09-10):** every angle here came from a flat, untilted detector transform (no tilts, no
> distortion). It fails a known-zero test — a grain's 111 against its own 222 comes out 4.2–5.2° instead of 0
> (the corrected transform gives 0.61°) — and it makes the apparent angle between two fixed directions scale
> with |q|. The four- and two-ladder models were artefacts of it, and so is the "angular spread grows with |q|"
> measurement. The radial positions (n·G/3 to better than 1 %) are unaffected, and the search-cone trap still
> stands as a method warning. With the corrected transform the satellites form Friedel/crossing quartets with a
> doublet at n = 1 and 2 only, 6.1° and 6.5° apart (`RESULTS_L5_three_open_points.md`). Source: the VOID section
> of `midas_defect/dev/paper/CHECKPOINT.md`.

*2026-09-08.* The radii are excellent: every picked component lands at `n·G/3` to better than
1 %. The directions are not collinear through the origin.

Four ladders died because candidates C and D have **no unique members** — they sit 3.73° and
7.48° from axis A and a 5° search cone around them re-collects A's and B's own components.
**Every member was borrowed.** *Generalise this:* a search cone around a candidate direction
will re-collect a neighbour's members, so a member count is not evidence that the direction
exists. Require members unique to the axis.

Two ladders died on the geometry: fitting each axis from the tightest rungs (n=4, 5), a line
through the high rungs misses even its own Bragg anchors by 1.9–2.8°, and the high rungs put A
and B **13.89°** apart while their n=1 spots are **8.5°** apart. Rigid ladders cannot do that.

What survives is a measurement with no model: the angular spread grows **faster than linearly**
with |q| (8.5° / 12.7° / 14.8° at n = 1 / 4 / 5, i.e. spread/n = 0.146 / 0.219 / 0.255), and at
n=1 there are four discrete directions at 2.03 / 4.43 / 8.16 / 10.51° from axis A. Constant-angle
ladders hold spread/n fixed; a fixed transverse offset makes it fall. Neither describes this.
**Cause unknown, and it is in tension with R18** — see `phase-4-rods.md`.

**R18 — "The 9R doublet is two orientation variants." PARTLY — the alternative is refuted, the
claim itself is INCONCLUSIVE by its own registration, and it has not been verified.**

*2026-09-07.* `PREREGISTER_doublet_scaling.md` fixed the discriminator before the measurement:
two variants give **Δω constant** with rung order, a fixed transverse q-offset gives
**Δω ∝ 1/n**. Measured 6.75° at n=1 and 6.50° at n=2 in **both** L5 and L7, against 6.5°/3.25°
predicted. H0 refuted. Valley depths 0.00 and 0.02–0.03 — the peaks are fully resolved, not a
shoulder. Measure Δω, not the q-separation: two referees measured the q-separation and
disagreed (0.175–0.205 vs 0.128 at n=2) because it is not independent of Δω.

The registration was read **INCONCLUSIVE** (skill log, 2026-09-07 02:14): its own pre-declared
inconclusive branch applied, because the high orders (n=4, 5) are single-peaked and all the
leverage rests on n=1 and n=2. What is solid is narrower than "two variants": **H0 (Δω ∝ 1/n)
is refuted outright** — it required 3.25° at n=2 and both layers gave 6.50°. H1 is consistent
with two rungs, untested beyond them, **not through `/verify`**, and consistent with the corrected-transform directions of the satellite components, 6.1° and 6.5°
apart at n = 1 and 2 (`RESULTS_L5_three_open_points.md`; the q-space "tension" once cited from R17 came
from a void transform).

*Correction, 2026-09-10:* this entry first read "STANDS", committed in `d9c4cd15`.

Note this is the **third** correction of R3's territory, which was itself a wrongly-retracted
positive. Read R3, R17 and R18 together before touching the doublet.

**R19 — "The §3.4 order-scaling exponent is 1.95, so the modulation is displacement."
RETRACTED — and so was the first explanation of why.**

*2026-09-07.* A re-extraction gave 0.10 against the recorded 1.95. The first write-up called
that extraction-dependence spanning both physical answers. `/verify` the same night (claim
`44a53a5cb198`) **refuted that claim as stated**, while the withdrawal of §3.4 survived on
better grounds: both numbers are unconverged artifacts of one knob, the tube radius; the
manuscript value's 95 % CI was [−0.91, +4.80]; and the stated cause, unequal ω coverage, was
wrong — coverage is 0.974 at n=1 and 1.000 elsewhere.

The lasting finding is that **the test was mis-posed**. n=3 is the 111 and n=6 is the 222, so
the rungs are satellites of different parents: a displacive modulation predicts 0 : 1 : 1 : 4
(n=1 vanishes at first order) and a compositional one an apparent exponent of −0.63. Neither is
2 or 0. In the converged data n=1 is the **second-brightest** rung, which a first-order
displacive modulation does not produce — open, and it needs its own preregistered test.
`ENVELOPE.md` §17.

*Correction, 2026-09-10:* the first version of this entry, committed in `d9c4cd15`, repeated the
refuted framing and blamed R16's threshold mechanism for the spread. Neither holds.

**R20 — "The satellite Ewald-crossing offset is a calibration residual, not satellite-specific."
RETRACTED — 4 of 4 lenses. And the reading it replaced is withdrawn too.**

*2026-09-09.* The predecessor claim — that Bragg reflections fail to reproduce the satellite
crossing rotation, so the difference is in the reflections themselves — was **withdrawn**
first: it compared angle-about-origin across populations differing 2× in |q|, with unmatched
segmentation, an axis undefined at the noise floor, and one of its four Bragg controls
mis-paired (see `phase-4-rods.md`, the 220 case).

The replacement claim then died on all four lenses. **The support sweep (h ≤ 30 px) enclosed
all five satellite groups and zero of five Bragg groups** (which need h = 36–100), so the two
arms were different measurements; with enclosing supports the ordering **inverts**. A
`slope < 0.10` selector introduced after seeing the data correlates with the compared quantity
at Spearman **+0.900** and keeps exactly the three smallest of five. The headline ordering
holds in **60 of 120** configurations and flips with the centring convention, which moves
Bragg 111 by **2.49×**. Two of my own group definitions were wrong: 220 g1/g2 are the **same**
grain (59.3° is the fcc {220}∧{220} angle of 60°), and its Friedel mates **do** exist — ids
59, 66, 68, 71.

**What survives — PROVISIONAL, not itself through `/verify`:** a column-antisymmetric
calibration residual is **quantitatively excluded** as a sufficient cause. Only BC_y has
leverage, at `2·k·px/Lsd` = 0.0191 1/Å per pixel analytically (finite differences 0.0171–0.0191
over ten reflections, mean 0.0188), **common-mode to ±6 %**, so it
predicts one |dq| for all of them; the observed spread is a factor **12.6**, and a bounded
11-parameter fit leaves 95–112 % of every offset standing. `ENVELOPE.md` §18.

**Net on the crossing offset: not truncation at n=1, not calibration, and not demonstrated to
be satellite-specific.** It stays out of the manuscript in either direction until the Bragg
cubes are re-cut at ±110 columns (111 is currently not measurable), a support-convergence
criterion is registered in advance, and the centring convention is decided on its merits.

**R21 — "§3.3 (the 9R lies on the Σ3 twin composition plane) is refuted." THE RETRACTION WAS
WRONG — but the claim it protected is PROVISIONAL and weaker than first written. The 9R itself is
not in doubt.**

*Reported refuted 2026-09-07; corrected; put through `/verify` 2026-09-10 (claim
`cc54aa5246ae`: all four lenses back (statistics UNCERTAIN, reproduction SURVIVES, physics UNCERTAIN, artifact UNCERTAIN): PROVISIONAL).* **Do not reinstate the refutation from `MEETING_BRIEF_2026-09-07.md` or
`REEVALUATION_2026-09-07.md`:** what died on 09-07 was the manuscript's *statistic* (99/99 against
a 25 % baseline), not the claim.

The measurement, from two inputs that share no file:

| quantity | derived from | value |
|---|---|---|
| ladder axis | G/3 diffuse shell alone, 12,246 voxels | [−0.2973, −0.8654, −0.4034] |
| shared Σ3 ⟨111⟩ | 99 parent–twin pairs alone | [+0.2884, +0.8661, +0.4083] |

They agree to **0.58°** as axes. The computation was an unsaved inline script until 2026-09-10;
it is now `midas_defect/dev/paper/repro/recovered_plane_angle_20260907.py`, re-runs to the same
numbers under midas_defect 0.1.7, and an independent pipeline rebuilt the ladder axis from the raw
detector zip to four decimals. `repro/claims.py` still hard-codes 0.58 as a literal.

**What verification did to the claim:**

* **The precision is about 1–4°, not 0.58°.** Bootstrap 95 % intervals run [0.13, 1.88]° over
  twin grains and [0.20, 4.19]° over grains and rotation blocks together; the ten layers give
  0.23–2.06°.
* **The discrimination is about one in seven, not 99 pairs.** The crystal has exactly seven
  distinct ⟨111⟩ axes — one shared, three parent-only, three twin-only. The 99 pairs are one
  orientation relation (63 distinct twin grains), and random parent × twin pairings reproduce the
  normal within 0.58° in 24.8 % of draws. The ladder sits 0.17° from the shared axis and about 70°
  from the other six, so *which* axis it is, is clear.
* **"Nine of twelve ⟨111⟩ carry zero" does not survive.** The twelve directions in the original
  output are the twin's four ⟨111⟩ counted about three times each: the three "composition-plane"
  hits are 8.7–9.3° apart and their shares sum to 297 %. The parent's three other ⟨111⟩ were never
  examined, and 22 of the 24 predicted G/3 positions for the six non-shared axes hold no voxel in
  any frame, so whether they were observable at all is unknown.
* **"Composition plane" is assumed.** What is measured is the shared Σ3 ⟨111⟩. It is the
  composition-plane normal only for coherent {111} twin boundaries, which far-field data does not
  measure; and parallel is not "lies on".
* **Parallel, not located.** The 0.58° is the mean direction of eight compact spots that each sit
  1.94–4.23° off the normal. It shows crystallographic parallelism only; whether the 9R sits at the
  boundary or in the bulk on a shared direction is not decidable from these data (`ENVELOPE.md` §4).
* **"Zero" means "below 30 counts".** This layer's raw zip is pre-thresholded at 30 counts, while
  the ladder's own voxels have a median of 79, so absence on the other axes is censored, not
  measured.
* **Double diffraction does not explain the 9R** (E1): it can only produce reflections at multiples
  of G111/3 along the shared axis, and the 9R-unique off-axis (0 1̄ 4) and (1̄ 0 5) at 4/9 and 5/9
  are present. It can add on-axis intensity; it is not a reason to doubt the ladder.
* **A frame-convention trap sits beside this number:** `geometry.py`'s docstring had the z sign
  reversed (corrected 2026-09-10). Rebuilding q from the old docstring moves the ladder axis 47.9°.

**For the manuscript:** replace the 99/99 statistic with the two-vector agreement, quoted as
"within about 1–4°; one orientation relation; one of seven candidate axes". The 9R itself needs no
such hedge: its off-axis reflections exclude double diffraction as its origin.

*Why this entry exists:* it is the fourth retraction in this notebook that was itself wrong —
**R3** and **R11** are two of the others. A refutation is a claim and needs the same gate as the
positive it kills, and so does the correction of one: the first version of this entry
(`d9c4cd15`) called §3.3 "better supported than the manuscript's own version", quoted "nine of
twelve carry zero", and had never been verified.

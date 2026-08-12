# Phase 5 — Report with provenance

> Part of the **DFXM doc set**; spine is [`README.md`](README.md). Every quantitative claim
> names the file and command that produced it and is re-derivable (rule; the whole point of
> the doc set).

---

## 5a. What the report must carry

| section | content | provenance |
|---|---|---|
| Configuration | material, reflection, energy, θ_B, 2θ, Λ, t/Λ regime, ε_ref, resolution widths | §1 calls, from metadata (not filenames) |
| Reduction | per-reflection orientation (and θ-scan strain) maps; p95 spread | §2 — **on background-subtracted frames**, cross-checked vs `darling` |
| Accuracy | injection-recovery gain; per-pixel σ; Poisson MC ratio | §2d — **not** a round-trip |
| Tensor | either F(x) with identifiability + covariance, **or** the registration wall | §3 — say which |
| Strain honesty | refraction treated as gauge or applied; t/Λ regime stated | §4a–b |
| Capability results | typing / defect model / design — **labelled simulation-grounded** | §4c–e |

State plainly, once and prominently, **which results are real-data and which are
simulation-grounded** (traps table). A reader must not have to assemble that from scattered
caveats.

## 5b. What a healthy number looks like (ranges, not thresholds)

These were measured on the ID03/ID06 campaign; outside those conditions they are not a
specification.

| quantity | observed | condition |
|---|---|---|
| pedestal share of centroid weight | ~98.5 % | raw ID03 frames — **subtract before the moment** |
| `darling` agreement, background-subtracted | corr 1.0, RMS ~1e-7° | same-estimator check, ID03 |
| injection-recovery gain | 0.9998–1.0000 | four ID03 scans |
| per-pixel orientation σ | ~2 mdeg | ID03, ~20–40× finer than the 80 mdeg step |
| 111 intragranular spread p95 | ~45 mdeg | ID06 111 |
| multi-reflection NCC (unregisterable) | 0.43 at search edge | ID06 111↔002 — **the wall** |
| kinematic strain validity | t ≲ 0.15–0.3 Λ | past it, dynamical forward (cross-model verified) |
| refraction gauge | 144 µε (Cu 002, 0.71 Å) | absolute offset, not a relative-map field |
| integrated ÷ per-pixel rocking FWHM | **2.6–2.7×** | one archived scan at 1.000 mdeg steps — quote the per-pixel width for sampling |
| detector gain, integrating sCMOS | `var = 2.23·y + 149` | photon transfer, pedestal removed — **measure per detector** |
| detector gain, photon-counting | var/mean **1.001–1.012** | counting statistics; verify, do not assume from the beamline |
| background level vs rocking curve | \|r\| ≲ 0.3 healthy; **+0.92…+0.97 pathological** | at r ≳ 0.9 you are subtracting a θ-dependent scalar and widths are biased |
| points per FWHM for a per-pixel model test | **≳ 12** | below it, moment-based statistics measure broadness, not shape |
| θ-rocking rank ceiling, Q̂ in a plane | **6** of 9, any rotation axis | rank 9 needs the oblique geometry (Detlefs 2025) |
| flagged-vs-unflagged enrichment for a doublet/bimodality claim | **≫ 1** required | 0.96–1.0 means the statistic describes the estimator |

**Two numbers a report must carry whenever it quotes a map statistic:** the field's
autocorrelation length and the resulting **n_eff**. An iid σ over map pixels is not a σ
(rule 19).

## 5c. Artifact / report generation

For a shareable HEDM-style report artifact, follow the group's report recipe (the
`hedm_report_artifact` pattern) — same provenance discipline. The paper
`packages/midas_dfxm/dev/paper/P_merged/` is the reference for every formula and its
verification.

## 5d. Before you finish — update `RUNBOOK.md`

Record where the last session stopped, what ran on which host, and any new healthy number,
in [`RUNBOOK.md`](RUNBOOK.md) §R3. If it is stale, the next session re-derives what you
already knew.

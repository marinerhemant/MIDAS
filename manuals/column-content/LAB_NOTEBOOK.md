# LAB NOTEBOOK — evidence ledger (registrations live in the development workspace, not in this repo)

Verdict words are copied from the registrations; nothing here is upgraded.

| date | registration | read | one line |
|---|---|---|---|
| 2026-09-24 | `PREREGISTER_column_content.md` | NOT VALIDATED (V5 only) | narrow Laue fit: V1-V4 pass; the per-frame completeness indicator (C_int) does not track misses; completeness = recall-by-share table |
| 2026-09-24 | `PREREGISTER_spectrum_budget.md` | REFUTED (/verify) | spectrum-constrained brightness scatter 0.68 point estimate; honest 0.74-0.79 > ln 2 |
| 2026-09-24 | `PREREGISTER_spectrum_budget_v2.md` | REFUTED (post-hoc robustness) | corrected spectrum "passed" via a selection artifact (T6); 0.911 on a model-independent spot set; route closed |
| 2026-09-24 | `PREREGISTER_streak_origin.md` | SUBSTRATE STREAKS, marginal | streaks are real, localised crystals, not an artifact; the rolled substrate streaks (sampleG substrate 0.22 vs deposit 0.17); sampleH 0.35; depth unresolved. Its "ownership" diagnostic was later CONTRADICTED (T8) |
| 2026-09-24 | `PREREGISTER_column_content_wide.md` | NOT VALIDATED (V2, V7: discovery) | wide fit passes the fit gates; recall limited by the indexer for spread crystals; real sampleH descriptive: unexplained 0.474 -> 0.421 |
| 2026-09-25 | `PREREGISTER_arc_axes.md` | NOT FEASIBLE | per-frame arc axis: fresh controls 12.6 deg error after a disclosed 9-setting tuning |
| 2026-09-25 | `PREREGISTER_arc_beads_pooled_axis.md` | INCONCLUSIVE (controls failed) | beads 1.9x (bar 2x); axis detection strong, location a plateau |
| 2026-09-25 | `PREREGISTER_arc_v2.md` | controls PASS; real: BEADED, no single common axis; /verify PROVISIONAL | 91.8% of sampleH arcs beaded vs 3.9% continuous controls; spacing 0.134 deg plane-normal rotation |
| 2026-09-25 | `PREREGISTER_arc_crowding.md` | CHANCE CHAINING EXCLUDED | 0 arcs >= 20 px in 200 compact-only frames up to 491 peaks (real median 208) |
| 2026-09-27 | `PREREGISTER_mono_validation.md` | NOT VALIDATED (V2 recall 0.887 vs 0.90); /verify PROVISIONAL | monochromatic implementation, 240 synthetic DAC columns: V1, V3, V4, V6, V7, V8 pass; V2 fails by 1.3 points (column-bootstrap 95% CI [0.855, 0.916]); strict one-to-one recall 0.864; misses are weak domains and 1-deg near pairs. Amendment 1 (OOM restart, 30 workers, resumable driver) and Amendment 2 (columns 219 and 221 needed a faster solver: 219 scored on the original, 221 on the block-solver re-run) were written before any result was read |

## Retractions / contradictions
- "The unexplained half of sampleH is under-fitted spread of found crystals" (streak ownership 64% vs null 3.6%):
  CONTRADICTED by the wide fit (T8).
- The spectrum-budget "USABLE" read (0.680): REFUTED. The bar sat inside the crystal-bootstrap CI, and
  leave-one-out and bias correction put it above ln 2.
- The corrected-spectrum "MEETS BAR" (0.581): a selection artifact (T6).

# FF-HEDM Handbook — moved

This document was split into a **doc set** on 2026-08-11. It is now a directory, because a
1378-line single file could not be loaded selectively: an agent either took the whole thing
into context or skipped the hard rules.

**→ [`manuals/ff-hedm/README.md`](ff-hedm/README.md)** — start there. It is the spine, and
the only part meant to stay loaded: scope gate, install gate, the order of operations, the
hard rules and the halt conditions. Everything else is opened when you reach it.

| Was | Is now |
|---|---|
| Handbook §0–§0a, hard rules, traps | [`ff-hedm/README.md`](ff-hedm/README.md) |
| Handbook §0b, §0c, §1 | [`ff-hedm/phase-0-survey.md`](ff-hedm/phase-0-survey.md) |
| Handbook §2–§5g | [`ff-hedm/phase-1-geometry.md`](ff-hedm/phase-1-geometry.md) |
| Handbook §6, §6b, §6c, §10 | [`ff-hedm/phase-2-configure.md`](ff-hedm/phase-2-configure.md) |
| Handbook §7, §12 | [`ff-hedm/phase-3-run.md`](ff-hedm/phase-3-run.md) |
| Handbook §8, §11, §14 | [`ff-hedm/phase-4-read-report.md`](ff-hedm/phase-4-read-report.md) |
| Handbook §9 reference numbers | [`ff-hedm/RUNBOOK.md`](ff-hedm/RUNBOOK.md) |
| Handbook §13 | [`ff-hedm/C_REFERENCE.md`](ff-hedm/C_REFERENCE.md) |
| `FF_HEDM_Lab_Notebook.md` | [`ff-hedm/LAB_NOTEBOOK.md`](ff-hedm/LAB_NOTEBOOK.md) |
| `utils/ff_diagnosis_reference.md` | [`ff-hedm/DIAGNOSIS.md`](ff-hedm/DIAGNOSIS.md) |

Section numbers are unchanged, so an existing `§n` reference still resolves — the table in
the spine says which file holds which. **The text was moved, not rewritten**: the split
commit changed no claim, so anything verified against the code before it is still verified.

This stub stays. Checkpoints, memory entries and older sessions reference the old path.

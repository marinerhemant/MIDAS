# PDF runbook — what is true right now

> Part of the **pdf doc set**. Spine: [`README.md`](README.md).
> The spine is the procedure; this file is the state. If they disagree, re-check this one:
> procedures age more slowly than facts.

**Owner:** Hemant Sharma. **Last reviewed 2026-09-10.**

## Current state — the pick-up point

Every line here is recorded in [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) or [`ENVELOPE.md`](ENVELOPE.md).
Nothing below is a plan; it is what has actually been exercised.

| what | state |
|---|---|
| End-to-end reduction on real data | **EXERCISED once** — one beamtime, one detector (2026-09-10) |
| Physical consistency of that reduction (⟨S⟩ → 1, low-r slope) | **FAILS on every sample.** Cause not established; oblique-incidence detector efficiency is the lead (Notebook §4) |
| Agreement with an independent GSAS-II reduction | **CONFIRMED as registered, PROVISIONAL** — not `/verify`-d; consistency, not accuracy |
| Honest lattice-constant uncertainty | recipe run: a_Ni = 3.524673 ± 3.9e-3 Å (5.1e-4 without the chain term) |
| Per-pixel σ | **every model refuted**; all bands and significances labelled uncalibrated |
| strain-PDF from one frame | in-plane components only; the package's CRLB and `recover_strain` output are unusable as they stand |
| Multiphase / core-shell | decoy INCONCLUSIVE; the core-shell null is non-zero with σ = 0.0 |
| RMC | 4 chains agree; the first-shell offset is the disorder bias, not a measurement of a (Notebook §11) |
| Bayesian | NUTS agrees with the MAP once two package crashes are worked around; SVI does not reliably; model ranking not decisive (Notebook §12) |
| An earlier report on these frames | nine of its claims **REFUTED** (Notebook §1); not withdrawn with its recipients |
| Package defects found | 12, listed in Notebook §10; **none fixed, maintainers not asked** |
| Behavioural floor check for this doc set | **none exists** |
| Doc set handed to a context-free model | **NOT run** |

## Conditions on "a usable G(r)"

A G(r) is usable **only against a stated condition**, never on its own:

* **S(Q) → 1** outside the normalisation anchor, and the low-r slope, pass a stated threshold,
  **or** the report says they failed and every model number is marked conditional.
* **Per-pixel σ** calibrated by interleaved slivers with a planted control, **or** every band
  labelled uncalibrated.
* **The Q scale** verified against isolated calibrant rings to the Q_max of the transform, **or**
  the unverified range declared.
* **A model parameter** quoted with σ_total by the fixed recipe (phase-6 §6.2), with and without
  the chain term.

## Where the next session should start

1. **The sensor thickness** of the detector (data sheet), then a **new** pre-registration of the
   detector-efficiency correction as a real correction, not a sensitivity arm.
2. **`/verify` the one positive result** (the GSAS-II agreement), before anyone cites it.
3. **Raise the package defects** in Notebook §10 with the maintainers; do not work around them
   silently in new code.
4. **Hand this doc set to a context-free model** and record what it gets wrong in `ENVELOPE.md`.

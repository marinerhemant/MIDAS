# Phase 6 — reporting

## Before writing anything

Read **`ENVELOPE.md`**. It decides what may be claimed; this file only decides how to say it.
A dataset can be squarely in scope and unable to support the question — §0 is the table.

## The three-way split every number falls into

| | say it as | example from this campaign |
|---|---|---|
| **Established** | plain, with the null that could have failed | "9R on the shared ⟨111⟩: 55.8 % vs a 1.9 % null, ~30×; 1018/1025 Σ3 pairs at 0.0°" |
| **Provisional** | labelled provisional, **in the same sentence** | "lateral coherence ≥ 5–10 nm — a lower bound, mosaic-contaminated" |
| **Not obtainable** | plainly, with **no suggestion** for fixed/intrinsic limits | "whether the 9R is bulk or an interfacial film is not decidable by FF-HEDM" |

The tier decides whether a counterfactual is allowed at all. Suggesting a change to a
**configured** limit is useful; suggesting one for a **fixed** or **intrinsic** limit reads as
not knowing the instrument and costs more trust than the observation gained.
`ENVELOPE.md` tier table.

## What must accompany every number

* **Provenance.** The file and the command that produced it, re-derivable. Read numbers from
  source, never from conversational memory.
* **The denominator that was actually evaluated**, and what was discarded beside it. "2 of 45"
  meant "2 of 45 *evaluated*" after 43 had been silently dropped.
* **The null**, and the statement that it could have failed. A null that cannot produce a hit
  proves nothing.
* **For any fraction, the separation method.** A budget percentage without it is arithmetic,
  not attribution.

## Say what did not work

The retractions are the feature. This campaign's ledger holds five, and one of them is a
retraction that was itself wrong and had to be reversed (`LAB_NOTEBOOK.md` R3). A method that
cannot fail visibly will produce confident wrong answers at scale, and a report that hides its
negatives is claiming exactly that.

A clean negative is a result: HCP at the grain boundaries was **not detected**, on two
independent tests, and that stands (`LAB_NOTEBOOK.md` E6).

## Structured reporting

If you are producing a diagnostic report rather than prose, the technique-neutral contract is
`beamreport` (`~/opt/beamreport`): `Results`, `Quality`, `Provenance`, and the `Sidecar` of
**per-observation residuals with the coordinates they were measured at**. Most pipelines
compute that misfit during the fit and then discard it; keeping it is usually a dozen lines
and is what makes an automated diagnostic report possible at all.

The generic diagnostics key off residuals against declared coordinates. This technique's own
symptoms — a textured-sample null, a scalar classifier at high |q|, an ω-split read as mosaic
— are not expressible that way and are declared in `DIAGNOSIS.md` instead, deliberately.

## Update the runbook

`RUNBOOK.md` ends with a pick-up point. Update it before you finish. A stale pick-up point is
worse than none: the next session re-derives what was already known and trusts the rest less.

# Phase 4 — which phase, and what pressure

## Identifying a phase, or a member of a structural series

```python
from midas_hkls.phase_id import ...        # line counts + volume-correct null
```

**Three traps fire together here, and on one dataset all three did.**

**1. ASYMMETRIC CELL PROVENANCE.** Giving the favourite a cell refined from THIS dataset at
THIS pressure, while the rivals get published ambient cells, tests **cells, not phases**. Every
candidate gets equal treatment: either all ambient, or all refined on this data, or all scaled
by a free factor — and say which.

**2. LINE DENSITY DECIDES A d-MATCHING COMPARISON.** A candidate offering more lines wins on
chance alone. Measured: a rival was scored with the wrong reflection conditions —
La4Ni3O10 is **Bmab with 97 lines**, not F with 50 — and the comparison inverted when fixed.
Report the line count of every candidate and use a volume-correct null.

**3. A residual without a line count is not evidence.** State both.

**Then check the leverage before believing the verdict.** If the accessible reflections carry
almost no `c` information and `c` is the only parameter separating the series, the comparison
**cannot** decide — measured, c-term 0.00–5.61 % of 1/d² on six spots, with n = 2 and n = 3
differing in-plane by 0.31 %, absorbed by any free scale. See `ENVELOPE.md` §4.

**Do not forget the non-sample phases.** A DAC pattern contains the gasket (Re), the anvils
(diamond), a pressure marker (Pt, ruby...) and possibly a medium (Ne, cBN). All of them must
be in the candidate list before any unexplained spot is called sample material. Read which
ones are present **from the delivered record for the sample in hand** — pressure markers and
pressures do not transfer between projects, and one such number leaked between two unrelated
DAC datasets and inverted a compression argument.

## Getting a pressure

There is **no MIDAS package for equation-of-state inversion**; it is a dozen lines of scipy.
That is fine, but it means nothing checks you. The rules below are the check.

**Anchor on the marker actually in the beam.** A gasket line, a pressure-marker line, and the
sample's own cell are three different gauges of three different things:

| gauge | reads | caveat |
|---|---|---|
| gasket (e.g. Re hcp) | the **gasket** | under axial stress a **LOWER BOUND**; not the chamber |
| marker (Pt, ruby) | near the sample | the intended gauge; use it if present |
| the sample's own cell | the sample | needs a published P–V ladder that REACHES this pressure |

**Quote the model-free compression alongside the pressure.** `V/V₀` is what you measured;
the GPa is `V/V₀` plus somebody's equation of state plus an ambient reference, and those add
several GPa of spread on their own. Measured across four published Re EoS, two V₀ values and
two temperatures: **38.5–48.0 GPa** from one `V/V₀ = 0.903–0.905`.

**The low-temperature correction is usually a model, not a measurement.** For Pt at 30 K it is
worth **−2.11 GPa (~11 %), six times the entire spread across every published EoS** — and for
Re below 77 K no measured lattice parameters could be found at all. Say which it is. And note
that substituting a cold V₀ into a 300 K EoS is **not** a cold isotherm: K_T stiffens too
(Anderson–Grüneisen moved one answer +3 %). Use a published thermal EoS if the number is
load-bearing.

**Sanity-check against a known case.** Run the whole chain on a sample whose pressure is
independently stated and see whether you recover it. Measured: an 18 GPa sample came back at
16.8–18.4 GPa from its Pt marker — that is what licenses the number on the unknown sample.

## Inverting through a fitted ladder — the failure that looks most like a result

**A spread obtained by inverting a measurement through a fitted curve is a property of the
FIT until you show otherwise.** Before quoting anything:

1. **Refit with ≥3 defensible functional forms** and quote the span. Measured: one axis gave
   40.6 (quadratic) / 38.3 (linear) / 30.2 (cubic) / undefined (exact quartic) / 5.1 (quadratic
   on the in-range points only) / 71.9 GPa. Interpolant-choice range **35.5 GPa**, against a
   claimed effect of 37.0 — i.e. the entire "result".
2. **Leave out each anchor in turn**, especially any whose provenance differs. Dropping one
   300 K anchor spliced into a 40 K in-situ ladder moved that gauge 40.6 → 5.1 GPa.
3. **Check the branch.** A root past the fitted curve's vertex is where the fit says the axis
   GROWS with pressure — not a measurement. One "dissolving" 5.1 GPa figure was exactly that.
4. **Check every fitted parameter against its bounds.** A railed fit is not a fit.
5. **Never splice an anchor of different temperature, phase or instrument into a ladder.**
   That is the asymmetric-provenance trap above, one level up.

**If the sample sits outside the ladder's range, say so.** Measured: a sample at ~45 GPa
compared against a ladder topping out at 24.6 GPa is ~20 GPa beyond calibration, and "the cell
is off the published path" is then mostly "**the published path does not reach here**". That is
a statement about the reference, not a finding about the sample.

## Report

Every number carries: the file and command that produced it, its q convention and cell setting,
its units, the null it beat, and the `ENVELOPE.md` limit that bounds it. If the envelope says
the ask is unobtainable, **that is the deliverable** — write it as the answer, not as a
failure.

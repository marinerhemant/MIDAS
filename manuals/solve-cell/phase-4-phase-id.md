# Phase 4 — which phase, and what pressure

## Identifying a phase, or a member of a structural series

```python
from midas_hkls.phase_id import PhaseCandidate, identify_phase
ranked = identify_phase(d_obs, [PhaseCandidate(name, crystal, cell_source="ambient"), ...],
                        free_scale=True, n_null_draws=200)
# each PhaseMatch: n_lines, worst_free_pct, best_scale, p_value (against a random phase
# with the SAME number of lines), cell, cell_source
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

**Use the c-axis repeat when you have one — it outranks d-matching — and give d-matching ONE domain's
indexed reflections.** On the delivered 2604 position, `identify_phase` fed the un-indexed kept spots ranked RP
n = 3 (0.592 %, p = 0.025, 171 lines) ABOVE n = 2 (0.796 %, p = 0.05, 127 lines). Fed the 12 distinct d of one
domain's INDEXED reflections, it ranked n = 2 first (0.235 %, p = 0.0, 115 lines) against n = 3 (0.937 %,
p = 0.175, 155 lines). The cell-free row finder had decided it either way: a 13-rung row repeating every
**9.6188 Å** along c* matches n = 2's c/2 at its fitted scale to +0.0 %, against +51.4 % for n = 3 and −25.0 %
for n = 1. Where a ladder is accessible read the member off it; where it is not (S5, c-term median 0.020 of
1/d²) the member is undeterminable, and both n = 2 and n = 3 then "match" at p = 0.0.

**`identify_phase`'s free scale is ISOTROPIC.** A layered phase under pressure compresses far more along c,
so the two-scale (ab, c) freedom the envelope calls for cannot be expressed through this API. A ranking it
produces for a layered phase at pressure is provisional by construction.

**Do not forget the non-sample phases.** A DAC pattern contains the gasket (Re), the anvils
(diamond), a pressure marker (Pt, ruby...) and possibly a medium (Ne, cBN). All of them must
be in the candidate list before any unexplained spot is called sample material. Read which
ones are present **from the delivered record for the sample in hand** — pressure markers and
pressures do not transfer between projects, and one such number leaked between two unrelated
DAC datasets and inverted a compression argument.

**Assign harmonic lines before inferring a new phase or a longer period.** A beam with harmonic contamination
diffracts the strong phases a second time at λ/3: the diamond anvils' (111) (d = 2.059 Å, 2θ ≈ 11.8° at
λ = 0.4246 Å) reappears near 2θ ≈ 3.97°. On both delivered La3Ni2O7 positions that ring, 3.97–4.14°, was the
ONLY ring in the collaborators' own table with real azimuthal-median contrast (+13.6 % on 2604, +9.0 / +7.3 %
on S5, against +0.11–0.19 % median for the rest of their table). Unassigned, it reads as a d ≈ 6.2 Å line —
exactly the kind of long spacing that gets called a new phase or a superlattice. Put `λ/3` (and, on a
monochromator that passes them, `λ/2`) copies of every strong phase in the candidate list.

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

**Take gauge rings from the UNSUBTRACTED image.** The defect ingest contract detects rings on the
background-subtracted maximum, and the polar background removes powder rings by design — that is its job.
On the delivered 2604 position those rings matched only 2 Re gasket lines, so there was no gauge. Run
`detect_powder_rings` on the median over live frames, with maps from `detector_angle_maps`.
Done that way, the test run re-derived **Re a = 2.669 Å from 5 of its 6 lines (8 matched rings), V/V₀ = 0.9033,
46.0 GPa** on 2604 —
the recorded gasket compression (0.903–0.905; that claim is PROVISIONAL) — and Pt a = 3.8475 Å from 2 lines
(18.7 GPa) on S5.

**Report the matched lines and the degrees of freedom with every pressure.** A cubic marker has one lattice
parameter, so two matched lines give one check; an hcp gasket with c/a free has two and needs three. On S5
the Pt marker gave a = 3.846 Å from two lines (19.1 GPa at 300 K), agreeing with an independent fit
(3.84499 Å) — quotable, but on one degree of freedom.

**Count DISTINCT lines, and check the strong ones are there.** With 33–60 rings on a frame, some ring lies
within ±0.04° of 11–19 % of the 3–21° band, so a scan over the lattice parameter always finds a few
coincidences. The test run's harness counted matched rings and reported "Pt, 3 lines, 41.7 GPa" on 2604 —
2 distinct lines, the (220) sitting on the Re gasket's (110), the strong (200) absent — and "Re, 4 lines,
108.9 GPa" on S5 — 3 distinct lines, Re's strongest (101) absent. Neither phase had been shown to be present.
Run the same scan for a phase known to be absent before believing a match (`ENVELOPE.md` §15).

**Quote the model-free compression alongside the pressure.** `V/V₀` is what you measured;
the GPa is `V/V₀` plus somebody's equation of state plus an ambient reference, and those add
several GPa of spread on their own. Measured across four published Re EoS, two V₀ values and
two temperatures: **38.5–48.0 GPa** from one `V/V₀ = 0.903–0.905`.
With one sourced set — Anzellini, Dewaele, Occelli, Loubeyre and Mezouar, J. Appl. Phys. 115, 043511
(2014): K₀ = 352.6 GPa, K₀' = 4.56, 300 K, helium medium, to 144 GPa — the recorded V/V₀ = 0.903–0.906 gives
**43.7–45.3 GPa** (Vinet and BM3 differ by 0.2 GPa); the uncited (360, 4.5) that produced the first "45" sits
0.8 GPa higher. That is what a citation buys: a number someone else can re-derive.

**Write the EoS parameters and their source next to every pressure.** No MIDAS package carries them and
this doc set does not either. On a held-out S5 position (dry run 2026-09-10) the reader quoted six Pt
parameter sets from memory and flagged them unverified — correctly, but a remembered parameter is a
systematic nobody can bound, and the project's own gasket scripts carry Re K₀ = 360 GPa, K₀' = 4.5 with no
citation. Look the parameters up, cite the paper and the table, and state V₀ with its temperature.

**The low-temperature correction is usually a model, not a measurement.** For Pt at 30 K it is
worth **−2.11 GPa (~11 %), six times the entire spread across every published EoS** — and for
Re below 77 K no measured lattice parameters could be found at all. Say which it is. And note
that substituting a cold V₀ into a 300 K EoS is **not** a cold isotherm: K_T stiffens too
(Anderson–Grüneisen moved one answer +3 %). Use a published thermal EoS if the number is
load-bearing; when there is none for a cold gauge, report the 300 K-V₀ and the cold-V₀ numbers together as
the envelope, each labelled a model, and do not pick one.

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

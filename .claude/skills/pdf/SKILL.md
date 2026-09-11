---
name: pdf
description: >-
  Take high-energy total-scattering area-detector frames (sample, empty
  container, air, calibrant) to S(Q), G(r) and a structural model with
  midas-pdf. Survey the folder; carry a verified geometry across from
  calibrate-integrate with RhoD and the residual map; integrate with the
  polarization and solid-angle corrections the variance integrators do not
  apply; reduce I(Q) → S(Q) → G(r) with container subtraction, absorption,
  multiple scattering and Compton; check the result against physical
  constraints before fitting. Then model it: small-box refinement with an honest
  uncertainty, a raw-data phase check before any multiphase or core-shell fit,
  strain-PDF read in the Fisher eigenbasis, RMC, and Bayesian posteriors. Use
  when asked to make a PDF, G(r), S(Q) or F(Q) from powder or liquid frames,
  refine a lattice constant or U_iso from G(r), test for a second phase, run RMC,
  or when a G(r) looks wrong (a spike below 0.3 Å, S(Q) that never settles to 1,
  a lattice constant with an impossibly small error). Measured on one beamtime
  (1-ID-E Varex, 67 keV); stops and asks outside it. Calibrating the detector is
  calibrate-integrate; spotty rings are ff-hedm or pf-hedm; continuous rings per
  tomographic voxel are xrd-ct.
---

# PDF / total scattering

**This skill is a pointer, not the procedure.** The procedure is a doc set in the repository, so
it lives beside the code it cites and stays usable without this skill.

## Start here

Read **`manuals/pdf/README.md`**, the spine. It is the only file meant to stay loaded: the scope
gate, the survey, the geometry hand-off, integration with corrections, the verification checks,
the halt conditions and the traps table. The files hang off it, each loaded on demand:

| file | when |
|---|---|
| `HARD_RULES.md` | applies to every phase; 18 rules, each written after a silent wrong answer |
| `phase-5-sq-gr.md` | the reduction recipe: I(Q) → S(Q) → G(r), and the arms to run beside it |
| `phase-6-model.md` | small box and its uncertainty, phase checks, multiphase, strain-PDF, RMC, Bayesian |
| `DIAGNOSIS.md` | a result looks wrong: symptom → test → cause → lever |
| `ENVELOPE.md` | **before trusting a path**: what has actually been run, on what data |
| `LAB_NOTEBOOK.md` | the evidence, including an earlier report on the same frames that did not survive |
| `RUNBOOK.md` | the pick-up point: what is true right now |

Then give, or work out from the folder:

```
Data folder:  <ABSOLUTE PATH>     # sample, empty container, air, calibrant frames
Samples:      <composition, density or packing, container geometry — stated, not tuned>
Energy or λ:  <keV or Å — or "find it": the tabulated K edge of the mono foil>
```

## Five things to know before you start

1. **The integrators do not correct intensity.** The variance integrators ignore the
   polarization and solid-angle flags. An uncorrected liquid G(r) carries a clean-looking spike
   of +400 at r ≈ 0.1 Å. Pass the correction yourself. Hard rule 1.

2. **The geometry hand-off drops two things.** `to_integration_spec()` carries neither `RhoD`
   nor the residual map. Without `RhoD`, pixel radii come out near 1e33 px; without the map, the
   outer rings were 5× worse. Hard rule 2.

3. **No σ from this chain is calibrated until a sliver test says so.** On the reference data
   √S was 15–19× too small, and even a measured noise model 3.5×. A refined σ also needs √χ²_ν
   and √(N/N_eff): a Hessian σ(a) of 1.7e-6 Å became 3.9e-3 Å by a fixed recipe. Hard rules 6, 7.

4. **Check the physics before the model, and do not read agreement as validation.** The
   reference S(Q) failed ⟨S⟩ → 1 and the low-r slope on every sample, while a GSAS-II reduction
   of the same frame agreed at Pearson 0.997. Spine §6; hard rule 16.

5. **The modelling tools give confident numbers where they have no information:**
   - percent-level strain along blind directions;
   - σ = 0.0 from a degenerate Hessian;
   - phase weights with no σ;
   - RMC atoms named `X`.

   Hard rules 9, 11, 12, 17.

## When something looks wrong

Go to **`manuals/pdf/DIAGNOSIS.md`**, indexed by symptom.

Before re-investigating anything, read **`manuals/pdf/LAB_NOTEBOOK.md`**:
- §1 lists nine claims from an earlier analysis of the same frames that the re-run refuted,
  including a ±1e-5 Å lattice constant and a "1499 of 1500 points significant" Δ-PDF.
- §3 records why the outer calibrant rings could not be fixed by binning.

## Sibling doc sets

`manuals/calibrate-integrate/` (skill `calibrate-integrate`) produces the geometry this one
consumes. Others:
- `manuals/xrd-ct/` (`xrd-ct`): powder rings per tomographic voxel;
- `manuals/ff-hedm/` (`ff-hedm`) and `manuals/pf-hedm/` (`pf-hedm`): spotty rings;
- `manuals/defect/` (`defect`): the diffuse field between Bragg spots of a rotation series.

## Log a halt

Technique skills carry no verdicts of their own. The one thing worth logging is when the doc set
**stops** you:

```bash
~/.claude/bin/skill-log --skill pdf --event invoked --verdict INVOKED \
  --subject "<which halt condition fired, or 'ran to completion'>" \
  --evidence <the file or reading that triggered it> \
  --note "<what would unblock it>"
```

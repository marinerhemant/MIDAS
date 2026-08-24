---
name: calibrate-integrate
description: >-
  Take a powder / calibrant detector dataset from raw frames to a calibrated
  geometry and integrated patterns: survey the folder, calibrate from scratch
  with midas-calibrate-v2, integrate one file, a folder, or a live stream with
  midas-integrate-v2 on CPU or GPU, and verify against the raw rings before
  quoting anything. Use when asked to calibrate a detector, integrate or
  azimuthally average diffraction frames, reduce a powder / CeO2 / LaB6 scan,
  set up a live integration server, or when integrated patterns look wrong
  (spiky near cardinal angles, rings offset, panels railed), or when reprocessing
  an archive of past beamtimes unattended. Covers tiled Pilatus, GE single and
  Hydra quad, Varex and Dexela frames through the midas_calibrate_v2 +
  midas_integrate_v2 chain; EIGER2 geometry is recoverable but NOT yet
  verified to spec (0 of 6 verified), so treat it as unproven and check
  ENVELOPE.md first; stops and asks outside that.
---

# Calibrate → integrate

**This skill is a pointer, not the procedure.** The procedure is a doc set in the
repository so it lives beside the code it cites and stays usable without this skill.

## Start here

Read **`manuals/calibrate-integrate/README.md`** — the spine. It is the only file
meant to stay loaded: the scope gate, the install gate, the survey, the dispatch
matrix, integration, verification, the halt conditions and the traps table.
Unlike the FF/NF spines it **carries the commands inline**, so a reduction can
run from it alone. Four files hang off it, and each is loaded on demand:

| file | when |
|---|---|
| `HARD_RULES.md` | applies to every phase; 13 rules, each written after something silently produced a wrong answer |
| `phase-4-calibrate.md` | the calibration recipe itself (the spine keeps the gates, this keeps the code) |
| `ENVELOPE.md` | **before trusting a detector class** — what has actually been calibrated, per class, with numbers |
| `RUNBOOK.md` | the pick-up point: what is true right now, and what is still not exercised |

Then give, or work out from the folder:

```
Data folder:  <ABSOLUTE PATH>     # frames, or a single file
Calibrant:    <CeO2 / LaB6 / Si / none — it is a sample, I have a geometry>
Energy or λ:  <keV or Å — or "find it", see §1>
```

Everything else is worked out from the files.

## Four things to know before you start

1. **A silent wrong answer is the normal failure here.** Discarded panel shifts,
   a truncated map, the wrong calibration block, a dark frame used as signal, a
   detector that cannot physically see the calibrant — every one of these
   produces a complete, plausible-looking result that passes the downstream
   diagnostics, because those diagnostics are grading a fit that converged. The
   spine lists **named halt conditions**; halt on those whether or not anything
   looks wrong.

2. **Do not seed the calibration from an existing parameter block.** `make_seed`
   finds the beam centre and distance from the image itself. Handing it a prior
   answer inherits that answer's errors — spine §2, and Lab Notebook §1 for the
   case where the active block was 3 mm wrong and every downstream number moved.
   If `result.seed_method` comes back `"fallback"`, the validated seeder gave up
   and the answer is not trustworthy.

3. **`SubPixelLevel` stays at 1.** Above 1 the CUDA integrator reads the wrong
   pixel; measured 24.3× on in-band bins. `0` is bit-identical to `1`. Hard
   rule 1.

4. **λ comes from the beamline and never from the fit.** Wavelength and `Lsd`
   are degenerate, and with the default `refine_distortion=True` the degeneracy
   is not weakly broken but **not broken at all** — a 5.9 % energy error leaves
   8.8e−06 µε of residual. The strain gate passes, and the distance is wrong by
   the same fraction. Hard rule 9; do not try to scan candidate energies for the
   best residual (Lab Notebook §15 records that attempt and its refutation).

## When something looks wrong

Go to **`manuals/calibrate-integrate/DIAGNOSIS.md`** — symptom → discriminating
test → cause → lever, indexed by *symptom*.

Before re-investigating anything, read **`manuals/calibrate-integrate/LAB_NOTEBOOK.md`**.
Several attractive hypotheses are recorded there as **refuted**, with the
measurement that killed each — §4 where the published cardinal-aliasing result
did *not* reproduce on real 20-ID data, §5 where the obvious per-panel knob is
the wrong functional form, and §15 where recovering the beam energy from the fit
residual looked like an 83 %-accurate result and was matched exactly by a
data-blind constant guess.

## Sibling doc sets

`manuals/ff-hedm/` (skill `ff-hedm`), `manuals/nf-hedm/` (`nf-hedm`),
`manuals/pf-hedm/` (`pf-hedm`), `manuals/xrd-ct/` (`xrd-ct`). Those consume the
geometry this one produces.

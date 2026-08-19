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
  (spiky near cardinal angles, rings offset, panels railed). Covers tiled
  Pilatus and single-panel area detectors through the midas_calibrate_v2 +
  midas_integrate_v2 chain; stops and asks outside that.
---

# Calibrate → integrate

**This skill is a pointer, not the procedure.** The procedure is a doc set in the
repository so it lives beside the code it cites and stays usable without this skill.

## Start here

Read **`manuals/calibrate-integrate/README.md`** — the spine. It is the only file
meant to stay loaded: the scope gate, the install gate, the dispatch matrix, the
commands, the hard rules and the halt conditions. Unlike the FF/NF spines it
**carries the commands inline**, so a reduction can run from it alone.

Then give, or work out from the folder:

```
Data folder:  <ABSOLUTE PATH>     # frames, or a single file
Calibrant:    <CeO2 / LaB6 / Si / none — it is a sample, I have a geometry>
Energy or λ:  <keV or Å — or "find it", see §1>
```

Everything else is worked out from the files.

## Three things to know before you start

1. **Do not seed the calibration from an existing parameter block.** `make_seed`
   finds the beam centre and distance from the image itself. Handing it a prior
   answer inherits that answer's errors — spine §2, and Lab Notebook §1 for the
   case where the active block was 3 mm wrong and every downstream number moved.

2. **`SubPixelLevel` stays at 1.** Above 1 the CUDA integrator reads the wrong
   pixel; measured 24.3× on in-band bins. `0` is bit-identical to `1`. Spine
   hard rule 1.

3. **A silent wrong answer is the normal failure here.** Discarded panel shifts,
   a truncated map, and the wrong calibration block all produce a complete,
   plausible-looking pattern. The spine lists **named halt conditions** — halt on
   those whether or not anything looks wrong.

## When something looks wrong

Go to **`manuals/calibrate-integrate/DIAGNOSIS.md`** — symptom → discriminating
test → cause → lever, indexed by *symptom*.

Before re-investigating anything, read **`manuals/calibrate-integrate/LAB_NOTEBOOK.md`**.
Several attractive hypotheses are recorded there as **refuted**, with the
measurement that killed each — including one (§4) where the published
cardinal-aliasing result did *not* reproduce on real 20-ID data, and one (§5)
where the obvious per-panel knob is the wrong functional form.

## Sibling doc sets

`manuals/ff-hedm/` (skill `ff-hedm`), `manuals/nf-hedm/` (`nf-hedm`),
`manuals/pf-hedm/` (`pf-hedm`), `manuals/xrd-ct/` (`xrd-ct`). Those consume the
geometry this one produces.

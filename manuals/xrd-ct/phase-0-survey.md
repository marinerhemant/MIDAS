# Phase 0 — survey: is this XRD-CT, and what is in the files?

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Goal:** decide whether these recipes apply at all, and decode the format before touching any
analysis. Nothing here is optional and none of it takes long.

---

## 0.1 The scope gate — continuous rings or spots?

XRD-CT integrates azimuthally and assumes a **powder-like** ring. If the rings break into
discrete spots the sample is coarse-grained, azimuthal intensity is a crystallite-count
fluctuation, and every texture number this pipeline produces is noise.

**Test.** Plot one raw frame from the middle of the scan, and one azimuthal cut through a
strong ring at the working bin size.

* Continuous, with amplitude modulation → XRD-CT. Continue.
* Discrete maxima with gaps at background → **stop.** This is scanning-3DXRD; use the
  `pf-hedm` doc set.
* Ambiguous → it is genuinely marginal. `azimuthal.mad_filter` (Mürer's 3-MAD rejection)
  suppresses the worst spottiness, but it cannot rescue a coarse-grained sample. Say the
  measurement is marginal and state it in the report.

**The dividing line is operational**, at the working (R, η) bin size — not a property of the
material.

## 0.2 Decode the format before anything else

Integrated DT output is a raw binary slab. Getting the axis order wrong gives an array that
reshapes fine and is transposed.

Establish, and write down:

```
n_omega, n_eta, n_R        frames, azimuthal bins, radial bins
dtype                      float64 is usual for integrated output
layout                     [nR][nEta] or [nEta][nR] -- CHECK, do not assume
n_files                    one per translation, usually
HeadSize                   header bytes; often already skips the throwaway frame
startOme, omeStep          and the SIGN convention
```

**Test the layout rather than assuming it.** Read one frame both ways and plot the mean along
each axis. The radial direction shows sharp rings on a falling background; the azimuthal
direction is smooth and roughly periodic. They are not confusable once plotted.

**Worked example (DAC Ti, 1-ID)** — decoded, in `LAB_NOTEBOOK.md` §7, do not re-derive:
`652 ω × 1207 η × 2400 R`, float64, layout **`[nR][nEta]`**, 25 files = 25 translations,
`startOme = −169`, `omeStep = 0.25`, **ω negated**, first frame already skipped via
`HeadSize 8396800`.

## 0.3 Two beamline conventions that cannot be recovered later

| Convention | Why it matters | How to establish it |
|---|---|---|
| **ω sign** | Flips the reconstruction. At 1-ID the aero stage needs **every ω negated** | Against the encoder / SPEC log, not by eye. It is a **halt condition** if unconfirmed |
| **Throwaway first frame** | 1-ID discards one frame every acquisition | Check whether `HeadSize` already skips it. Double-skipping loses a real frame; not skipping shifts every ω by one step |

Neither is checkable after the fact from a finished reconstruction — both produce output that
looks entirely healthy.

## 0.4 Inventory the scan

```
translations       count, step size, total span
omega              count, step, range, any blocked spans
rings visible      count, and roughly where
phases expected    and their space groups -- needed for symmetry in phase 4
```

Cross-check the **translation span against the sample size**. If the span is much larger than
the sample, most rays are vacuum and will contribute rows that constrain nothing.

## 0.5 Look at a radial profile

**Do not skip this. Four analyses on the DAC Ti scan were invalidated by skipping it.**

Plot the azimuthally-averaged radial profile of one frame, on a linear scale, with the
background visible. In one plot you learn:

* **Peak-to-background** — the number that decides what is answerable (`ENVELOPE.md` §0). On
  the Ti scan it was 1.17×, i.e. peaks 17 % above background at best.
* **Which "rings" are multiplets** — α(101) showed maxima at 381.6 and 393.6 px with a dip
  between, and had been matched as one line.
* **The FWHM in pixels** — sets the window width in phase 2. It was ~8 px, which meant the
  windows in use were wrong in both directions.

Five minutes. Four wasted analyses.

## 0.6 What to hand forward

```
Technique confirmed:   XRD-CT / not XRD-CT (and if not, which doc set)
Format:                n_omega, n_eta, n_R, dtype, layout, HeadSize
Omega:                 startOme, omeStep, SIGN (confirmed against what?)
Scan:                  n_translations, step, span vs sample size
Rings:                 candidate radii, which are multiplets
Contrast:              peak/background per candidate ring   <- decides the deliverable
FWHM:                  in PIXELS
Phases + space groups: for phase 4
```

Then go to `phase-1-geometry.md`.

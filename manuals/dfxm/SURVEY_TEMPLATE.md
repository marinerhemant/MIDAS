# SURVEY — <dataset name>

> Fill this in **before promising anything** (phase-0-survey.md §0b). Every value read from
> the file/metadata, never from a folder or file name (README rule 10). Copy this file into
> your work directory, rename it `SURVEY.md`, and delete the guidance in angle brackets.

**Date surveyed:** <YYYY-MM-DD>  ·  **Surveyed by:** <you>
**Scan folder (absolute):** `<path>`
**Beamline:** <ESRF ID06-HXM | archived ID03 | other — if 6-ID-C, STOP: gated (README scope)>

---

## What is here (per acquisition)

| field | value | source file / key | notes |
|---|---|---|---|
| scan kind | <mosaicity (rock/roll) / strain (θ,energy) / multi-reflection set> | | decides the path |
| reflection(s) | <e.g. 002, 111> | <the aligned-Bragg metadata, NOT the filename> | |
| energy / wavelength | <keV / Å> | | cross-checked ≥2 ways |
| effective pixel size (sample) | <µm/px> | <objective magnification record> | differs per reflection |
| frame count + step | <N frames, Δ per step> | | too few points flattens the moment |
| background / dark present? | <yes → path / NO> | | **must subtract before any moment (§2); if NO, stop** |
| co-registration metadata? | <yes / NO> | | **absence = §3 halt for a tensor** |
| ground truth? | <almost always NO> | | if NO, accuracy = injection-recovery, not round-trip (§2d) |
| scan still being written? | <no — counted twice, 120 s apart> | | never reduce a growing scan |
| **flux / monitor column?** | <yes → column name / NO> | <the motor metadata, column by column> | **absence = halt** for any intensity comparison across separately-acquired groups (rule 16). The frame total is **not** a monitor on a rocking scan |
| **magnification provenance** | <value, and the ≥2 routes that agree> | <the optical record — NOT a script constant> | sets every µm/px; a factor ~2 error is the common one (rule 15). A focus check does **not** verify scale |
| **detector type + measured gain** | <photon-counting / integrating; `var = a·y + b`> | <detector record, then photon transfer on pedestal-subtracted frames> | absolute χ²/dof and σ scale as 1/gain (rule 13). Measure **per detector**; never reuse another's |
| **sampling, per channel** | <step and µm/px for *each* reflection> | | a weak channel is often far coarser, so the best map can be blind to the physics of interest (Notebook §7g) |
| **does each scan's window bracket its own peak?** | <yes / N of M truncated> | <per-scan argmax vs window edges> | one fixed window reused across a raster while θ_B drifts biases widths and integrals unequally between channels (Notebook §5l) |
| **prior reduction — whose, and read?** | <none / path to their scripts, read: yes> | | read them before comparing channels or reporting a discrepancy; the correction is often already there (rule 11) |

---

## Configuration (from phase-1-configure.md — fill after §1)

**One row per reflection.** Λ ∝ 1/|F|, so a weak satellite and its strong parent can sit in
different regimes in the same crystal — never classify once per sample (§1b, Notebook §7g).

| quantity | value | call |
|---|---|---|
| θ_B / 2θ | | `bragg_angle_deg` / `bragg_two_theta_deg` |
| extinction length Λ (µm) | <per reflection> | `extinction_length` |
| t / Λ regime | <thin ≲0.15 / marginal / thick ≳0.3, per reflection> | governs what §4 may claim |
| coherent block size used for t | <NOT the mosaic width> | mosaic spread ≠ coherent block size (Notebook §5k) |
| Im χ_h (absorption) | | `susceptibility_fourier` |
| refraction gauge ε_ref | <χ₀ᵣ/(2sin²θ_B)> | a reference offset, NOT a per-pixel strain (§4a) |
| resolution widths (anisotropic) | | `poulsen_resolution_widths` |

## Reduction health (from phase-2-reduce.md — fill after §2)

| quantity | value | healthy | why |
|---|---|---|---|
| pedestal share of centroid weight, `f_ped` | | subtract regardless | ≳95 % dilutes the moment ~1/(1−f_ped) (§2a). **Definition below — the estimator changes the answer by 4×** |
| measured gain, this detector | <`var = a·y + b`> | quote it, don't assume 1 | absolute χ²/dof scales as 1/gain (§2a′) |
| r(background level, rocking curve) | | \|r\| ≲ 0.3 | ≳0.9 means a θ-dependent scalar → widths biased, centroid not (§2a) |
| kernel vs downsampled ROI | <kernel px / ROI px> | kernel < ROI | a kernel exceeding the ROI degenerates to a scalar (§2a) |
| **per-pixel** rocking FWHM | <argmax-local, contiguous crossings> | — | never infer sampling from an integrated/published width (§2f) |
| points per FWHM, per-pixel | | ≳12 for a model-selection test | below it, moment statistics measure broadness (§2f) |
| injection-recovery gain | | 0.9998–1.0000 observed | physical accuracy — not a round-trip (§2d) |

### How to compute `f_ped` — the estimator is the whole answer

`f_ped` is the fraction of total integrated intensity carried by a flat floor:

```python
import numpy as np
f_ped = float(np.median(d)) * d.size / float(d.sum())   # d = raw frames, un-subtracted
dilution_predicted = 1.0 / (1.0 - f_ped)
```

**Use the median for the floor level.** Measured on `darling.assets.mosaicity_scan()`,
2026-08-12:

| floor estimator | `f_ped` | predicted dilution |
|---|---|---|
| **median** | **0.9849** | **66.3×** |
| percentile 5 | 0.9459 | 18.5× |
| percentile 1 | 0.9264 | 13.6× |
| min | 0.7704 | 4.4× |

The median reproduces the documented 98.5 % and ~67× (Notebook §1a) on a *different* scan
from the one those came from. The other estimators are wrong by up to 15×, so an
unqualified "pedestal share" number is not comparable between sessions.

Two traps, both hit in practice:

- **This is not the estimator you subtract with.** For *subtraction* a conservative low
  percentile is right, because over-subtracting eats signal — `reduce_energy_chiltepin.py`
  uses percentile 5 deliberately. For *measuring* `f_ped` the median is right, because the
  question is where the flat floor sits, not how much you can safely remove. Using the
  subtraction estimator here understates `f_ped` badly.
- **A conservative high-tail estimate can give `f_ped > 1`**, which is not a fraction and
  means the estimator, not the data, is wrong. If you get that, you used a floor level above
  the mean intensity.

`f_ped` *predicts* the dilution; it is not a measurement of it. The predicted 66.3× against a
directly measured 71.8× on the same scan is the expected level of agreement — quote both if
you have both, and never quote `f_ped` as if it were the observed dilution.

---

## Halt-condition check (README STOP table)

- [ ] not 6-ID-C (or confirmed OK to proceed on a gated instrument)
- [ ] background available to subtract before any moment
- [ ] if a tensor is asked for: co-registration metadata exists (else report the wall)
- [ ] any uniform ~100s-µε strain offset understood as refraction gauge, not a field
- [ ] any strain claim is inside t ≲ 0.3 Λ, or uses the dynamical forward
- [ ] if intensities are compared across separately-acquired groups: a flux monitor exists
- [ ] µm/px traced to an optical record, not to a script constant
- [ ] before reporting a discrepancy with someone else's reduction: their scripts read
- [ ] if resolution is worse than the instrument's best and a correction is wanted: a vibration
      **spectrum** exists (an amplitude cannot answer it)
- [ ] every control run can state what result would have refuted it

## Open questions / what would unblock the next step

<one line each>

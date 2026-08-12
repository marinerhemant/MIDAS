# Phase 2 — Reduce raw frames to per-pixel maps

> Part of the **DFXM doc set**; spine is [`README.md`](README.md). This is the core
> **real-data-proven** step. The two things that silently corrupt it are the pedestal
> (rule 1) and quoting a round-trip as accuracy (rule 5).

---

## 2a. Subtract the pedestal — before any moment

**This is not optional and cannot be undone later.** On raw ESRF ID03 frames the pedestal
carried 98.5 % of the centroid weight, so the naive first moment was ~67× too small
(Notebook §1a). Subtract the scalar background (the deposit's own dark, or `darling`'s
background) first:

```python
frames_bs = frames - background      # background from the dark / darling's own scalar
```

**Where `background` actually comes from — three routes, name the one you used.** The doc set
said "`darling`'s own scalar" without ever naming the call, so a reader had to open
`darling/_dataset.py` to find it.

| route | call | when |
|---|---|---|
| the deposit's own dark | read it from the file | a separately-acquired dark exists |
| `darling`'s estimator | `DataSet.estimate_background()` — the only background method on the class | no dark, and you want `darling`'s own conservative scalar |
| flat percentile of the frames themselves | `np.percentile(frames, 5)` | no dark; this is what `reduce_energy_chiltepin.py` uses, deliberately conservative |

Neither bundled `darling` asset carries a dark, so on those the second or third route is the
only option — which is **not** what §0b's "background present? if absent, stop" implies if you
read it as requiring a separately-acquired dark frame. It requires a *defensible pedestal*,
not a dark file.

Whichever you use, **state it beside the number**: the estimator changes `f_ped` by up to 15×
(`SURVEY_TEMPLATE.md`), so an unqualified pedestal figure is not comparable between sessions.

If there is **no** defensible pedestal at all (§0b), stop — the moment is not trustworthy.

**And do not over-subtract: never let the subtraction follow the signal.** Correlate the
per-frame *level* of whatever you remove against the integrated rocking curve. At r ≳ 0.9 you
are subtracting a θ-dependent scalar, which distorts curve **shape** — every per-pixel
width/FWHM computed afterwards is biased, while the centroid is barely affected, so the map
still looks right. The usual cause is a morphological-opening / rolling-ball background whose
structuring element **exceeds the downsampled ROI**, on an ROI with no non-diffracting pixels
to serve as a common-mode reference: it collapses to exactly that scalar (r = +0.919 measured,
and still +0.966 where the kernel did fit). Check kernel size against the ROI, and prefer
dark-only (rule 14, Notebook §7c).

## 2a′. Measure the detector gain before quoting any absolute χ²/dof

Do this once, on the pedestal-subtracted frames, by photon transfer — the variance of
nearest-neighbour differences against local mean:

- slope ≈ 1, intercept ≈ 0 → photon-counting statistics, gain 1
- slope ≳ 2 → the counts are ADU; every absolute χ²/dof computed with `var = counts` is
  inflated by that factor
- var/mean **below 1** → you did not remove the pedestal; the estimate is invalid

One integrating sCMOS measured `var = 2.23·y + 149`, which turned an adequate model (true
χ²/dof 1.08) into an apparently rejected one (2.6). Measure it **per detector** and never carry
one detector's gain onto another's frames — doing that inflated σ ~2.2× and flipped a result's
sign. Likelihood ratios and ROC/AUC are invariant and need no correction (rule 13,
Notebook §7b).

## 2b. Moment orientation → the per-pixel map

For a mosaicity (rock/roll) scan, the per-pixel centre-of-mass over the two tilt axes is the
orientation map:

```python
import numpy as np, torch
from midas_dfxm.mosaicity_fit import moment_orientation

# darling hands back data (a, b, m, n) and motors (2, m, n). moment_orientation wants
# data (P, M) and chi/phi each (M,) -- FLATTENED. You must reshape; see the trap below.
a, b, m, n = frames_bs.shape
d   = torch.as_tensor(frames_bs.astype(np.float32)).reshape(a * b, m * n)
chi = torch.as_tensor(motors[0]).reshape(-1)
phi = torch.as_tensor(motors[1]).reshape(-1)

com = moment_orientation(d, chi, phi)          # (P, 2): [:,0]=chi, [:,1]=phi
com = com.reshape(a, b, 2)                     # back to a detector-shaped map
```

> **TRAP — passing the un-reshaped arrays does not raise. It returns the wrong answer.**
> With `data (50,70,62,43)` and `chi/phi (62,43)`, torch broadcasts happily and returns
> `(50,70,62,2)` instead of `(P,2)`: a centre of mass taken over the *last motor axis only*.
> Verified 2026-08-12. The number that comes back looks reasonable and reports essentially
> no pedestal effect, so this fails the §2a check silently in the direction that makes you
> think the pedestal does not matter.
>
> **Assert the shape.** `assert com.shape == (a * b, 2)` before you use it.

> Verified against the installed package 2026-08-12: the signature is
> `moment_orientation(data, chi, phi)`. An earlier revision of this section showed
> `moment_orientation(frames_bs, angles)` unpacked into three values; that raises
> `TypeError` as written.

The intragranular misorientation magnitude is `hypot(com_fast - median, com_slow - median)`
over the grain mask (see `make_real_multibragg.py` for the exact idiom used on ID06). Report
the p95 spread, not the max.

**Cross-check against `darling`** on the *same background-subtracted frames* — this is an
arithmetic check (rule; Notebook §1b), so expect correlation → 1.0, RMS ~1e-7°. It validates
the estimator, **not** the physics.

> **HALT — import order. Doing this in the order this doc set prescribes segfaults the
> interpreter.** `darling.properties.moments` is numba-parallel-JIT'd. If `torch` has
> already been imported — which the spine's §0 import gate does, transitively, via
> `midas_dfxm` — the first call into `darling`'s JIT path dies with **SIGSEGV (exit 139),
> no traceback**. Reproduced three times, 2026-08-12.
>
> Run the `darling` cross-check in **its own process**, before or separately from anything
> that imports `midas_dfxm`/`torch`:
>
> ```bash
> python -c "from darling import assets, properties; ..."   # darling alone
> python -c "import midas_dfxm; ..."                        # midas_dfxm alone
> ```
>
> A crash with no traceback in a cross-check is easy to misread as bad data. It is not.

## 2c. Mosaicity: fit, do not just moment

The moment (and a phenomenological Gaussian) report the *measured* spread = intrinsic
mosaicity ⊛ instrument resolution. For the intrinsic spread, fit the physical forward with
the anisotropic resolution from §1c:

```python
import torch
from midas_dfxm.mosaicity_fit import fit_orientation_mosaicity

# res_cov is POSITIONAL; there is no `resolution=` keyword. Returns a dict.
fit = fit_orientation_mosaicity(
    torch.as_tensor(frames_bs), chi, phi, res_cov)                     # deconvolved
```

> Verified 2026-08-12:
> `fit_orientation_mosaicity(data: torch.Tensor, chi: torch.Tensor, phi: torch.Tensor,
> res_cov, *, n_components=1, lambda_smooth=0.0, shape=None, steps=500, lr=0.02,
> max_offset=1.5) -> dict`. The `resolution=` form an earlier revision showed does not
> exist.

Report the intrinsic mosaicity with the resolution kernel named (DIAGNOSIS: mosaicity_too_broad).

## 2d. Validate physical accuracy by injection-recovery — not round-trip

The public scans have no ground truth. **Do not** quote a forward/inverse round-trip (it
returns ~1e-16 because it inverts its own generator — a software-consistency metric, rule 5).
Instead resample a *known* orientation shift into the measured raw frames and recover it
against the real noise, background and detector:

- inject a known Δorientation → recover it through the full pipeline (subtract → moment)
- report the recovery **gain** (0.9998–1.0000 on the four ID03 scans, Notebook §1c)
- propagate counting statistics analytically and check against a Poisson Monte-Carlo
  (ratio ≈ 0.97), giving per-pixel σ (≈ 2 mdeg on ID03, ~20–40× finer than the 80 mdeg step)

## 2e. Strain (θ / energy) scans

A strain scan varies the Bragg angle / energy; the peak shift per pixel maps to d-spacing
(hence normal strain). Before reporting strain, apply the §4 checks: the **refraction gauge**
(a uniform offset is a reference, not a field) and the **~0.3 Λ** validity boundary. A raw
peak-shift strain map past 0.3 Λ is biased (§4b).

## 2f. Measure the per-pixel rocking width before any per-pixel model test

If anything downstream depends on how well the rocking curve is sampled — a lineshape fit,
model selection, a bimodality or doublet test — measure the width **on these frames, per
pixel**. Do not divide the step into a published or integrated FWHM: the integrated curve is
broadened by mosaic spread *across* pixels and ran **2.6–2.7× wider** than the per-pixel
median on one archived set, so points-per-FWHM comes out ~2.6× too optimistic. That single
error invalidated the premise of a whole preregistration (rule 12, Notebook §7a).

Use **argmax-local, contiguous** half-max crossings and check contiguity — global outermost
crossings let one noise spike set the width, and a non-contiguous above-half-max set spans the
gap between two disjoint islands. As a rough gate, a per-pixel model-selection test needs
≳ 12 points per FWHM; below that, moment-based statistics measure curve *broadness* rather
than shape (Notebook §5l).

Also confirm each scan's window **brackets its own maximum**. One fixed window reused across a
raster while θ_B drifts truncates some positions, biasing widths and integrals unequally
between channels of different width — and it can manufacture apparent two-population structure
downstream. Check the producing pipeline's intent first: edge frames are sometimes its dark
reference by design.

**Output of Phase 2:** per-pixel orientation (and, for a θ scan, normal-strain) maps for
*each reflection separately*, each on its own grid, plus the measured gain, the per-pixel
rocking width, and the background's θ-correlation. Fusing reflections is Phase 3, and only
if the gate passes.

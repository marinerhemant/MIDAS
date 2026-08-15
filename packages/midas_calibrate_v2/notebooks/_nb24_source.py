"""Cell source for notebook 24 — kept in its own file so _build.py stays
readable; imported by _build.py and registered in NOTEBOOKS.

Notebook 24: wavelength AND distance from a scan of KNOWN detector travel.
Self-contained synthetic ground truth (no $V2_TEST_BASE needed), with an
optional real-data section for 11-ID-C.
"""
from __future__ import annotations

from typing import List, Tuple

Cell = Tuple[str, str]

NB_24: List[Cell] = [
    ("md", """\
# 24 — Wavelength *and* Distance from a Scan of Known Travel

**The situation.** You do not know the wavelength exactly. You do not know
the sample-detector distance well. But you *do* know, very precisely, how far
the detector stage moved between images. Can you recover both?

**Yes — but not the way most people first try.**

The trap is worth stating up front, because "take a few distances and fit
them jointly" is the natural instinct and it does not work. A ring lands at

    R = L_sd · tan(2θ),   and at high energy   2θ ≈ λ / d

so the transformation

    λ → k·λ ,    L_sd,i → k·L_sd,i   (every image)

leaves **every predicted ring radius unchanged** to first order. Stacking more
images does not help, because each new image brings its own free `L_sd` to
rescale along with everything else. This is a *gauge* degeneracy, not a
shortage of data.

What breaks it is constraining the **differences**. If

    L_sd,i = L₀ + Δᵢ        with Δᵢ known exactly

then `k·L_sd,i` is no longer of that form unless `k = 1`. The whole scan now
carries a single distance unknown instead of one per image, and λ becomes
identifiable.

This notebook builds a synthetic scan with known truth, shows the degeneracy
is real, shows the constraint removes it, and ends with the two practical
traps that bit a real 107 keV dataset at APS 11-ID-C.
"""),

    ("md", """\
## 1. A synthetic distance scan with known truth

CeO₂, 107 keV, a Varex-like 2880² detector at 150 µm. The stage readback runs
600 → 2400 mm; the *true* distance is offset from it by `L0_TRUE`, which is
exactly the quantity a beamline never knows precisely.
"""),

    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
from scipy.optimize import least_squares
import torch

from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table, max_resolvable_ring_radius_px
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

LAM_TRUE = 0.11595        # A  (~106.9 keV)
A_CEO2   = 5.4116         # A  SRM 674b
PX       = 150.0          # um
NPIX     = 2880
L0_TRUE  = -1040.0        # um -- true Lsd = stage readback + L0
SIGMA_PX = 0.05           # per-point radial precision
MOTORS   = np.arange(600, 2401, 200, dtype=float) * 1000.0   # um, EXACT

def params(lsd_um, lam, max_ring_rad=1400.0):
    return CalibrationParams(
        NrPixelsY=NPIX, NrPixelsZ=NPIX, pxY=PX, pxZ=PX,
        Lsd=float(lsd_um), BC_y=NPIX/2, BC_z=NPIX/2, tx=0., ty=0., tz=0.,
        Wavelength=float(lam), SpaceGroup=225,
        LatticeConstant=(A_CEO2,)*3 + (90.,)*3,
        MaxRingRad=max_ring_rad, MinRingRad=200.0,
        RhoD=NPIX/2*PX, nIterations=1)

def synth(lsd_um, lam, rng, n_eta=48):
    rt = build_ring_table(params(lsd_um, lam))
    eta = np.linspace(-np.pi, np.pi, n_eta, endpoint=False)
    Y, Z, tt, dsp = [], [], [], []
    for R, t2, d in zip(rt.r_ideal_px, rt.two_theta_deg, rt.d_spacing):
        Rn = R + rng.normal(0., SIGMA_PX, n_eta)
        Y.append(NPIX/2 + Rn*np.cos(eta)); Z.append(NPIX/2 + Rn*np.sin(eta))
        tt.append(np.full(n_eta, t2));     dsp.append(np.full(n_eta, d))
    t = lambda a: torch.as_tensor(np.concatenate(a), dtype=torch.float64)
    return t(Y), t(Z), t(tt), t(dsp)

rng = np.random.default_rng(0)
scan = [synth(m + L0_TRUE, LAM_TRUE, rng) for m in MOTORS]
print(f'{len(scan)} images, stage {MOTORS.min()/1e3:.0f}-{MOTORS.max()/1e3:.0f} mm')
print(f'rings per image: {[len(build_ring_table(params(m+L0_TRUE, LAM_TRUE))) for m in MOTORS]}')
"""),

    ("md", """\
## 2. Fit it both ways

Same data, same optimiser, same residual. The *only* difference is the
parameterisation:

- **free** — one `L_sd` per image plus a shared λ (the instinctive approach)
- **linked** — a single `L₀` plus the known `Δᵢ`, plus a shared λ
"""),

    ("py", """\
def resid(pts, lsd_um, lam):
    Y, Z, tt, dsp = pts
    f = lambda v: torch.as_tensor(float(v), dtype=torch.float64)
    p = dict(Lsd=f(lsd_um), BC_y=f(NPIX/2), BC_z=f(NPIX/2), tx=f(0.), ty=f(0.),
             tz=f(0.), pxY=f(PX), pxZ=f(PX), Parallax=f(0.), Wavelength=f(lam))
    with torch.no_grad():
        r = pseudo_strain_residual(Y, Z, tt, p, rho_d=f(NPIX/2*PX),
                                   ring_d_spacing_A=dsp)
    return r.numpy()

deltas = MOTORS - MOTORS.mean()          # re-centred; L0 is then mid-scan

def fit_linked(data):
    fun = lambda x: np.concatenate([resid(p, x[0]*1e3 + d, x[1]*1e-3)
                                    for p, d in zip(data, deltas)])
    x0 = [(MOTORS.mean() + 5000.)/1e3, LAM_TRUE*1.004*1e3]   # 5 mm, 4000 ppm off
    s = least_squares(fun, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return s.x[0]*1e3 - MOTORS.mean(), s.x[1]*1e-3, s

def fit_free(data):
    n = len(data)
    fun = lambda x: np.concatenate([resid(p, x[i]*1e3, x[n]*1e-3)
                                    for i, p in enumerate(data)])
    x0 = [(m + L0_TRUE)/1e3 for m in MOTORS] + [LAM_TRUE*1.004*1e3]
    s = least_squares(fun, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return s.x[n]*1e-3, s

L0_f, lam_l, sol_l = fit_linked(scan)
lam_f, sol_f       = fit_free(scan)

print(f'truth            lambda = {LAM_TRUE:.6f} A   L0 = {L0_TRUE:+.1f} um')
print(f'linked  (L0+D)   lambda = {lam_l:.6f} A   L0 = {L0_f:+.1f} um   '
      f'({(lam_l/LAM_TRUE-1)*1e6:+.1f} ppm)')
print(f'free    (Lsd_i)  lambda = {lam_f:.6f} A                '
      f'({(lam_f/LAM_TRUE-1)*1e6:+.1f} ppm)')
print()
print(f'cond(J^T J)   linked = {np.linalg.cond(sol_l.jac.T@sol_l.jac):.2e}')
print(f'cond(J^T J)   free   = {np.linalg.cond(sol_f.jac.T@sol_f.jac):.2e}')
"""),

    ("md", """\
### Read the conditioning, not just the point estimate

On *noiseless* synthetic data the degeneracy is not mathematically exact — the
`tan`/`arcsin` nonlinearity does single out the true λ — so a determined
optimiser can crawl to the right answer either way. The degeneracy is an
**ill-conditioning** statement, and that is how it bites in practice: with
real noise, the free-`L_sd` λ wanders.

The honest test is therefore the *spread* over noise realisations, not one
number.
"""),

    ("py", """\
lam_linked, lam_free = [], []
for seed in range(4):
    r = np.random.default_rng(seed)
    d = [synth(m + L0_TRUE, LAM_TRUE, r) for m in MOTORS]
    lam_linked.append(fit_linked(d)[1])
    lam_free.append(fit_free(d)[0])

s_l = np.std(lam_linked)/LAM_TRUE*1e6
s_f = np.std(lam_free)/LAM_TRUE*1e6
print(f'sigma(lambda)/lambda over 4 noise seeds')
print(f'  linked : {s_l:8.2f} ppm')
print(f'  free   : {s_f:8.2f} ppm     ({s_f/max(s_l,1e-9):.0f}x worse)')
"""),

    ("md", """\
## 3. Doing it with the production pipeline

Everything above is the argument. In practice call `autocalibrate_multi` with
`lsd_offsets_um` — the exactly-known travel, one value per image, in µm and on
any common origin (the raw stage readback is fine; it is re-centred
internally). That switches on the linked mode: a single shared refined `L₀`,
a shared refined `Wavelength`, and per-image beam centre and tilts.

```python
from midas_calibrate_v2.pipelines.multi import autocalibrate_multi

res = autocalibrate_multi(
    v1_per_image, images,
    lsd_offsets_um=[m * 1000.0 for m in motor_mm],   # EXACT travel
    n_iter=3, verbose=False,
)
print(res.L0_um, res.lsd_per_image_um)
print(float(res.shared_unpacked['Wavelength']))
```

or from the command line:

```bash
midas-calibrate-v2 ps.txt --mode multi \\
    --images  d600.tif d800.tif d1000.tif \\
    --paramsfiles ps600.txt ps800.txt ps1000.txt \\
    --lsd-offsets 600000 800000 1000000
```

`res.shared_unpacked['Lsd']` is `L₀`, **not** a usable per-image distance —
read those from `res.lsd_per_image_um`.
"""),

    ("md", """\
## 4. The slope-1 diagnostic

There is a cheaper route that needs no new machinery and makes a very good
sanity check on the joint fit. Calibrate each image independently with λ
**pinned**, then regress the fitted `L_sd` against the known travel. If the
pinned λ is wrong by a factor `k = λ_true/λ_pinned`, every fitted distance is
wrong by the same factor, so

    L_sd,i(fit) = k · (L₀ + Δᵢ)

- **slope** = k          → λ_true = k · λ_pinned
- **intercept/slope** = L₀

**The correct wavelength is the one that makes the slope exactly 1.** Iterate
if the first pass moves it much; the small-angle argument is not exact.

The residuals about that line are the real prize. They are not noise — they
are your instrument. Structure there means a distance-dependent systematic,
and the next cell shows how to tell *what kind*.
"""),

    ("py", """\
lam_pinned = LAM_TRUE * 1.004      # deliberately 4000 ppm wrong

fitted = []
for m in MOTORS:
    pts = synth(m + L0_TRUE, LAM_TRUE, np.random.default_rng(int(m)))
    fun = lambda x: resid(pts, x[0]*1e3, lam_pinned)
    s = least_squares(fun, [(m + L0_TRUE)/1e3], xtol=1e-14, ftol=1e-14)
    fitted.append(s.x[0]*1e3)
fitted = np.array(fitted)

A = np.column_stack([MOTORS, np.ones_like(MOTORS)])
beta, *_ = np.linalg.lstsq(A, fitted, rcond=None)
k = beta[0]
print(f'pinned lambda      = {lam_pinned:.6f} A')
print(f'slope k            = {k:.7f}')
print(f'  -> lambda_true   = {k*lam_pinned:.6f} A  (truth {LAM_TRUE})')
print(f'  -> L0            = {beta[1]/k:+.1f} um   (truth {L0_TRUE:+.1f})')
print(f'residual rms       = {(fitted - A@beta).std(ddof=2):.2f} um')
"""),

    ("md", """\
### Telling a stage error from a detector error, with no model at all

If those residuals are structured, one measurement separates the two causes.
Take the *measured* radius of each individual ring at each distance and
regress it against the travel. Then:

| cause | signature |
|---|---|
| distance / stage error `δL` | `δR/R = δL/L` — the **same fractional** shift for every ring. Curves collapse against *stage position*, separate against *detector radius*. |
| detector radial error `f(R)` | `δR = f(R)` — a **different fractional** shift per ring. Curves collapse against *detector radius*, separate against *stage position*. |

On the real 11-ID-C scan this gave a common signal of 418 ppm against a
ring-to-ring spread of 54 ppm — 8:1 in favour of a distance-like error.

**One caveat that dataset also taught:** at small 2θ, `δR/R = δL/L` and
`δR/R = δλ/λ` are the *same observable* (the separating nonlinearity is ~2.5
ppm on a 1000 ppm signal). So if your scan runs monotonically in distance,
distance and time are confounded and a slow beam-energy drift is
indistinguishable from a stage error. **Randomise the distance order**, or
interleave repeats of one distance, and the confound disappears.
"""),

    ("md", """\
## 5. Two traps this exercise exists to warn you about

### Ring overlap — check before you fit

A ring table comes from crystallography and knows nothing about your pixel
pitch. For a dense calibrant at short wavelength it can be far denser than the
detector resolves, and the peak fitter will happily fit unresolved blends: the
fit "succeeds", the residuals look structureless, and the geometry is quietly
biased.
"""),

    ("py", """\
for m_mm in (330, 500, 1000, 2000, 3000):
    rt = build_ring_table(params(m_mm*1000.0, LAM_TRUE, max_ring_rad=1420.0))
    r_cap, n_ok = max_resolvable_ring_radius_px(rt, min_separation_px=8.0,
                                                r_min_px=200.0)
    cap = f'{r_cap:7.1f}' if r_cap else '     --'
    print(f'Lsd {m_mm:5d} mm: {len(rt):4d} rings in table, '
          f'{n_ok:3d} resolvable, cap {cap} px')
"""),

    ("md", """\
At 330 mm the table holds hundreds of rings inside 1420 px — roughly one every
2 px — and only a couple are genuinely separable. Because ring separation in
pixels scales with `L_sd`, **the shortest distance in a scan sets the usable
2θ ceiling for the whole thing**; for CeO₂ at 107 keV that ceiling is about
6.4°, set by the (331)/(420) pair at 5.353°/5.492°.

Capping the ring set accordingly took the real dataset from 300–600 µε per
image down to 13–36 µε.

### Good per-image strain does not mean the geometry is right

On that same dataset, applying the cap improved per-image strain by an order
of magnitude and left the distance residual **completely unchanged** (630 →
625 µm rms, curves lying on top of each other). Strain is a within-image
statistic; it is blind to an error shared by every image. Judge the geometry
with the slope-1 residual, not with strain.

## Summary

- Several distances with free `L_sd` does **not** make λ identifiable.
  Known travel does.
- Use `autocalibrate_multi(..., lsd_offsets_um=...)`, or `--lsd-offsets`.
- Cross-check with the slope-1 regression; read its residuals.
- Cap the ring set with `max_resolvable_ring_radius_px` before fitting.
- Randomise the distance order so a beam drift cannot masquerade as a
  stage error.
- **Hard floor:** if the travel is really `L₀ + s·Δᵢ` with the stage scale `s`
  unknown, λ and `s` are exactly degenerate — σ(λ)/λ ≥ σ(s)/s. Trust in λ is
  trust in the encoder. And λ is only ever measured relative to the adopted
  calibrant lattice constant.
"""),
]

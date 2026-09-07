"""Cell source for notebook 26 — kept in its own file so _build.py stays
readable; imported by _build.py and registered in NOTEBOOKS.

Notebook 26: a focused, standalone walkthrough of the frozen-point
calibration pipeline (`autocalibrate_frozen_point`) on a real large-tilt
(tz~14 deg, beam centre off-detector) CeO2 dataset — recovering the
geometry from a rough tilt guess, a tilt+distortion-correct visual ring
overlay, and a distortion-basis-selection comparison (no distortion vs.
isotropic-only vs. the full harmonic basis). This notebook does not re-run
or compare against the alternating pipelines (see notebooks 01/09 for
those).
"""
from __future__ import annotations

from typing import List, Tuple

Cell = Tuple[str, str]

NB_26: List[Cell] = [
    ("md", """\
# 26 — Frozen-Point Calibration for Large-Tilt Detectors

The alternating pipelines in this package (`autocalibrate_pv` and friends)
extract peak positions at the *current* trial geometry, refit, and
re-extract at the new geometry — repeated for a few outer iterations. When
the starting geometry is far enough off (a large, only roughly-known
detector tilt with the beam centre off the active area is a good example),
that re-extraction can lock onto the wrong ring correspondences and the
outer loop then "honestly" converges to the wrong answer, because each
iteration's fit is only as good as the previous iteration's extraction.

`autocalibrate_frozen_point` sidesteps this by extracting once: it finds
genuine local-intensity maxima directly on the raw image (no azimuthal
binning, no windowed re-sampling at a moving trial geometry), freezes that
point cloud, and does a single bounded least-squares fit against it. No
re-extraction step exists for a bad early estimate to compound through.
This notebook is a focused walkthrough of that one pipeline on a genuinely
hard real dataset — it does not re-run or compare against the alternating
pipelines (see notebooks **01**/**09** for those).

By the end of this notebook you will have:

1. Loaded a real large-tilt CeO₂ dataset (beam centre off the detector,
   tz ≈ 14°) as the worked example.
2. Run `autocalibrate_frozen_point` from a realistic rough tilt guess and
   recovered the true geometry.
3. Visually confirmed the result by overlaying the tilt- and
   distortion-corrected predicted ring positions on the raw image.
4. Escaped a **completely blind** starting guess (`tz=0`, no tilt
   information at all) using `iterate_frozen_point_until_stable` — a
   re-seeding wrapper around `autocalibrate_frozen_point`, with a strict
   parameter-stability convergence criterion validated (offline, 13-point
   sweep) to reach the correct basin from starting guesses up to ~35°
   apart, with zero false-basin acceptances.
5. Compared a geometry-only fit against several distortion-basis choices
   (isotropic-only vs. the full 15-coefficient harmonic basis), using this
   package's existing freeze/thaw convention on `CalibrationSpec` — the
   same mechanism every other pipeline in this package already uses for
   that choice — and read the results off one comparison table.

Wall time: under a minute for steps 1-3 and 5; the blind-start demo in
step 4 re-seeds and refits repeatedly and takes on the order of a minute
by itself (still far cheaper than a manual multi-start grid search).

## Pre-flight: locate the dataset

This notebook uses one real experiment's calibration data as the worked
example. Point `DATA_DIR` at your own project layout to run this against a
different dataset — you need three things: detector/beam metadata (pixel
size, wavelength, calibrant), a dark-subtracted/frame-averaged image, and a
rough starting geometry (beam centre + sample-to-detector distance; the
tilt-angle guess is supplied explicitly in this notebook, see below).
"""),
    ("py", """\
import os, json
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import numpy as np

DATA_DIR = Path(os.environ.get(
    'LARGE_TILT_CALIB_DIR',
    '/home/beams0/DBENIWAL/MIDAS/scratch_calibrate/large_tilt_calib'))

with open(DATA_DIR / 'exp_info.json') as f:
    exp_info = json.load(f)          # this project's own metadata bookkeeping,
                                      # not a midas_calibrate_v2 file format
with open(DATA_DIR / 'auto_seed' / 'auto_seed.json') as f:
    auto_seed = json.load(f)['seed']  # tilt-blind BC/Lsd only -- no tz guess

# Pre-processed: average of 10 paired-dark-subtracted frames, negative-clipped.
# Substitute your own similarly-preprocessed frame for a different dataset.
image = np.load(DATA_DIR / 'v1' / 'avg_dark_subtracted.npy').astype(np.float64)

NY = exp_info['detector']['n_pixels_y']
NZ = exp_info['detector']['n_pixels_z']
PX_UM = exp_info['detector']['pixel_pitch_um']
WAVELENGTH_A = exp_info['beam']['trusted_wavelength_angstrom']
CALIBRANT = exp_info['dataset']['calibrant']

print(f'image: {image.shape}, dtype={image.dtype}')
print(f'detector: {NY}x{NZ} px, {PX_UM} um pixels')
print(f'calibrant: {CALIBRANT}, wavelength={WAVELENGTH_A} A')
print(f'auto-seed (tilt-blind): BC=({auto_seed[\"BC_y\"]:.2f}, {auto_seed[\"BC_z\"]:.2f}) px, '
      f'Lsd={auto_seed[\"Lsd_um\"]/1000.0:.2f} mm')
"""),
    ("md", """\
## Why this dataset is a hard case

The beam centre sits off the active detector area and the detector tilt is
large (~14°). A "tilt-blind" seed like `auto_seed` above can only measure
BC/Lsd from low-angle ring curvature; it has no information about the tilt
angle at all. Some non-zero, only roughly-known tilt guess is unavoidable —
below we deliberately use a **rough, round-number** guess (15°) rather than
a precisely pre-measured value, to demonstrate realistic usage rather than
a best-case scenario.
"""),
    ("py", """\
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.seed.calibrant import resolve_calibrant

cal = resolve_calibrant(CALIBRANT)

def max_ring_rad_px(bc_y, bc_z, margin=20.0):
    corners = [(0, 0), (NY, 0), (0, NZ), (NY, NZ)]
    return max(np.hypot(cy - bc_y, cz - bc_z) for cy, cz in corners) + margin

TZ_ROUGH_GUESS = 15.0   # a rough round-number estimate, not the precise answer

v1_seed = V1Params(
    NrPixelsY=NY, NrPixelsZ=NZ, pxY=PX_UM, pxZ=PX_UM,
    Lsd=auto_seed['Lsd_um'], BC_y=auto_seed['BC_y'], BC_z=auto_seed['BC_z'],
    tx=0.0, ty=0.0, tz=TZ_ROUGH_GUESS,
    Wavelength=WAVELENGTH_A, SpaceGroup=cal['sg'],
    LatticeConstant=(cal['a'], cal['b'], cal['c'], cal['alpha'], cal['beta'], cal['gamma']),
    MaxRingRad=max_ring_rad_px(auto_seed['BC_y'], auto_seed['BC_z']),
    Refine={'Lsd': True, 'BC': True, 'ty': True, 'tz': True,
            'Wavelength': False, 'Parallax': False,
            **{f'p{i}': False for i in range(15)}},
)
v1_seed.validate()
print('seed geometry built OK')
"""),
    ("md", """\
## Frozen-point calibration

`autocalibrate_frozen_point` picks its point cloud once (a one-shot,
geometry-independent 2D local-intensity-maximum search — see
`forward/point_pick.py`) and never re-extracts. `tx` is force-frozen
inside the pipeline itself: a rotation about the beam axis only changes a
ring's azimuthal (eta) labelling, never its radius, so it is unobservable
from a single powder image's ring radii — a genuine gauge freedom, not a
numerical artifact.

We start with all 15 distortion coefficients frozen (a geometry-only fit;
distortion selection comes back in the last section) and give the LM solver
generous bounds around the seed.
"""),
    ("py", """\
import time

from midas_calibrate_v2.pipelines.frozen_point import autocalibrate_frozen_point
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_params
from midas_calibrate_v2.forward.distortion import P_COEF_NAMES

spec_geom_only = spec_from_v1_params(v1_seed)
spec_geom_only.freeze(*P_COEF_NAMES)
spec_geom_only.parameters['BC_y'].bounds = (v1_seed.BC_y - 100.0, v1_seed.BC_y + 100.0)
spec_geom_only.parameters['BC_z'].bounds = (v1_seed.BC_z - 100.0, v1_seed.BC_z + 100.0)
spec_geom_only.parameters['Lsd'].bounds = (v1_seed.Lsd - 30_000.0, v1_seed.Lsd + 30_000.0)
spec_geom_only.parameters['ty'].bounds = (v1_seed.ty - 10.0, v1_seed.ty + 10.0)
spec_geom_only.parameters['tz'].bounds = (v1_seed.tz - 16.0, v1_seed.tz + 16.0)

t0 = time.time()
res_frozen = autocalibrate_frozen_point(
    v1_seed, image, spec=spec_geom_only, snr_min=5.0,
    point_pick_kwargs=dict(downsample=4, footprint_px=7, snr_threshold=5.0,
                            min_ring_gap_deg=0.3, subpixel_half_px=2),
    lm_max_iter=150, verbose=False,
)
elapsed_frozen = time.time() - t0
fit_frozen = res_frozen.history[0]
print(f'elapsed: {elapsed_frozen:.1f} s')
print(f'Lsd={fit_frozen.Lsd:.1f} um  BC=({fit_frozen.BC_y:.2f},{fit_frozen.BC_z:.2f}) px  '
      f'ty={fit_frozen.ty:.3f} deg  tz={fit_frozen.tz:.3f} deg')
print(f'mean pseudo-strain: {fit_frozen.mean_strain_uE:.1f} microstrain')
"""),
    ("md", """\
This single non-alternating fit converges in about a second and lands at a
low pseudo-strain — and, as the overlay below confirms, a geometry that
actually tracks the measured rings.
"""),
    ("md", """\
## Visual check: predicted rings over the raw image

As a sanity check (not the quantitative arbiter — pseudo-strain above is),
forward-project each ring's ideal 2θ through the *full* tilted-detector
geometry (not a zero-tilt circle — at 14° tilt the correct curve is an
off-centre ellipse), and overlay the predicted curve on the raw image. We
restrict to the same well-isolated, low-2θ rings the frozen-point fit
actually used (`min_ring_gap_deg` above dropped the rest as too close to a
neighbour to trust) so the plot isn't cluttered with dozens of weak,
effectively invisible high-angle reflections.

The rings are drawn via this codebase's standard recipe for a
calibration-accurate overlay — write the fitted result to a v1 paramstest,
read it back as a `midas_integrate_v2` `IntegrationSpec`, and evaluate that
package's own per-pixel forward model (`eval_pixel_REta`) over the whole
detector. That model includes the fitted tilt **and** the full 15-term
distortion basis, so the overlay reflects every correction the calibration
actually applied, not just a zero-tilt/zero-distortion circle. The ring
curves themselves are true iso-contours of that per-pixel field (via
`contourpy`), not a scatter of individually-inverted `(R, η)` samples — so
they render as continuous curves with no per-point inversion noise. The
figure is a `plotly` widget rendered inline: drag to zoom into any region
(e.g. a single ring crossing) and double-click to reset.
"""),
    ("py", """\
import math

import contourpy
import plotly.graph_objects as go

from midas_calibrate.rings import build_ring_table
from midas_calibrate_v2.compat.to_v1 import write_v1_paramstest
from midas_integrate_v2.compat.from_v1 import spec_from_v1_paramstest
from midas_integrate_v2.forward.pixels import eval_pixel_REta

rt_for_plot = build_ring_table(v1_seed)
two_thetas_to_plot = sorted(set(rt_for_plot.two_theta_deg.tolist()))[:20]

def ring_overlay_lines(res, tag):
    \"\"\"Continuous (Y, Z) ring polylines for a fitted result, tilt +
    distortion included.

    Writes the fit to a v1 paramstest and reads it back as a v2
    `IntegrationSpec` so we evaluate the exact per-pixel forward model
    `midas_integrate_v2` itself uses (`eval_pixel_REta`), then takes true
    iso-contours of that field at each ring's ideal flat-panel radius --
    the standard, non-hand-rolled way to draw a MIDAS ring overlay.
    Returns None-separated (Y, Z) arrays for one `go.Scattergl(mode='lines')`
    trace, plus the fitted beam centre.
    \"\"\"
    path = f'_nb26_{tag}_paramstest.txt'
    write_v1_paramstest(res.unpacked, v1_seed, path)
    spec = spec_from_v1_paramstest(path)
    R = eval_pixel_REta(spec)[0].detach().numpy()   # (NZ, NY) px

    ideal_radii_px = sorted(set(
        float(spec.Lsd) / float(spec.pxY) * math.tan(math.radians(tt))
        for tt in two_thetas_to_plot))   # dedupe: distinct rings can share a two_theta

    cg = contourpy.contour_generator(z=R, line_type=contourpy.LineType.Separate)
    yy, zz = [], []
    for level in ideal_radii_px:
        for line in cg.lines(level):
            yy.extend(line[:, 0].tolist() + [None])
            zz.extend(line[:, 1].tolist() + [None])
    return yy, zz, float(spec.BC_y), float(spec.BC_z)

def downsample_for_display(img, factor=4):
    \"\"\"Block-average `img` by `factor` for display only; bin centres stay
    in the ORIGINAL pixel coordinate frame so the full-resolution ring
    overlay still lands right on top of the downsampled image.
    \"\"\"
    nz, ny = img.shape
    nz2, ny2 = nz - nz % factor, ny - ny % factor
    small = img[:nz2, :ny2].reshape(nz2 // factor, factor, ny2 // factor, factor).mean(axis=(1, 3))
    y_centers = (np.arange(ny2 // factor) + 0.5) * factor - 0.5
    z_centers = (np.arange(nz2 // factor) + 0.5) * factor - 0.5
    return np.round(small).astype(int), y_centers, z_centers

img_disp, img_y, img_z = downsample_for_display(image, factor=4)
vmax = float(np.percentile(image[image > 0], 99.5)) if (image > 0).any() else 1.0

yy, zz, bc_y, bc_z = ring_overlay_lines(res_frozen, 'frozen')

fig = go.Figure()
fig.add_trace(go.Heatmap(z=img_disp, x=img_y, y=img_z, colorscale='gray_r',
                          zmin=0, zmax=vmax, showscale=False, hoverinfo='skip'))
# Thin (but 2x more visible than the first pass), semi-transparent lines:
# the raw-intensity rings underneath must stay visible through the
# predicted overlay, not be hidden by it. Scattergl (not Scatter): tens of
# thousands of contour points across ~20 rings, and WebGL rendering is what
# keeps interactive zoom/pan smooth at that point count.
fig.add_trace(go.Scattergl(x=yy, y=zz, mode='lines',
                            line=dict(color='red', width=1.2), opacity=0.55,
                            name='predicted rings', hoverinfo='skip'))
fig.add_trace(go.Scatter(x=[bc_y], y=[bc_z], mode='markers',
                          marker=dict(symbol='cross-thin', size=12,
                                      color='deepskyblue', line=dict(width=2)),
                          name='beam centre', hoverinfo='skip'))
fig.update_layout(
    width=900, height=900,
    title_text=(f'autocalibrate_frozen_point ring overlay '
                f'(tz={fit_frozen.tz:.2f} deg, {fit_frozen.mean_strain_uE:.0f} microstrain) '
                '— drag to zoom, double-click to reset'),
    xaxis=dict(title='Y (px)', range=[0, NY], constrain='domain'),
    yaxis=dict(title='Z (px)', range=[NZ, 0], scaleanchor='x'),
)
fig.show()
"""),
    ("md", """\
The predicted rings track the visible arcs well at this whole-detector
scale — which is itself the point: a systematic few-pixel-per-ring offset
of the kind that separates a genuinely correct geometry from a
self-consistent-but-wrong one is generally **not** visible by eye at full
detector scale. That is exactly why this package's pipelines report
pseudo-strain (and, for `autocalibrate_pv`, formal reliability gates — see
notebook **01** step 5) rather than relying on an overlay-by-eye — use the
interactive zoom above if you want to check sub-pixel agreement at a
specific ring crossing more rigorously than a whole-detector view allows.
"""),
    ("md", """\
## Escaping a genuinely blind starting guess

The fit above still needed *some* rough tilt guess (15°, ~1° off the true
answer) — `autocalibrate_frozen_point` is one-shot: it picks its point
cloud once, at whatever geometry it's handed, and never re-extracts.
If that starting geometry is far enough off, the one-shot extraction can
still miss or mislabel rings, same as any other method.

`iterate_frozen_point_until_stable` removes the need for a rough guess at
all by **re-seeding `autocalibrate_frozen_point` from its own result,
repeatedly** — each iteration builds a fresh geometry-only spec with
Lsd/BC/ty/tz bounds *re-centered* on the previous iteration's converged
geometry (not a fixed window), which is what lets the estimate walk many
degrees from a bad start. It stops as soon as a trailing window of 5
consecutive iterations agree to within 0.01° (tz, ty), 0.05 px (BC), and
5 µm (Lsd) of each other — a **strict parameter-stability** criterion,
deliberately, with no strain gate.

That strictness matters. On this exact dataset, a *loose* stability
tolerance produces a false positive: iterations partway through a genuine
escape trajectory sit in a plateau where tz creeps by only ~0.1–0.6° per
step for 8+ iterations in a row — stable enough to fool a loose
criterion — while still being ~10–12° from the true answer (pseudo-strain
~30× too high). The tight tolerances above have a two-order-of-magnitude
margin against that plateau's per-step movement, so they only fire once
the fit has landed on a genuine, bit-reproducing fixed point.

Below, from a completely blind `tz=0` (no tilt information at all, not
even the rough 15° guess used above):
"""),
    ("py", """\
import dataclasses as _dc

from midas_calibrate_v2.pipelines.frozen_point import iterate_frozen_point_until_stable


# NOTE: autocalibrate_frozen_point mutates its v1_params argument in place
# (it writes the fitted geometry back onto it), so by this point in the
# notebook v1_seed's Lsd/BC/ty/tz already hold the CORRECT refined
# geometry from the fit above -- not the original tilt-blind auto_seed.
# A genuinely blind test must reset BC/Lsd back to auto_seed's tilt-blind
# values too, not just zero tz/ty on top of an already-correct BC/Lsd.
v1_blind = _dc.replace(
    v1_seed, BC_y=auto_seed['BC_y'], BC_z=auto_seed['BC_z'],
    Lsd=auto_seed['Lsd_um'], ty=0.0, tz=0.0,
)

t0 = time.time()
out_blind = iterate_frozen_point_until_stable(
    v1_blind, image, verbose=True,
    point_pick_kwargs=dict(downsample=4, footprint_px=7, snr_threshold=5.0,
                            min_ring_gap_deg=0.3, subpixel_half_px=2),
)
elapsed_blind = time.time() - t0

fit_blind = out_blind.fit
print(f'\\nconverged={out_blind.converged} after {out_blind.n_iter} iterations, '
      f'{elapsed_blind:.1f} s total')
print(f'Lsd={fit_blind.Lsd:.1f} um  BC=({fit_blind.BC_y:.2f},{fit_blind.BC_z:.2f}) px  '
      f'ty={fit_blind.ty:.3f} deg  tz={fit_blind.tz:.3f} deg')
print(f'mean pseudo-strain: {fit_blind.mean_strain_uE:.1f} microstrain')
print(f'\\nagreement with the tz=15deg-seeded result above: '
      f'Δtz={abs(fit_blind.tz - fit_frozen.tz):.4f} deg, '
      f'ΔLsd={abs(fit_blind.Lsd - fit_frozen.Lsd) / 1000:.4f} mm')
"""),
    ("md", """\
From `tz=0`, with no tilt information at all, this lands on the identical
geometry as the informed-seed fit above.

This was characterized more broadly in an **offline sweep** (13 starting
`tz` guesses, not re-run live here — 35 iterations × 13 starts is several
minutes; see `scratch_calibrate/large_tilt_calib/best_solution/README.md`
for the full methodology and numbers):

| starting tz | converged? | iterations | final tz |
|---|---|---|---|
| -15° | No | 35 (cap) | -16.33° |
| -10° | No | 35 (cap) | -7.16° |
| -5° | **Yes** | 33 | 13.9897° |
| 0° | **Yes** | 20 | 13.9897° |
| 5°–25° | **Yes** | 5–7 | 13.9897° |
| 30° | **Yes** | 27 | 13.9897° |
| 40° | No | 35 (cap) | 43.53° |

Every convergent start reached the *exact same, bit-identical* fixed
point — zero false-basin acceptances — across a ~35° capture range
(`tz0 ∈ [-5°, +30°]`); outside it, the function correctly reports
`converged=False` rather than a silently wrong answer. This is
substantially wider than every other method tried on this dataset (the
alternating pipeline's single-shot basin is ~5–6° wide; a single
frozen-point shot without iteration is ~19° wide) — but it has only been
validated on this one real dataset so far, so treat the capture range as
dataset-specific until confirmed elsewhere.
"""),
    ("md", """\
## Choosing a distortion model

This package has no separate "distortion model" flag — there is one fixed
15-coefficient radial-distortion basis (`iso_R2, iso_R4, iso_R6` isotropic
terms, plus 6 two-fold-to-six-fold harmonic pairs `a_k`/`phi_k`), and
"choosing a model" means freezing or thawing named coefficients on the
`CalibrationSpec` — exactly the convention notebook **09** uses for
`autocalibrate_pv`. `autocalibrate_frozen_point` uses the same
`CalibrationSpec` machinery, so the identical convention applies here.
(The automatic `select_basis_bic`/`auto_select_basis` helpers are wired
only to `autocalibrate_pv` today; below we select manually, which is how
every *other* pipeline in this package already does it.)

We compare three choices, starting from the frozen-point fit's own
recovered geometry so each variant's LM step is short:
- **geometry-only** — all 15 coefficients frozen at 0 (what we ran above).
- **isotropic-only** — only `iso_R2/iso_R4/iso_R6` refined.
- **full harmonic basis** — all 15 coefficients refined.
"""),
    ("py", """\
import dataclasses

from midas_calibrate_v2.forward.distortion import ISO_NAMES

v1_from_frozen = dataclasses.replace(
    v1_seed, Lsd=fit_frozen.Lsd, BC_y=fit_frozen.BC_y, BC_z=fit_frozen.BC_z,
    ty=fit_frozen.ty, tz=fit_frozen.tz,
)

basis_results = {}   # name -> (res, IterRecord, k refined)

def run_with_basis(name, thaw_names):
    spec = spec_from_v1_params(v1_from_frozen)
    spec.freeze(*P_COEF_NAMES)
    if thaw_names:
        spec.thaw(*thaw_names)
    spec.parameters['BC_y'].bounds = (fit_frozen.BC_y - 20.0, fit_frozen.BC_y + 20.0)
    spec.parameters['BC_z'].bounds = (fit_frozen.BC_z - 20.0, fit_frozen.BC_z + 20.0)
    spec.parameters['Lsd'].bounds = (fit_frozen.Lsd - 5_000.0, fit_frozen.Lsd + 5_000.0)
    spec.parameters['ty'].bounds = (fit_frozen.ty - 2.0, fit_frozen.ty + 2.0)
    spec.parameters['tz'].bounds = (fit_frozen.tz - 2.0, fit_frozen.tz + 2.0)

    res = autocalibrate_frozen_point(
        v1_from_frozen, image, spec=spec, snr_min=5.0,
        point_pick_kwargs=dict(downsample=4, footprint_px=7, snr_threshold=5.0,
                                min_ring_gap_deg=0.3, subpixel_half_px=2),
        lm_max_iter=150, verbose=False,
    )
    fit = res.history[0]
    k = len(spec.refined_names())
    print(f'{name:16s} k={k:2d} refined  strain={fit.mean_strain_uE:8.2f} microstrain  '
          f'tz={fit.tz:.3f} deg  rc={fit.rc}')
    basis_results[name] = (res, fit, k)
    return res

_ = run_with_basis('geometry-only', ())
_ = run_with_basis('isotropic-only', ISO_NAMES)
_ = run_with_basis('full harmonic', P_COEF_NAMES)
"""),
    ("md", """\
With this notebook's own choice of rings (14-15 well-isolated, low-2θ
reflections, per `min_ring_gap_deg=0.3` above), the distortion coefficients
are only weakly constrained — the low-angle-only ring set that makes the
frozen-point extraction reliable here does not by itself carry much
distortion information. If your calibration needs a well-constrained
distortion basis, prefer `autocalibrate_pv`'s full ring coverage for that
final refinement once the frozen-point fit has established the correct
geometry basin — the two pipelines are complementary, not exclusive: use
`autocalibrate_frozen_point`'s result as a `spec=`/seed to a subsequent
`autocalibrate_pv` call, exactly as you would seed it from any other source.
"""),
    ("md", """\
## Summary

The table below compares the three distortion-basis choices from the
previous section, side by side — all three fits start from the identical
established geometry (`v1_from_frozen`) and identical point cloud settings,
so any difference is attributable to the distortion basis alone.
"""),
    ("py", """\
import pandas as pd

rows = []
for name, (res, fit, k) in basis_results.items():
    rows.append({
        'mode': name,
        'distortion coeffs refined': k - 5,   # k minus the 5 always-on geometry params
        'total params refined': k,
        'n points fitted': fit.n_fitted,
        'Lsd (mm)': fit.Lsd / 1000.0,
        'BC_y (px)': fit.BC_y,
        'BC_z (px)': fit.BC_z,
        'ty (deg)': fit.ty,
        'tz (deg)': fit.tz,
        'mean strain (microstrain)': fit.mean_strain_uE,
        'rc': fit.rc,
    })
summary_df = pd.DataFrame(rows).set_index('mode')
summary_df.round(4)
"""),
    ("md", """\
**When to reach for `autocalibrate_frozen_point`**: a large or only
roughly-known detector tilt, an off-detector beam centre, or any case
where an alternating pipeline like `autocalibrate_pv` converges "cleanly"
but to an implausible pseudo-strain or geometry — the single-shot,
no-re-extraction design removes the feedback loop that traps alternating
pipelines in a self-consistent wrong answer. Once a good geometry basin is
established this way, `autocalibrate_pv` (or the distortion-basis tools in
notebook **09**) remains the right tool for a full, distortion-inclusive
final refinement across the detector's whole ring coverage.
"""),
]

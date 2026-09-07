"""Cell source for notebook 26 — kept in its own file so _build.py stays
readable; imported by _build.py and registered in NOTEBOOKS.

Notebook 26: a focused, standalone walkthrough of the frozen-point
calibration pipeline (`autocalibrate_frozen_point` /
`iterate_frozen_point_until_stable`) on a real large-tilt (tz~14 deg, beam
centre off-detector) CeO2 dataset. Loads the raw HDF5 frames directly and
defines all metadata (wavelength, pixel size, seed geometry) inline --
no exp_info.json / auto_seed.json / pre-averaged .npy dependency.
"""
from __future__ import annotations

from typing import List, Tuple

Cell = Tuple[str, str]

NB_26: List[Cell] = [
    ("md", """\
# 26 — Frozen-Point Calibration for Large-Tilt Detectors

**Why**: this package's alternating pipelines (`autocalibrate_pv`) re-extract
peak positions at each trial geometry. If the starting tilt guess is far
off, a biased extraction can compound across iterations into a
self-consistent but *wrong* answer.

**What**: `midas_calibrate_v2.pipelines.frozen_point` provides two fixes —
`autocalibrate_frozen_point` (extracts peaks **once**, directly on the raw
image, then a single bounded fit — no re-extraction loop to compound
through) and `iterate_frozen_point_until_stable` (re-seeds that one-shot
fit from its own result until the geometry stops moving, recovering the
correct answer even from a **completely blind** tilt guess).

**How**: below — load the raw frames, fit from a rough tilt guess, verify
with a ring overlay, then repeat from a blind guess and compare distortion
models. Wall time ~2 min.
"""),
    ("py", """\
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

DATA_DIR = Path(os.environ.get(
    'LARGE_TILT_CALIB_DIR',
    '/home/beams0/DBENIWAL/MIDAS/scratch_calibrate/large_tilt_calib'))
H5_PATH = DATA_DIR / 'data' / 'CeO2_RT_pirex_1000mm_71p676keV_022104.vrx.h5'

# Detector + beam metadata for this dataset (Varex varexD 2880x2880, CeO2
# powder calibrant). Defined inline -- no metadata file dependency.
NY, NZ = 2880, 2880
PX_UM = 150.0
WAVELENGTH_A = 0.172979   # from the filename-encoded 71.676 keV
CALIBRANT = 'CeO2'

# Tilt-blind beam-centre/L_sd estimate (low-angle ring-curvature fit; no
# tilt information at all).
SEED_BC_Y = 3044.262076653854
SEED_BC_Z = 1344.2233042856406
SEED_LSD_UM = 1054947.0296702771

# Average 9 of the 10 paired-dark-subtracted frames (frame 0 excluded: a
# shutter-settling transient, ~8x weaker than frames 1-9 despite matching
# raw levels), negative-clipped.
with h5py.File(H5_PATH, 'r') as f:
    diff = f['exchange/data'][()].astype(np.float64) - f['exchange/data_dark'][()].astype(np.float64)
image = diff[1:].mean(axis=0)
image[image < 0] = 0.0

print(f'image: {image.shape}, dtype={image.dtype}')
print(f'detector: {NY}x{NZ} px, {PX_UM} um pixels, calibrant: {CALIBRANT}, '
      f'wavelength={WAVELENGTH_A} A')
print(f'tilt-blind seed: BC=({SEED_BC_Y:.2f}, {SEED_BC_Z:.2f}) px, '
      f'Lsd={SEED_LSD_UM / 1000.0:.2f} mm')
"""),
    ("py", """\
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.seed.calibrant import resolve_calibrant

cal = resolve_calibrant(CALIBRANT)

def max_ring_rad_px(bc_y, bc_z, margin=20.0):
    corners = [(0, 0), (NY, 0), (0, NZ), (NY, NZ)]
    return max(np.hypot(cy - bc_y, cz - bc_z) for cy, cz in corners) + margin

TZ_ROUGH_GUESS = 15.0   # rough guess; the beam centre is off-detector and
                        # the true tilt (~14 deg) is large, so a tilt-blind
                        # seed has no way to measure it -- see fit below.

v1_seed = V1Params(
    NrPixelsY=NY, NrPixelsZ=NZ, pxY=PX_UM, pxZ=PX_UM,
    Lsd=SEED_LSD_UM, BC_y=SEED_BC_Y, BC_z=SEED_BC_Z,
    tx=0.0, ty=0.0, tz=TZ_ROUGH_GUESS,
    Wavelength=WAVELENGTH_A, SpaceGroup=cal['sg'],
    LatticeConstant=(cal['a'], cal['b'], cal['c'], cal['alpha'], cal['beta'], cal['gamma']),
    MaxRingRad=max_ring_rad_px(SEED_BC_Y, SEED_BC_Z),
    Refine={'Lsd': True, 'BC': True, 'ty': True, 'tz': True,
            'Wavelength': False, 'Parallax': False,
            **{f'p{i}': False for i in range(15)}},
)
v1_seed.validate()
print('seed geometry built OK')
"""),
    ("md", """\
## Frozen-point fit from a rough tilt guess

One-shot: pick peaks once (`forward/point_pick.py`), freeze them, single
bounded LM fit. `tx` is force-frozen (unobservable from ring radii alone).
All 15 distortion coefficients start frozen too (geometry-only fit).
"""),
    ("py", """\
import time

from midas_calibrate_v2.pipelines.frozen_point import autocalibrate_frozen_point
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_params
from midas_calibrate_v2.forward.distortion import P_COEF_NAMES

POINT_PICK_KWARGS = dict(downsample=4, footprint_px=7, snr_threshold=5.0,
                          min_ring_gap_deg=0.3, subpixel_half_px=2)

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
    point_pick_kwargs=POINT_PICK_KWARGS, lm_max_iter=150, verbose=False,
)
elapsed_frozen = time.time() - t0
fit_frozen = res_frozen.history[0]
print(f'elapsed: {elapsed_frozen:.1f} s')
print(f'Lsd={fit_frozen.Lsd:.1f} um  BC=({fit_frozen.BC_y:.2f},{fit_frozen.BC_z:.2f}) px  '
      f'ty={fit_frozen.ty:.3f} deg  tz={fit_frozen.tz:.3f} deg')
print(f'mean pseudo-strain: {fit_frozen.mean_strain_uE:.1f} microstrain')
"""),
    ("md", """\
## Visual check

Ring overlay via this codebase's standard recipe: write the fit to a v1
paramstest, read back as a `midas_integrate_v2` `IntegrationSpec`, evaluate
its per-pixel forward model (`eval_pixel_REta` — tilt + full 15-term
distortion), then draw true iso-contours (`contourpy`) at each ring's ideal
radius. Interactive `plotly` figure — drag to zoom, double-click to reset.
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
    distortion included. Returns None-separated arrays for one
    `go.Scattergl(mode='lines')` trace, plus the fitted beam centre.
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
# Thin, semi-transparent lines so the raw-intensity rings stay visible
# underneath; Scattergl (not Scatter) for smooth zoom/pan at this point count.
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
Rings track the visible arcs well at whole-detector scale — a systematic
few-pixel offset generally is *not* visible by eye here, which is why
pseudo-strain (not overlay-by-eye) is the quantitative check. Zoom in above
for a sub-pixel look at a specific ring crossing.
"""),
    ("md", """\
## Escaping a blind starting guess

`autocalibrate_frozen_point` above still needed a rough tilt guess (15°).
`iterate_frozen_point_until_stable` removes even that: it re-seeds the
one-shot fit from its own result, each time re-centering the bounds on the
new estimate, until a trailing window of 5 iterations agrees to within
0.01° (tz/ty), 0.05 px (BC), 5 µm (Lsd) — a **strict** criterion,
deliberately with no strain gate.

Why strict: a loose tolerance was found to falsely accept a mid-trajectory
plateau (~8 iterations, tz creeping 0.1–0.6°/step) that is still ~10–12°
from the true answer. The tight tolerances have ~100x margin against that.

Below: from a completely blind `tz=0`.
"""),
    ("py", """\
import dataclasses as _dc

from midas_calibrate_v2.pipelines.frozen_point import iterate_frozen_point_until_stable

# autocalibrate_frozen_point mutates v1_seed in place, so it now holds the
# REFINED geometry from the fit above -- reset BC/Lsd back to the tilt-blind
# seed too, not just tz/ty, for a genuinely blind test.
v1_blind = _dc.replace(
    v1_seed, BC_y=SEED_BC_Y, BC_z=SEED_BC_Z, Lsd=SEED_LSD_UM, ty=0.0, tz=0.0,
)

t0 = time.time()
out_blind = iterate_frozen_point_until_stable(
    v1_blind, image, verbose=True, point_pick_kwargs=POINT_PICK_KWARGS,
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
From `tz=0`, this lands on the identical geometry as the informed-seed fit.

Characterized more broadly offline (13 starting `tz` guesses; see
`best_solution/README.md` for the full sweep):

| starting tz | converged? | iterations | final tz |
|---|---|---|---|
| -15°, -10° | No | 35 (cap) | -16.33°, -7.16° |
| -5° to 30° | **Yes** | 5–33 | 13.9897° (bit-identical every time) |
| 40° | No | 35 (cap) | 43.53° |

Zero false-basin acceptances across a ~35° capture range — wider than every
other method tried on this dataset (alternating pipeline ~5–6°; a single
frozen-point shot without iteration ~19°). Validated on this one dataset
so far.
"""),
    ("md", """\
## Distortion basis

One fixed 15-coefficient basis (`iso_R2/R4/R6` + 6 harmonic pairs
`a_k`/`phi_k`); "choosing a model" = freezing/thawing named coefficients on
`CalibrationSpec` (same convention as notebook **09**). Compare
geometry-only vs. isotropic-only vs. full harmonic, refit from the
established geometry:
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
        point_pick_kwargs=POINT_PICK_KWARGS, lm_max_iter=150, verbose=False,
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
Strain barely moves across bases here — this ring set (14-15 well-isolated,
low-2θ reflections) doesn't carry much distortion information. For a
well-constrained distortion basis, seed `autocalibrate_pv`'s full ring
coverage from this result instead — the two pipelines are complementary.
"""),
    ("md", """\
## Summary

Same starting geometry and point-cloud settings for all three; any
difference below is attributable to the distortion basis alone.
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
**When to use `autocalibrate_frozen_point`**: large/roughly-known tilt,
off-detector beam centre, or `autocalibrate_pv` converging "cleanly" to an
implausible strain. **When to add `iterate_frozen_point_until_stable`**: no
tilt guess at all. Once a good basin is established, `autocalibrate_pv`
(or notebook **09**'s distortion tools) is still the right choice for a
full, distortion-inclusive final refinement.
"""),
]

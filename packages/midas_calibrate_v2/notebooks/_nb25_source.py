"""Cell source for notebook 25 — kept in its own file so _build.py stays
readable; imported by _build.py and registered in NOTEBOOKS.

Notebook 25: two calibrants on one exposure (CeO2 + LaB6).  Self-contained
synthetic ground truth, no external data needed.
"""
from __future__ import annotations

from typing import List, Tuple

Cell = Tuple[str, str]

NB_25: List[Cell] = [
    ("md", """\
# 25 — Two Calibrants on One Exposure (CeO2 + LaB6)

**The situation.** Someone loaded a mixed calibrant — two powders in the beam
at once, or two capillaries stuck together. You now have one frame with two
interleaved ring sets.

**What you get out of it, and what you don't.**

| | |
|---|---|
| ✅ More rings, denser radial sampling | roughly √N tighter σ on Lsd / BC / tilts |
| ✅ A genuine cross-check | the two calibrants must agree; if they don't, your formal error bar is too small |
| ✅ Per-phase sample position | one powder sitting further along the beam shows up as its own Lsd |
| ❌ Wavelength identifiability | both phases enter only through their d-spacings, so λ↔Lsd stays degenerate |
| ❌ Azimuthal harmonics | both phases illuminate the *same* arc of the detector |

The last two are worth internalising before you spend beamtime on this: a
second calibrant adds **rows to the Jacobian, not a new direction**. If λ is
what you need, use a scan of known detector travel (notebook 24). If the
azimuthal distortion is what you need, you need more azimuth.
"""),

    ("py", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from midas_integrate.geometry import build_tilt_matrix, pixel_to_REta
from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table, drop_blended_rings
from midas_calibrate_v2.seed.calibrant import phases_from_calibrants

TRUTH = dict(Lsd=1_000_000.0, BC_y=512.0, BC_z=512.0, ty=0.40, tz=0.25)
NPIX, PX, LAMBDA = 1024, 200.0, 0.173


def make_params(phases=None, **kw):
    p = CalibrationParams()
    p.NrPixelsY = p.NrPixelsZ = NPIX
    p.pxY = p.pxZ = PX
    p.Lsd = TRUTH["Lsd"]; p.BC_y = TRUTH["BC_y"]; p.BC_z = TRUTH["BC_z"]
    p.tx = 0.0; p.ty = TRUTH["ty"]; p.tz = TRUTH["tz"]
    p.Wavelength = LAMBDA
    p.SpaceGroup = 225
    p.LatticeConstant = (5.411, 5.411, 5.411, 90.0, 90.0, 90.0)
    p.MaxRingRad = 480.0; p.MinRingRad = 0.0
    p.RhoD = 480.0 * PX          # µm, matched to the OUTER ring (see below)
    p.Width = 1500.0; p.EtaBinSize = 10.0; p.RBinSize = 1.0
    p.SNRMin = 1.5
    p.tolLsd = 5000.0; p.tolBC = 8.0; p.tolTilts = 1.0
    p.Refine = {"Lsd": True, "BC": True, "ty": True, "tz": True,
                "Wavelength": False, "Parallax": False,
                **{f"p{i}": False for i in range(15)}}
    if phases:
        p.Phases = list(phases)
    for k, v in kw.items():
        setattr(p, k, v)
    return p
"""),

    ("md", """\
## 1. Declare both calibrants

`phases_from_calibrants` turns names (or custom lattice dicts) into the
`Phases` list that `build_ring_table` reads. With `Phases` set, the ring table
is the union of both phases, sorted by radius, with a `phase_idx` column.

In a parameter file the same thing is two `Phase` lines:

```
Phase CeO2 225 5.41153 5.41153 5.41153 90 90 90
Phase LaB6 221 4.15689 4.15689 4.15689 90 90 90
MinRingSeparation 12.0
```
"""),

    ("py", """\
p_both = make_params(phases_from_calibrants(["CeO2", "LaB6"]))
rt = build_ring_table(p_both)

print(f"phases: {rt.phase_names}")
for name in rt.phase_names:
    print(f"  {name}: {int(rt.phase_mask(name).sum())} rings")
print(f"total: {len(rt)} rings, sorted by radius")

n_merged = sum(1 for a in rt.hkl_aliases if a)
print(f"\\nrows that absorbed an exact d-spacing degeneracy: {n_merged}")
print("  (e.g. LaB6 (300)/(221) — 9 = 9, so ONE physical ring with two hkl")
print("   labels. Left unmerged these look like zero-separation doublets")
print("   and a blend rule throws away perfectly good rings.)")
"""),

    ("md", """\
## 2. Exclude the blends

Two interleaved ring sets always collide somewhere. A blended ring's centroid
is dragged by its neighbour, and nothing downstream knows it happened — the
fit "succeeds" with a quietly biased geometry.

`drop_blended_rings` flags the colliding rings *individually*. That is the
difference from `max_resolvable_ring_radius_px`, which returns a radial cutoff
and therefore discards every ring outside the first collision.
"""),

    ("py", """\
for cut in (0.0, 6.0, 12.0, 25.0):
    kept, n_dropped = drop_blended_rings(rt, min_separation_px=cut)
    per = {n: int(kept.phase_mask(n).sum()) for n in kept.phase_names}
    print(f"cut {cut:5.1f} px -> keep {len(kept):3d} rings "
          f"(dropped {n_dropped:2d})  {per}")

print("\\nOnly cross-phase collisions, leaving same-phase doublets to the")
print("doublet co-fitter (which CAN model a pair):")
kept_x, n_x = drop_blended_rings(rt, min_separation_px=12.0,
                                  cross_phase_only=True)
print(f"  keep {len(kept_x)} rings (dropped {n_x})")
"""),

    ("md", """\
### Blends of three or more

A chain of three rings inside the window is **not** two doublets. The 2-peak
co-fitter has an analytic Jacobian written for exactly two centres, so
chaining pairs fits the interior ring twice at two different centres.
`cluster_rings` separates the cases so those rings get dropped instead.
"""),

    ("py", """\
from midas_calibrate_v2.forward.doublets import cluster_rings, doublet_index_map

demo = np.array([100.0, 110.0, 120.0, 300.0, 500.0, 508.0])
c = cluster_rings(demo, min_separation_px=25.0)
print(f"pairs      (co-fittable): {[(g.i, g.j) for g in c.pairs]}")
print(f"n_ary  (>=3, must drop): {c.n_ary}")
print(f"singletons             : {c.singletons}")

partner, _ = doublet_index_map(demo, min_separation_px=25.0)
print(f"\\npartner map: {partner.tolist()}")
print("  -1 = fit alone, -2 = inside a >=3 blend, >=0 = doublet partner")
"""),

    ("md", """\
## 3. Calibrate, and read the per-phase residual

Pass a **list** to `calibrate()`. Seeding still uses the first entry (the
seeder matches an arc pattern against one ring table), so list the stronger,
smoother powder first.

The number to look at is not the pooled strain — it is the **per-phase**
breakdown. A geometry fitted on one powder can be systematically wrong for the
other while looking fine on its own rings.
"""),

    ("py", """\
def simulate(params, sigma_px=1.5, seed=0):
    rt_ = build_ring_table(params)
    px = 0.5 * (params.pxY + params.pxZ)
    TRs = build_tilt_matrix(params.tx, params.ty, params.tz)
    Y, Z = np.meshgrid(np.arange(params.NrPixelsY, dtype=float),
                       np.arange(params.NrPixelsZ, dtype=float))
    R, _ = pixel_to_REta(Y, Z, Ycen=params.BC_y, Zcen=params.BC_z, TRs=TRs,
                         Lsd=params.Lsd, RhoD=params.RhoD, px=px,
                         parallax=params.Parallax)
    rng = np.random.default_rng(seed)
    img = np.full(R.shape, 50.0) + rng.normal(0, 5.0, size=R.shape)
    for r in rt_.r_ideal_px:
        img += (1000.0 / (1.0 + r / 100.0)) * np.exp(-0.5 * ((R - r) / sigma_px) ** 2)
    return img


image = simulate(p_both)
print(f"synthetic two-phase frame: {image.shape}, max {image.max():.0f}")
"""),

    ("py", """\
from midas_calibrate_v2.pipelines.single import autocalibrate
from midas_calibrate_v2.loss.diagnostics import per_phase_summary
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

seed = make_params(phases_from_calibrants(["CeO2", "LaB6"]),
                   MinRingSeparation=12.0)
seed.Lsd += 400.0; seed.BC_y += 1.5; seed.BC_z -= 1.0

res = autocalibrate(seed, image, n_iter=3, verbose=False,
                    build_residual_corr=False)
u = res.unpacked
print(f"Lsd  {float(u['Lsd']):10.1f} um   (truth {TRUTH['Lsd']:.1f})")
print(f"BC   ({float(u['BC_y']):.3f}, {float(u['BC_z']):.3f})   "
      f"(truth {TRUTH['BC_y']}, {TRUTH['BC_z']})")
print(f"ty   {float(u['ty']):+.4f}  tz {float(u['tz']):+.4f}   "
      f"(truth {TRUTH['ty']:+.4f}, {TRUTH['tz']:+.4f})")

fd = res.fits_final
with torch.no_grad():
    r = pseudo_strain_residual(fd.Y_pix, fd.Z_pix, fd.ring_two_theta_deg, u,
                               rho_d=fd.rho_d, weights=None).abs() * 1e6
print()
print(per_phase_summary(r, fd.phase_idx, fd.phase_names))
"""),

    ("md", """\
**Read the ⚠ line, not the mean.** If one phase's residual is well above the
other's, the calibrants disagree, and the honest uncertainty on your geometry
is the *spread between them*, not the fit's formal σ. Two usual suspects:

1. **The assumed lattice constant.** `da/a` is exactly degenerate with
   `dLsd/Lsd` — an `a` wrong by 1e-4 hands you an Lsd wrong by 100 ppm and
   nothing in the residual complains. Check which certificate your value
   traces to; `lattice_uncertainty_lsd_ppm` converts the certified σ into the
   Lsd error it implies.
2. **The powders are not in the same place.** See §5.
"""),

    ("py", """\
from midas_calibrate_v2.seed.calibrant import (CALIBRANTS,
                                               lattice_uncertainty_lsd_ppm)

print("registered calibrants and their traceability:")
for name, spec in CALIBRANTS.items():
    sig = spec.get("a_sigma")
    tail = (f"sigma_a = {sig:g} A -> "
            f"{lattice_uncertainty_lsd_ppm(spec['a'], sig):.1f} ppm on Lsd"
            if sig else "sigma_a not set — fill from your certificate")
    print(f"  {name:6s} a = {spec['a']:.5f} A  [{spec.get('srm', '?')}]  {tail}")
"""),

    ("md", """\
## 4. Check that the fit is answering a question the data can answer

Two gates catch the failures that a strain number alone will not.

**Azimuthal coverage.** An offset or off-axis detector sees only a wedge of
every ring. Over a narrow wedge the azimuthal harmonics `a1..a6` stop being
separable from the beam centre (1-fold) and the tilts (2-fold). The fit then
rails them and the E↔M loop stops converging. A second calibrant does **not**
help — both powders share the wedge.

**RhoD scaling.** The distortion polynomial lives in `rho = R_um / RhoD`. Set
RhoD far beyond the outermost ring and `rho` stays small, so `rho^4` and
`rho^6` collapse and the high-order radial terms become unmeasurable.
"""),

    ("py", """\
from midas_calibrate_v2.pipelines.diagnostics import (azimuth_coverage_gate,
                                                       rho_d_scaling_gate)

for g in (azimuth_coverage_gate(fd, u, spec=res.spec),
          rho_d_scaling_gate(fd, u, spec=res.spec)):
    icon = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[g.severity]
    print(f"[{icon}] {g.name}: {g.message}")
"""),

    ("py", """\
# What the same gate says about a detector that sees only a 70-degree wedge
# while refining the harmonics — the real 1-ID ge1 case.
from midas_calibrate_v2.pipelines._common import FittedDataset

eta = np.deg2rad(np.linspace(-35, 35, 200))
Yw = 512.0 - 300.0 * np.cos(eta)
Zw = 512.0 + 300.0 * np.sin(eta)
t = lambda v: torch.as_tensor(v, dtype=torch.float64)
wedge = FittedDataset(
    Y_pix=t(Yw), Z_pix=t(Zw), ring_idx=torch.zeros(200, dtype=torch.long),
    snr=torch.full((200,), 10.0, dtype=torch.float64),
    ring_two_theta_deg=torch.full((200,), 3.0, dtype=torch.float64),
    rho_d=t(96000.0), weights=torch.ones(200, dtype=torch.float64))
u_h = {k: t(float(v)) for k, v in dict(
    Lsd=1e6, BC_y=512.0, BC_z=512.0, tx=0.0, ty=0.0, tz=0.0,
    pxY=200.0, pxZ=200.0, a1=1e-3, a2=1e-3, a3=1e-3).items()}
g = azimuth_coverage_gate(wedge, u_h)
print(f"[{g.severity.upper()}] {g.message}")
"""),

    ("md", """\
## 5. Two capillaries: per-phase sample position

If the powders are not co-located, each sees its own geometry:

```
Lsd_phase  = Lsd0  + dx          (displaced along the beam)
BC_y_phase = BC_y0 + dy / px     (displaced transversely — the cone apex moves)
```

Only the *relative* offset is identifiable; phase A's own offset is absorbed
into the global Lsd/BC. `autocalibrate_multi` models this directly: pass the
same frame twice, one calibrant each, and use **`mode="same_detector"`**.

That mode matters. The default leaves the tilts per-image, which is wrong when
one detector took both exposures — and it biases what is left. On a real
CeO2+LaB6 frame, independently-refined tilts absorbed the difference between
the calibrants and reported a **1.43 mm** relative offset where sharing the
tilts gives **72 ± 34 µm**.
"""),

    ("py", """\
from midas_calibrate_v2.pipelines.multi import build_multi_spec

v1_ce = make_params(phases_from_calibrants(["CeO2"]))
v1_la = make_params(phases_from_calibrants(["LaB6"]))
v1_la.SpaceGroup = 221
v1_la.LatticeConstant = (4.15689, 4.15689, 4.15689, 90.0, 90.0, 90.0)

ms = build_multi_spec([v1_ce, v1_la], mode="same_detector")
print("shared (detector properties):")
print("   ", sorted(n for n in ms.shared if not n.startswith("panel_")))
print("per-phase (that powder's sample position):")
print("   ", sorted(ms.per_image[0]))
"""),

    ("md", """\
### The catch you cannot fit your way out of

`dLsd/Lsd` for phase B is **exactly degenerate** with a relative error in phase
B's lattice constant. One frame cannot tell *"the LaB6 capillary sits 72 µm
closer"* from *"the LaB6 lattice constant is 2.6e-5 too small"*.

Breaking it needs a second detector distance: a physical offset stays fixed in
mm while a lattice error scales with Lsd. That is
`autocalibrate_multi(..., lsd_offsets_um=[...])` — notebook 24.
"""),

    ("py", """\
dLsd_um, Lsd_um, a_LaB6 = 71.8, 2_731_920.0, 4.15689
rel = dLsd_um / Lsd_um
print(f"a per-phase offset of {dLsd_um:.1f} um is {rel:.2e} relative,")
print(f"indistinguishable from da = {rel * a_LaB6:+.5f} A "
      f"(a = {a_LaB6:.5f} -> {a_LaB6 * (1 + rel):.5f})")
"""),

    ("md", """\
## Summary

1. Pass a **list** of calibrants; list the smoother powder first (it seeds).
2. Set **`MinRingSeparation`** (12 px is a reasonable start) — interleaved ring
   sets always collide, and blended centroids bias the geometry silently.
3. Read the **per-phase residual**, not the pooled mean. Disagreement between
   calibrants is your real error bar.
4. Check the **azimuth** and **RhoD** gates before believing any refined
   distortion coefficient, and check `result.unconstrained` / `result.at_bounds`.
5. For per-phase sample position use `mode="same_detector"`, and remember it is
   degenerate with the lattice constant unless you move the detector.
"""),
]

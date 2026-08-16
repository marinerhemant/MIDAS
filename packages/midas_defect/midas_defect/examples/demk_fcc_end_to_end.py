#!/usr/bin/env python3
"""End-to-end diffuse-defect inventory for the demk FCC dataset.

The flagship `midas_defect` example: from a sparse q-space voxel cloud + the
indexed grains, reproduce the full defect inventory —

    geometry QC → Bragg/diffuse split → 100 % intensity budget
                → forbidden-reflection test → ⟨111⟩ rod enrichment → fault-α

**Phase-agnostic**: swap ``crystal`` (and the distortion/geometry block) to point
at any other dataset — e.g. a genuine CuAl₂ superstructure sample via
``lattice.cual2_crystal()``.

Reuses `midas_transforms.apply_tilt_distortion` for the validated, distortion-
aware pixel→lab map (the same path that produced the published numbers). The
sample-frame rotation uses the validated ω map ω = 180 − 0.25·frame (NOT the old
sign stored in some NPZs).

Usage
-----
    python -m midas_defect.examples.demk_fcc_end_to_end \
        --voxels /path/voxels_layerXXXX.npz --grains /path/Grains.csv
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch

from midas_transforms.fit_setup.transform import apply_tilt_distortion

from midas_defect.lattice import fcc_cu_crystal
from midas_defect.bragg_diffuse import (
    predicted_reflection_points, classify_voxels, on_lattice_fraction,
)
from midas_defect.intensity_budget import intensity_budget
from midas_defect.defect_tests import (
    forbidden_reflection_test, rod_family_enrichment, fault_probability_alpha,
    fault_rod_alignment,
)
from midas_defect.williamson_hall import dislocation_density_per_grain

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-defect"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



@dataclass
class DemkGeometry:
    """Validated gold-calibrant detector model for the demk Sep-2025 beamtime."""
    lsd_um: float = 652665.632541
    bcy_px: float = 698.420228
    bcz_px: float = 813.680035
    tx_deg: float = 0.095682
    ty_deg: float = -0.196484
    tz_deg: float = 0.534276
    px_um: float = 172.0
    rho_d_um: float = 219964.42
    wavelength_A: float = 0.172979
    # 15-vector lens-distortion coeffs (p0..p3 meaningful, rest 0)
    p_coeffs: tuple = (0.000230, 0.001234, 0.000211, 32.904494) + (0.0,) * 11
    omega_start_deg: float = 180.0     # validated ω map: ω = start + step·frame
    omega_step_deg: float = -0.25


def voxels_to_qsample(indices: np.ndarray, det_bin: int, geom: DemkGeometry) -> np.ndarray:
    """(frame,row,col) detector voxels → sample-frame q (1/Å), validated path."""
    frame = indices[:, 0].astype(np.float64)
    # bin-centre unbinning (col = horizontal = Y_pix, row = vertical = Z_pix)
    y_pix = indices[:, 2].astype(np.float64) * det_bin + (det_bin - 1) / 2.0
    z_pix = indices[:, 1].astype(np.float64) * det_bin + (det_bin - 1) / 2.0
    Yl, Zl = apply_tilt_distortion(
        torch.tensor(y_pix), torch.tensor(z_pix),
        Lsd=torch.tensor(geom.lsd_um), BC_y=torch.tensor(geom.bcy_px),
        BC_z=torch.tensor(geom.bcz_px), tx=torch.tensor(geom.tx_deg),
        ty=torch.tensor(geom.ty_deg), tz=torch.tensor(geom.tz_deg),
        p_coeffs=torch.tensor(geom.p_coeffs, dtype=torch.float64),
        px=torch.tensor(geom.px_um), rho_d=torch.tensor(geom.rho_d_um),
    )
    Yl = Yl.numpy(); Zl = Zl.numpy()
    k = 2.0 * math.pi / geom.wavelength_A
    L = np.sqrt(geom.lsd_um ** 2 + Yl * Yl + Zl * Zl)
    g1 = -1.0 + geom.lsd_um / L
    g2 = Yl / L
    zn = Zl / L
    om = np.radians(-(geom.omega_start_deg + geom.omega_step_deg * frame))
    co, so = np.cos(om), np.sin(om)
    return np.stack([k * (g1 * co - g2 * so), k * (g1 * so + g2 * co), k * zn], axis=1)


def inventory_from_qsample(qs: np.ndarray, val: np.ndarray, OM: np.ndarray,
                           *, crystal=None, q_max_inv_A: float = 8.0,
                           verbose: bool = True) -> dict:
    """Run the full inventory on already-converted sample-frame q + intensity.

    Shared core for both the binned-NPZ example and the full-res-zarr
    validation, so neither duplicates the analysis logic.

    The default ``q_max_inv_A = 8.0`` matches the canonical demk FCC analysis
    (``comprehensive_uq_par.py``): it includes the FCC {331}/{420} shells
    while excluding {422} at ~8.47 1/A. Pushing past 8.0 shrinks every voxel's
    distance-to-lattice, pulling voxels from ``inter_bragg`` into ``asterism``
    and ``bragg`` -- visible as a ~1-3% shift in the budget fractions and a
    ~1% shift in ``on_lattice_bright_frac``.
    """
    crystal = crystal if crystal is not None else fcc_cu_crystal()
    qmag = np.linalg.norm(qs, axis=1)
    P = predicted_reflection_points(OM, crystal, q_max_inv_A=q_max_inv_A).numpy()

    olf = on_lattice_fraction(qs, val, P, bright_percentile=99.5, tol_inv_A=0.1)
    split = classify_voxels(qs, val, P, tol_inv_A=0.05)
    budget = intensity_budget(split.dist_to_lattice, qmag, val)
    forb = forbidden_reflection_test(qs, val, OM, crystal)
    fault = fault_probability_alpha(qs, val, OM, crystal)
    wh = dislocation_density_per_grain(qs, val, OM, crystal)
    frods = fault_rod_alignment(qs, val, OM, crystal)

    out = {
        "n_voxels": len(val),
        "n_grains": len(OM),
        "on_lattice_bright_frac": olf,
        "budget": budget.fractions,
        "budget_closes": budget.closes(atol=1e-9),
        "forbidden_excess_median": forb.excess_median,
        "fault_alpha_median": fault.alpha_median,
        "rho_median_per_m2": wh.rho_median_per_m2,
        "domain_size_A_median": wh.domain_size_A_median,
        "fault_rod_along_perp_median": frods.along_over_perp_median,
        "fault_rod_frac_enriched": frods.frac_grains_enriched,
    }
    if verbose:
        print(f"voxels {out['n_voxels']:,} | grains {out['n_grains']}")
        print(f"geometry QC: {100*olf:.1f}% of bright voxels on lattice")
        print(budget)
        print(f"dislocation density (WH, b={wh.burgers_A:.3f} Å): median "
              f"{wh.rho_median_per_m2:.2e} m⁻²  (D≈{wh.domain_size_A_median:.0f} Å, "
              f"{wh.n_grains_fit}/{wh.n_grains} grains fit)")
        print(f"forbidden excess (median): {forb.excess_median:+.4f}  "
              f"({forb.n_grains_excess}/{forb.n_grains} grains with excess)")
        print(f"⟨111⟩ fault rods (explicit along/perp): median "
              f"{frods.along_over_perp_median:.2f}, "
              f"{100*frods.frac_grains_enriched:.0f}% of grains > 1.2×")
        print(f"fault α (median): {fault.alpha_median:.4f}  "
              f"(faulted third {fault.faulted_third_median:.4f})")
    return out


def run_inventory(voxels_npz: str, grains_csv: str, *, crystal=None, geom=None,
                  q_max_inv_A: float = 8.0, verbose: bool = True) -> dict:
    """Full diffuse-defect inventory from a sparse voxel NPZ (indices/values)."""
    geom = geom if geom is not None else DemkGeometry()
    d = np.load(voxels_npz, allow_pickle=False)
    det_bin = int(d["det_bin"])
    qs = voxels_to_qsample(d["indices"], det_bin, geom)
    val = d["values"].astype(np.float64)
    OM = np.genfromtxt(grains_csv, comments="%")[:, 1:10].reshape(-1, 3, 3)
    return inventory_from_qsample(qs, val, OM, crystal=crystal,
                                  q_max_inv_A=q_max_inv_A, verbose=verbose)


def main():
    ap = _midas_make_parser(description=__doc__)
    ap.add_argument("--voxels", required=True, help="sparse voxel NPZ (indices/values)")
    ap.add_argument("--grains", required=True, help="MIDAS Grains.csv")
    ap.add_argument("--q-max", type=float, default=8.5)
    args = ap.parse_args()
    run_inventory(args.voxels, args.grains, q_max_inv_A=args.q_max)


if __name__ == "__main__":
    main()

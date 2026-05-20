"""Stage: calc_radius_v — V-map foundation (P8 of the V-map plan).

This stage computes **theoretical per-ring intensities** and
**per-spot relative volumes** via
:mod:`midas_transforms.radius.theoretical`.  Outputs feed the
``refine_vmap`` stage which fits per-voxel V and per-ring K jointly.

Outputs (under ``<layer_dir>/Output/``):

* ``Radius_V.csv``  — per-spot table with columns::

      spot_id  scan_nr  ring_number  ring_idx  intensity  V_rel  omega_deg  eta_deg

* ``I_theory_per_ring.csv`` — per-ring theoretical intensity::

      ring_number  two_theta_deg  I_theory

The stage no-ops cleanly if:

* :attr:`PipelineConfig.vmap.run` is ``False`` (default — guards the
  existing pipelines that don't opt into the V-map work).
* mandatory inputs (``hkls.csv``, ``InputAllExtraInfoFittingAll*.csv``,
  ``vmap.crystal_cif`` path, wavelength) are missing — falls through
  to the stub with a clear log message.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .._logging import LOG
from ..results import CalcRadiusVResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def _resolve_paramstest(layer_dir: Path) -> Path | None:
    for c in (layer_dir / "paramstest.txt", layer_dir / "Output" / "paramstest.txt"):
        if c.exists():
            return c
    return None


def _wavelength_from_paramstest(paramstest: Path) -> float | None:
    if not paramstest.exists():
        return None
    for line in paramstest.read_text().splitlines():
        s = line.strip()
        if s.lower().startswith("wavelength"):
            try:
                # MIDAS lines often have a trailing semicolon: "Wavelength 0.173;"
                tok = s.split()[1].rstrip(";").strip()
                return float(tok)
            except (IndexError, ValueError):
                continue
    return None


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.config.vmap
    if not cfg.run:
        return stub_run("calc_radius_v", ctx)

    layer_dir = Path(ctx.layer_dir)
    out_dir = layer_dir / "Output"

    hkls_csv = layer_dir / "hkls.csv"
    if not hkls_csv.exists():
        alt = out_dir / "hkls.csv"
        hkls_csv = alt if alt.exists() else hkls_csv
    if not hkls_csv.exists():
        LOG.warning("calc_radius_v: hkls.csv not found under %s; skipping.", layer_dir)
        return stub_run("calc_radius_v", ctx)

    spot_csvs = sorted(layer_dir.glob("InputAllExtraInfoFittingAll*.csv"))
    if not spot_csvs:
        LOG.warning(
            "calc_radius_v: no InputAllExtraInfoFittingAll*.csv under %s; skipping.",
            layer_dir,
        )
        return stub_run("calc_radius_v", ctx)

    if not cfg.crystal_cif:
        LOG.warning(
            "calc_radius_v: vmap.crystal_cif is empty; cannot compute "
            "theoretical I_ring — skipping."
        )
        return stub_run("calc_radius_v", ctx)

    crystal_cif = Path(cfg.crystal_cif)
    if not crystal_cif.exists():
        LOG.warning("calc_radius_v: vmap.crystal_cif %s does not exist; skipping.",
                    crystal_cif)
        return stub_run("calc_radius_v", ctx)

    wavelength_A = cfg.wavelength_A
    if wavelength_A <= 0.0:
        params = _resolve_paramstest(layer_dir)
        if params is not None:
            wavelength_A = _wavelength_from_paramstest(params) or 0.0
    if wavelength_A <= 0.0:
        LOG.warning(
            "calc_radius_v: wavelength unknown (vmap.wavelength_A=0 and "
            "paramstest absent or missing). Skipping."
        )
        return stub_run("calc_radius_v", ctx)

    started = time.time()
    import torch
    from midas_hkls import read_cif
    from midas_transforms.radius import (
        load_rings_from_hkls_csv,
        load_spots_from_input_extra_info_csvs,
        per_spot_relative_volume,
        theoretical_intensity_per_ring,
    )

    dtype = torch.float64

    ring_table = load_rings_from_hkls_csv(hkls_csv, dtype=dtype)
    spots = load_spots_from_input_extra_info_csvs(
        layer_dir, ring_table=ring_table, dtype=dtype,
    )
    crystal = read_cif(crystal_cif)
    crystal_t = crystal.to_torch(dtype=dtype)

    two_theta_max = (
        cfg.two_theta_max_deg
        if cfg.two_theta_max_deg > 0.0
        else float(ring_table.two_theta_deg.max().item()) + 1.0
    )
    I_theory = theoretical_intensity_per_ring(
        crystal_t,
        torch.tensor(wavelength_A, dtype=dtype),
        ring_table,
        two_theta_max_deg=two_theta_max,
        two_theta_tol_deg=cfg.two_theta_tol_deg,
        polarization=cfg.polarization,
    )

    V_rel = per_spot_relative_volume(
        spots.ring_idx, spots.intensity, I_theory,
    )
    n_valid = int((V_rel > 0).sum().item())

    out_dir.mkdir(parents=True, exist_ok=True)
    radius_csv = out_dir / "Radius_V.csv"
    np.savetxt(
        radius_csv,
        np.column_stack([
            spots.spot_id.numpy(), spots.scan_nr.numpy(),
            spots.ring_number.numpy(), spots.ring_idx.numpy(),
            spots.intensity.numpy(), V_rel.numpy(),
            spots.omega_deg.numpy(), spots.eta_deg.numpy(),
        ]),
        header="spot_id scan_nr ring_number ring_idx intensity V_rel omega_deg eta_deg",
        fmt=["%d", "%d", "%d", "%d", "%.6e", "%.6e", "%.6f", "%.6f"],
        comments="",
    )

    theory_csv = out_dir / "I_theory_per_ring.csv"
    np.savetxt(
        theory_csv,
        np.column_stack([
            ring_table.ring_numbers.numpy(),
            ring_table.two_theta_deg.numpy(),
            I_theory.detach().numpy(),
        ]),
        header="ring_number two_theta_deg I_theory",
        fmt=["%d", "%.6f", "%.6e"],
        comments="",
    )

    finished = time.time()
    LOG.info(
        "calc_radius_v: wrote %d-spot Radius_V.csv (%d valid V_rel) and "
        "%d-ring I_theory.csv in %.2fs",
        int(spots.spot_id.numel()), n_valid, int(ring_table.ring_numbers.numel()),
        finished - started,
    )
    return CalcRadiusVResult(
        stage_name="calc_radius_v",
        started_at=started, finished_at=finished, duration_s=finished - started,
        inputs={
            "hkls_csv": str(hkls_csv),
            "crystal_cif": str(crystal_cif),
            "wavelength_A": str(wavelength_A),
        },
        outputs={str(radius_csv): "", str(theory_csv): ""},
        metrics={
            "n_spots": int(spots.spot_id.numel()),
            "n_valid": n_valid,
            "n_rings": int(ring_table.ring_numbers.numel()),
            "wavelength_A": float(wavelength_A),
        },
        radius_csv=str(radius_csv),
        theory_csv=str(theory_csv),
        n_spots=int(spots.spot_id.numel()),
        n_rings=int(ring_table.ring_numbers.numel()),
    )

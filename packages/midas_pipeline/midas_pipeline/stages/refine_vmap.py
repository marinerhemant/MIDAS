"""Stage: refine_vmap — joint V (+ K, +μ, +beam) refinement (P8).

Reads ``Radius_V.csv`` and ``I_theory_per_ring.csv`` from the upstream
``calc_radius`` stage, plus per-voxel grain assignment + lab-frame
positions from the layer's existing artifacts.  Runs
:func:`midas_transforms.radius.refine_vmap_joint` and writes:

* ``Output/v_map.h5``                 — per-voxel V (+ grain_map + voxel_pos)
* ``Output/k_per_ring.csv``           — per-ring scale K
* ``Output/vmap_loss_history.csv``    — LBFGS / Adam loss trace

The stage is a clean skip when ``vmap.run`` is False or the upstream
inputs are absent.

Voxel-grid resolution: a synthetic-friendly layout is supported via
``Output/voxel_grid.csv`` (columns: ``voxel_idx, x_um, y_um, z_um,
grain_id``).  When that file is absent, the stage tries to derive a
single-grain compact layout from ``Radius_V.csv`` by treating every spot
as belonging to a notional grain 0 at the origin — useful for early
integration tests; production PF data ships ``voxel_grid.csv`` from
``find_grains``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .._logging import LOG
from ..results import RefineVmapResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def _try_load_voxel_grid(out_dir: Path):
    """Return (voxel_pos_um (N,3), grain_map (N,) int) or (None, None)."""
    p = out_dir / "voxel_grid.csv"
    if not p.exists():
        return None, None
    arr = np.loadtxt(p, comments="#", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    xyz = arr[:, 1:4].astype(np.float64)
    g = arr[:, 4].astype(np.int64)
    return xyz, g


def _try_load_scan_positions(layer_dir: Path):
    """Return (scan_pos_um (n_scans,) float, scan_to_spatial (n_scans,) int) or
    (None, None).  Reads MIDAS ``positions.csv`` (one Y per line)."""
    p = layer_dir / "positions.csv"
    if not p.exists():
        return None, None
    y = np.loadtxt(p, dtype=np.float64).flatten()
    sts = np.argsort(y, kind="stable")
    return y, sts


def _wavelength_from_paramstest(layer_dir: Path) -> float | None:
    for p in (layer_dir / "paramstest.txt", layer_dir / "Output" / "paramstest.txt"):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            s = line.strip()
            if s.lower().startswith("wavelength"):
                try:
                    return float(s.split()[1])
                except (IndexError, ValueError):
                    continue
    return None


def _beam_size_from_config_or_paramstest(ctx: StageContext) -> float:
    beam = ctx.config.scan.beam_size_um
    if beam > 0:
        return float(beam)
    for p in (
        Path(ctx.layer_dir) / "paramstest.txt",
        Path(ctx.layer_dir) / "Output" / "paramstest.txt",
    ):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            s = line.strip()
            if s.lower().startswith("beamsize"):
                try:
                    return float(s.split()[1])
                except (IndexError, ValueError):
                    continue
    return 0.0


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.config.vmap
    if not cfg.run:
        return stub_run("refine_vmap", ctx)

    layer_dir = Path(ctx.layer_dir)
    out_dir = layer_dir / "Output"

    radius_csv = out_dir / "Radius_V.csv"
    theory_csv = out_dir / "I_theory_per_ring.csv"
    if not radius_csv.exists() or not theory_csv.exists():
        LOG.warning(
            "refine_vmap: missing Radius_V.csv or I_theory_per_ring.csv "
            "in %s — run calc_radius first.", out_dir,
        )
        return stub_run("refine_vmap", ctx)

    voxel_pos_np, grain_map_np = _try_load_voxel_grid(out_dir)
    if voxel_pos_np is None:
        LOG.warning(
            "refine_vmap: %s/voxel_grid.csv not present; the refine_vmap "
            "stage requires a (voxel_idx, x, y, z, grain_id) table. Skipping.",
            out_dir,
        )
        return stub_run("refine_vmap", ctx)

    scan_pos_np, _ = _try_load_scan_positions(layer_dir)

    beam_size_um = _beam_size_from_config_or_paramstest(ctx)
    if beam_size_um <= 0:
        LOG.warning(
            "refine_vmap: beam_size_um is 0 (no scan.beam_size_um and no "
            "'BeamSize' in paramstest). Skipping."
        )
        return stub_run("refine_vmap", ctx)

    started = time.time()
    import torch

    from midas_transforms.geometry import SampleGrid, TopHat
    from midas_transforms.radius import (
        predicted_spot_intensities,
        refine_K_per_ring_closed_form,
        refine_vmap_joint,
    )

    dtype = torch.float64
    arr = np.loadtxt(radius_csv, comments="#", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    spot_id   = arr[:, 0].astype(np.int64)
    scan_nr   = arr[:, 1].astype(np.int64)
    ring_idx  = arr[:, 3].astype(np.int64)
    intensity = arr[:, 4].astype(np.float64)
    omega_deg = arr[:, 6].astype(np.float64)

    theo = np.loadtxt(theory_csv, comments="#", skiprows=1)
    if theo.ndim == 1:
        theo = theo.reshape(1, -1)
    n_rings = int(theo.shape[0])
    I_theory = torch.as_tensor(theo[:, 2], dtype=dtype)

    spot_grain_idx = np.full_like(spot_id, -1)
    # Spot → grain via scan_nr -> nearest voxel in scan column
    # (PF assumption: each spot's scan column directly attributes to the
    # grain occupying that voxel; refine_vmap_joint already handles the
    # beam-fraction weighting, so this is a per-spot grain assignment, not
    # a per-(spot, voxel) attribution.)
    if scan_pos_np is not None:
        # Map scan_nr -> spatial column → grain at that column via the
        # supplied voxel_grid.  Pick the voxel whose y is closest to
        # scan_pos[scan_nr].
        scan_pos_t = torch.as_tensor(scan_pos_np, dtype=dtype)
        vp_y = torch.as_tensor(voxel_pos_np[:, 1], dtype=dtype)
        for k, s in enumerate(scan_nr):
            if 0 <= s < scan_pos_t.numel():
                # closest voxel by |y - scan_pos|
                d = (vp_y - scan_pos_t[s]).abs()
                spot_grain_idx[k] = int(grain_map_np[int(d.argmin())])
    else:
        # No positions.csv → assume scan_nr directly indexes the voxel.
        for k, s in enumerate(scan_nr):
            if 0 <= s < voxel_pos_np.shape[0]:
                spot_grain_idx[k] = int(grain_map_np[s])

    sg = SampleGrid.from_arrays(
        voxel_positions=voxel_pos_np,
        voxel_size_um=beam_size_um,
        grain_map=grain_map_np,
    )

    spot_scan_pos_um = (
        torch.as_tensor(scan_pos_np[scan_nr], dtype=dtype)
        if scan_pos_np is not None
        else torch.as_tensor(scan_nr * beam_size_um, dtype=dtype)
    )
    spot_ring_t = torch.as_tensor(ring_idx, dtype=torch.int64)
    spot_grain_t = torch.as_tensor(spot_grain_idx, dtype=torch.int64)
    spot_obs_t = torch.as_tensor(intensity, dtype=dtype)
    spot_ome_rad = torch.as_tensor(np.deg2rad(omega_deg), dtype=dtype)

    beam = TopHat(beam_size_um, refine=cfg.refine_beam)

    # Resolve scan_axis: "auto" -> "pf" for PF mode, "none" for FF.
    if cfg.scan_axis == "auto":
        scan_axis = "pf" if ctx.config.is_pf else "none"
    else:
        scan_axis = cfg.scan_axis
    LOG.info("refine_vmap: scan_axis=%s (resolved from cfg.scan_axis=%s, "
             "scan_mode=%s)", scan_axis, cfg.scan_axis, ctx.config.scan.scan_mode)

    V_init = torch.ones(sg.n_voxels, dtype=dtype)
    K_init = refine_K_per_ring_closed_form(
        V_init, I_theory, spot_obs_t,
        spot_ring_t, spot_grain_t, spot_scan_pos_um, spot_ome_rad,
        sg, beam, n_rings=n_rings,
        scan_axis=scan_axis,
    )

    mu_init = None
    if cfg.use_absorption and cfg.element:
        from midas_hkls.absorption import linear_absorption_coefficient
        wavelength_A = (
            cfg.wavelength_A
            if cfg.wavelength_A > 0
            else (_wavelength_from_paramstest(layer_dir) or 0.0)
        )
        if wavelength_A > 0:
            mu_init = torch.as_tensor(
                float(linear_absorption_coefficient(cfg.element, wavelength_A)),
                dtype=dtype,
            )

    result = refine_vmap_joint(
        V_init=V_init, K_init=K_init,
        spot_observed_intensity=spot_obs_t,
        spot_ring_idx=spot_ring_t, spot_grain_idx=spot_grain_t,
        spot_scan_pos_um=spot_scan_pos_um, spot_omega_rad=spot_ome_rad,
        sample_grid=sg, beam_profile=beam,
        theoretical_intensity_per_ring=I_theory,
        scan_axis=scan_axis,
        refine_V=cfg.refine_V, refine_K=cfg.refine_K,
        refine_mu=cfg.refine_mu, refine_beam=cfg.refine_beam,
        max_iter=cfg.max_iter, loss_kind=cfg.loss_kind,
        tolerance=cfg.tolerance,
        use_absorption=cfg.use_absorption and (mu_init is not None),
        mu_init=mu_init,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    v_map_h5 = out_dir / "v_map.h5"
    try:
        import h5py
        with h5py.File(v_map_h5, "w") as f:
            f.create_dataset("voxels/V",            data=result.V_voxel.numpy())
            f.create_dataset("voxels/positions_um", data=voxel_pos_np)
            f.create_dataset("voxels/grain_map",    data=grain_map_np)
            f.create_dataset("rings/K",             data=result.K_ring.numpy())
            f.create_dataset("rings/I_theory",      data=I_theory.numpy())
            if result.mu_per_cm is not None:
                f.create_dataset("absorption/mu_per_cm",
                                 data=result.mu_per_cm.numpy())
            f.attrs["n_iterations"] = int(result.n_iterations)
            f.attrs["converged"] = int(result.converged)
            f.attrs["final_loss"] = float(result.loss_history[-1]) if result.loss_history is not None else 0.0
    except ImportError:
        # h5py not available — write a numpy npz instead
        v_map_h5 = out_dir / "v_map.npz"
        np.savez(
            v_map_h5,
            V=result.V_voxel.numpy(),
            positions_um=voxel_pos_np,
            grain_map=grain_map_np,
            K=result.K_ring.numpy(),
            I_theory=I_theory.numpy(),
            n_iterations=int(result.n_iterations),
            converged=int(result.converged),
        )

    k_csv = out_dir / "k_per_ring.csv"
    np.savetxt(
        k_csv,
        np.column_stack([np.arange(n_rings), result.K_ring.numpy()]),
        header="ring_idx K_ring", fmt=["%d", "%.6e"], comments="",
    )

    loss_csv = out_dir / "vmap_loss_history.csv"
    if result.loss_history is not None and result.loss_history.numel() > 0:
        np.savetxt(
            loss_csv,
            np.column_stack([
                np.arange(int(result.loss_history.numel())),
                result.loss_history.numpy(),
            ]),
            header="iter loss", fmt=["%d", "%.6e"], comments="",
        )

    diag_paths: dict[str, str] = {}
    if cfg.emit_diagnostics:
        from ..diagnostics.vmap import (
            plot_loss_history, plot_per_grain_v_histograms,
            plot_spot_residuals, plot_v_map_overlay,
            write_k_per_ring_table, write_v_map_tif,
        )
        diag_dir = layer_dir / "diag"
        diag_dir.mkdir(parents=True, exist_ok=True)
        recon_dir = layer_dir / "Recons"
        recon_dir.mkdir(parents=True, exist_ok=True)

        V_np = result.V_voxel.numpy()
        K_np = result.K_ring.numpy()
        try:
            tif_path = write_v_map_tif(
                voxel_pos_np, V_np, recon_dir / "v_map.tif",
                axes=tuple(cfg.diag_axes),
            )
            diag_paths["v_map_tif"] = str(tif_path)
        except Exception as e:  # pragma: no cover - belt + suspenders
            LOG.warning("refine_vmap: v_map TIF failed (%s)", e)

        # Per-ring residual stats: gather from result.residuals_per_spot
        # grouped by spot_ring_idx (already loaded into spot_ring_t).
        resid_stats: dict[int, dict] = {}
        if result.residuals_per_spot is not None:
            resid_np = result.residuals_per_spot.numpy()
            ri_np = spot_ring_t.numpy()
            for r in range(n_rings):
                sel = (ri_np == r) & (resid_np != 0)
                if sel.any():
                    rs = resid_np[sel]
                    resid_stats[r] = {
                        "mean": float(rs.mean()), "std": float(rs.std()),
                        "n": int(rs.size),
                    }

        try:
            k_table = write_k_per_ring_table(
                K_np, I_theory.numpy(), diag_dir / "k_per_ring.csv",
                ring_numbers=np.arange(n_rings),
                residual_stats=resid_stats,
            )
            diag_paths["k_per_ring_csv"] = str(k_table)
        except Exception as e:  # pragma: no cover
            LOG.warning("refine_vmap: k_per_ring table failed (%s)", e)

        try:
            plot_v_map_overlay(
                voxel_pos_np, V_np, grain_map_np,
                diag_dir / "v_map_overlay.png",
                axes=tuple(cfg.diag_axes),
            )
            diag_paths["v_map_overlay_png"] = str(diag_dir / "v_map_overlay.png")
        except Exception as e:  # pragma: no cover
            LOG.warning("refine_vmap: v_map_overlay PNG failed (%s)", e)

        try:
            # Compute final predicted intensities for the residual plot
            with torch.no_grad():
                I_pred_final = predicted_spot_intensities(
                    result.V_voxel, result.K_ring, I_theory,
                    spot_ring_t, spot_grain_t,
                    spot_scan_pos_um, spot_ome_rad,
                    sg, beam,
                )
            plot_spot_residuals(
                spot_obs_t.numpy(), I_pred_final.numpy(),
                diag_dir / "spot_residuals.png",
            )
            diag_paths["spot_residuals_png"] = str(diag_dir / "spot_residuals.png")
        except Exception as e:  # pragma: no cover
            LOG.warning("refine_vmap: spot residual plot failed (%s)", e)

        try:
            if result.loss_history is not None and result.loss_history.numel() > 0:
                plot_loss_history(
                    result.loss_history.numpy(),
                    diag_dir / "refine_loss_history.png",
                )
                diag_paths["loss_png"] = str(diag_dir / "refine_loss_history.png")
        except Exception as e:  # pragma: no cover
            LOG.warning("refine_vmap: loss history plot failed (%s)", e)

        try:
            plot_per_grain_v_histograms(
                V_np, grain_map_np, diag_dir / "per_grain_v_histograms.png",
            )
            diag_paths["per_grain_hist_png"] = str(diag_dir / "per_grain_v_histograms.png")
        except Exception as e:  # pragma: no cover
            LOG.warning("refine_vmap: per-grain V histograms failed (%s)", e)

    finished = time.time()
    final_loss = (
        float(result.loss_history[-1].item())
        if result.loss_history is not None and result.loss_history.numel() > 0
        else 0.0
    )
    LOG.info(
        "refine_vmap: %d voxels × %d rings; %d iterations; "
        "converged=%s; final_loss=%.3e; %d diag artifacts; %.2fs",
        sg.n_voxels, n_rings, result.n_iterations,
        result.converged, final_loss, len(diag_paths), finished - started,
    )
    outputs_dict = {str(v_map_h5): "", str(k_csv): "", str(loss_csv): ""}
    for v in diag_paths.values():
        outputs_dict[v] = ""
    return RefineVmapResult(
        stage_name="refine_vmap",
        started_at=started, finished_at=finished, duration_s=finished - started,
        inputs={"radius_csv": str(radius_csv), "theory_csv": str(theory_csv)},
        outputs=outputs_dict,
        metrics={
            "n_voxels": sg.n_voxels, "n_rings": n_rings,
            "n_iterations": result.n_iterations,
            "converged": result.converged, "final_loss": final_loss,
            "n_diag_artifacts": len(diag_paths),
        },
        v_map_h5=str(v_map_h5),
        k_ring_csv=str(k_csv),
        loss_history_csv=str(loss_csv),
        n_voxels=sg.n_voxels, n_rings=n_rings,
        n_iterations=result.n_iterations,
        final_loss=final_loss,
        converged=result.converged,
    )

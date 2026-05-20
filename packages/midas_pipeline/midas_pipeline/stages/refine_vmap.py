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


def _spot_grain_from_indexing(
    layer_dir: Path,
    out_dir: Path,
    grain_map_np: np.ndarray,
    scan_nr: np.ndarray,
    ring_number: np.ndarray,
    omega_deg: np.ndarray,
    eta_deg: np.ndarray,
):
    """Orientation-based spot→grain attribution.

    Builds ``global_spot_id → grain`` from the indexer's per-voxel matched
    IDs (``IndexBest_IDs_all.bin``) and the voxel→grain map, then joins to
    the per-scan Radius_V spots via the physical ``(scan, ring, ω, η)``
    tuple (validated 1:1 against ``Spots.bin``). Returns an ``(n_spots,)``
    int64 array (``-1`` where a spot was not matched by any indexed voxel),
    or ``None`` when the required indexer artifacts are absent.
    """
    ids_path = out_dir / "IndexBest_IDs_all.bin"
    spots_path = layer_dir / "Spots.bin"
    if not spots_path.exists():
        spots_path = out_dir / "Spots.bin"
    if not ids_path.exists() or not spots_path.exists():
        return None
    try:
        from ..find_grains._consolidation_io import open_ids
    except ImportError:
        return None

    # global spot id → grain (orientation match). One solution per voxel in
    # the production PF path, so every matched id of voxel v belongs to its
    # grain. Voxels of the same clustered grain agree; on the rare cross-grain
    # conflict, last-writer-wins (negligible: ~0.04% of keys collide).
    idr = open_ids(ids_path)
    n_vox = int(grain_map_np.shape[0])
    gid_to_grain: dict[int, int] = {}
    for v in range(min(n_vox, idr.n_voxels)):
        g = int(grain_map_np[v])
        if g < 0:
            continue
        ids_v = idr.get_ids(v)
        if ids_v is None:
            continue
        for sid in ids_v:
            gid_to_grain[int(sid)] = g
    if not gid_to_grain:
        return None

    # global spot id → (scan, ring, ω, η) from Spots.bin; fold into a tuple
    # key → grain map. Spots.bin cols: [x y ome int spotID ring eta theta ds scan].
    sb = np.fromfile(spots_path, dtype=np.float64).reshape(-1, 10)
    key_to_grain: dict[tuple, int] = {}
    for r in sb:
        g = gid_to_grain.get(int(r[4]))
        if g is None:
            continue
        key_to_grain[(int(r[9]), int(r[5]), round(float(r[2]), 3),
                      round(float(r[6]), 3))] = g

    out = np.full(scan_nr.shape[0], -1, dtype=np.int64)
    for i in range(scan_nr.shape[0]):
        g = key_to_grain.get((int(scan_nr[i]), int(ring_number[i]),
                              round(float(omega_deg[i]), 3),
                              round(float(eta_deg[i]), 3)))
        if g is not None:
            out[i] = g
    return out


def _spot_grain_soft_from_indexing(
    layer_dir: Path,
    out_dir: Path,
    grain_map_np: np.ndarray,
    scan_nr: np.ndarray,
    ring_number: np.ndarray,
    omega_deg: np.ndarray,
    eta_deg: np.ndarray,
):
    """Soft multi-grain attribution.

    Most PF spots are matched by voxels of more than one grain (peak overlap +
    tolerance matching). Instead of forcing each spot onto a single grain (and
    dumping its whole intensity there), this returns an *expanded* membership
    list: for every (observed spot, grain) pair, a weight equal to the fraction
    of that spot's matching voxels belonging to the grain (Σ weights = 1 per
    spot). The forward model blends each grain's beam-weighted V-sum by these
    weights.

    Returns ``(out_index, grain, weight)`` numpy arrays (row = one membership;
    ``out_index`` points into the Radius_V row order) or ``None`` when the
    indexer artifacts are absent.
    """
    ids_path = out_dir / "IndexBest_IDs_all.bin"
    spots_path = layer_dir / "Spots.bin"
    if not spots_path.exists():
        spots_path = out_dir / "Spots.bin"
    if not ids_path.exists() or not spots_path.exists():
        return None
    try:
        from ..find_grains._consolidation_io import open_ids
    except ImportError:
        return None

    # global spot id → {grain: matched-voxel count}
    idr = open_ids(ids_path)
    n_vox = int(grain_map_np.shape[0])
    gid_counts: dict[int, dict[int, int]] = {}
    for v in range(min(n_vox, idr.n_voxels)):
        g = int(grain_map_np[v])
        if g < 0:
            continue
        ids_v = idr.get_ids(v)
        if ids_v is None:
            continue
        for sid in ids_v:
            d = gid_counts.setdefault(int(sid), {})
            d[g] = d.get(g, 0) + 1
    if not gid_counts:
        return None

    # (scan, ring, ω, η) tuple key → {grain: count}, via Spots.bin global ids.
    sb = np.fromfile(spots_path, dtype=np.float64).reshape(-1, 10)
    key_counts: dict[tuple, dict[int, int]] = {}
    for r in sb:
        c = gid_counts.get(int(r[4]))
        if c is None:
            continue
        key_counts[(int(r[9]), int(r[5]), round(float(r[2]), 3),
                    round(float(r[6]), 3))] = c

    out_index: list[int] = []
    grain: list[int] = []
    weight: list[float] = []
    for i in range(scan_nr.shape[0]):
        c = key_counts.get((int(scan_nr[i]), int(ring_number[i]),
                            round(float(omega_deg[i]), 3),
                            round(float(eta_deg[i]), 3)))
        if not c:
            continue
        total = float(sum(c.values()))
        for g, n in c.items():
            out_index.append(i)
            grain.append(g)
            weight.append(n / total)
    if not out_index:
        return None
    return (np.asarray(out_index, dtype=np.int64),
            np.asarray(grain, dtype=np.int64),
            np.asarray(weight, dtype=np.float64))


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
    spot_id     = arr[:, 0].astype(np.int64)
    scan_nr     = arr[:, 1].astype(np.int64)
    ring_number = arr[:, 2].astype(np.int64)
    ring_idx    = arr[:, 3].astype(np.int64)
    intensity   = arr[:, 4].astype(np.float64)
    omega_deg   = arr[:, 6].astype(np.float64)
    eta_deg     = arr[:, 7].astype(np.float64)

    theo = np.loadtxt(theory_csv, comments="#", skiprows=1)
    if theo.ndim == 1:
        theo = theo.reshape(1, -1)
    n_rings = int(theo.shape[0])
    I_theory = torch.as_tensor(theo[:, 2], dtype=dtype)

    # Spot → grain attribution. The physically correct source is the
    # *orientation* match recorded during indexing: a spot belongs to the
    # grain whose candidate orientation produced it, not to whatever grain
    # happens to sit nearest the spot's scan column. We recover that from
    # IndexBest_IDs_all.bin (per-voxel matched global spot IDs) + the
    # voxel→grain map, joined to Radius_V's per-scan spots via the physical
    # (scan, ring, omega, eta) tuple. Falls back to the scan-column spatial
    # heuristic when those indexer artifacts are absent (e.g. synthetic
    # fixtures that ship only voxel_grid.csv).
    spot_grain_idx = _spot_grain_from_indexing(
        layer_dir, out_dir, grain_map_np,
        scan_nr, ring_number, omega_deg, eta_deg,
    )
    if spot_grain_idx is None or not np.any(spot_grain_idx >= 0):
        if spot_grain_idx is None:
            LOG.info("refine_vmap: orientation-based spot→grain unavailable "
                     "(no IndexBest_IDs/Spots.bin); using scan-column fallback.")
        else:
            LOG.warning("refine_vmap: orientation-based spot→grain matched no "
                        "spots; using scan-column fallback.")
        spot_grain_idx = np.full_like(spot_id, -1)
        if scan_pos_np is not None:
            scan_pos_t = torch.as_tensor(scan_pos_np, dtype=dtype)
            vp_y = torch.as_tensor(voxel_pos_np[:, 1], dtype=dtype)
            for k, s in enumerate(scan_nr):
                if 0 <= s < scan_pos_t.numel():
                    d = (vp_y - scan_pos_t[s]).abs()
                    spot_grain_idx[k] = int(grain_map_np[int(d.argmin())])
        else:
            for k, s in enumerate(scan_nr):
                if 0 <= s < voxel_pos_np.shape[0]:
                    spot_grain_idx[k] = int(grain_map_np[s])
    LOG.info("refine_vmap: spot→grain attributed %d/%d spots across %d grains",
             int(np.count_nonzero(spot_grain_idx >= 0)), spot_grain_idx.size,
             len(set(spot_grain_idx[spot_grain_idx >= 0].tolist())))

    # Optional soft multi-grain attribution (expanded membership list).
    soft_expand = None
    if getattr(cfg, "soft_grain_attribution", False):
        soft_expand = _spot_grain_soft_from_indexing(
            layer_dir, out_dir, grain_map_np,
            scan_nr, ring_number, omega_deg, eta_deg,
        )
        if soft_expand is not None:
            oi, _gg, _ww = soft_expand
            n_spots_soft = len(set(oi.tolist()))
            LOG.info("refine_vmap: SOFT attribution — %d memberships over %d "
                     "spots (mean %.2f grains/spot)", oi.size, n_spots_soft,
                     oi.size / max(n_spots_soft, 1))
        else:
            LOG.warning("refine_vmap: soft attribution requested but "
                        "unavailable; using hard attribution.")

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

    # Soft attribution forwards an *expanded* (spot, grain) membership list to
    # the forward model; observed intensity stays per observed spot. K_init is
    # still computed from the hard (argmax) attribution above — it only needs
    # to be in the right ballpark; LBFGS refines K under the soft model.
    fwd_ring, fwd_grain = spot_ring_t, spot_grain_t
    fwd_scan, fwd_ome = spot_scan_pos_um, spot_ome_rad
    fwd_weight = fwd_out_index = None
    fwd_n_out = None
    if soft_expand is not None:
        oi, gg, ww = soft_expand
        oi_t = torch.as_tensor(oi, dtype=torch.int64)
        fwd_ring = spot_ring_t[oi_t]
        fwd_scan = spot_scan_pos_um[oi_t]
        fwd_ome = spot_ome_rad[oi_t]
        fwd_grain = torch.as_tensor(gg, dtype=torch.int64)
        fwd_weight = torch.as_tensor(ww, dtype=dtype)
        fwd_out_index = oi_t
        fwd_n_out = int(spot_obs_t.numel())

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
        spot_ring_idx=fwd_ring, spot_grain_idx=fwd_grain,
        spot_scan_pos_um=fwd_scan, spot_omega_rad=fwd_ome,
        sample_grid=sg, beam_profile=beam,
        theoretical_intensity_per_ring=I_theory,
        scan_axis=scan_axis, beam_geometry=cfg.beam_geometry,
        spot_weight=fwd_weight, spot_out_index=fwd_out_index,
        n_out_spots=fwd_n_out,
        refine_V=cfg.refine_V, refine_K=cfg.refine_K,
        refine_mu=cfg.refine_mu, refine_beam=cfg.refine_beam,
        max_iter=cfg.max_iter, loss_kind=cfg.loss_kind,
        tolerance=cfg.tolerance, gauge_reg=cfg.v_gauge_reg,
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
                    fwd_ring, fwd_grain, fwd_scan, fwd_ome,
                    sg, beam, scan_axis=scan_axis,
                    beam_geometry=cfg.beam_geometry,
                    spot_weight=fwd_weight, spot_out_index=fwd_out_index,
                    n_out_spots=fwd_n_out,
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

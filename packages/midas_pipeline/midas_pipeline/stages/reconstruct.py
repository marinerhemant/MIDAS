"""Stage: reconstruct (PF only).

Builds per-grain spatial reconstructions from the sinograms emitted by
the find_grains stage. Five backends per plan §3h, dispatched on
``ReconConfig.method``:

  fbp       → ``recon.fbp.fbp_recon_per_grain`` (delegates to TOMO/midas_tomo_python)
  mlem/osem → ``recon.mlem.mlem_recon`` / ``osem_recon`` (torch-native)
  voxelmap  → ``recon.voxelmap.voxelmap_recon`` (bypass tomo, direct
              per-voxel assignment from the indexer's top candidate)
  bayesian  → fbp + ``fuse.bayesian_fusion`` (handled by the fuse
              stage; reconstruct just runs fbp here as the prior)

FF mode is a no-op (FF has no tomographic recon step).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .._logging import LOG
from ..results import ReconResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def _read_sinograms(layer_dir: Path, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate sinos_{variant}_*.bin / omegas_*.bin / nrHKLs_*.bin emitted by find_grains.

    Returns (sinos, omegas, nr_hkls). The file naming uses a
    ``_<nGr>_<maxNHKLs>_<nScans>.bin`` suffix; we read whichever
    matching file is in the layer's Output dir.
    """
    output_dir = layer_dir / "Output"
    sino_files = sorted(output_dir.glob(f"sinos_{variant}_*.bin"))
    if not sino_files:
        raise FileNotFoundError(
            f"reconstruct: no sinos_{variant}_*.bin in {output_dir}"
        )
    sinos_path = sino_files[0]
    # Suffix is _<nGr>_<maxNHKLs>_<nScans>.bin
    parts = sinos_path.stem.split("_")
    n_grs, max_n_hkls, n_scans = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
    sinos = np.fromfile(sinos_path, dtype=np.float64).reshape(
        n_grs, max_n_hkls, n_scans,
    )
    # omegas + nr_hkls share the same n_grs/maxNHKLs prefix.
    omegas_path = next(output_dir.glob("omegas_*.bin"))
    nrhkls_path = next(output_dir.glob("nrHKLs_*.bin"))
    omegas = np.fromfile(omegas_path, dtype=np.float64).reshape(n_grs, max_n_hkls)
    nr_hkls = np.fromfile(nrhkls_path, dtype=np.int32)
    return sinos, omegas, nr_hkls


def _out_of_field_grains(layer_dir: Path, cutoff: float) -> list[int]:
    """Grains whose sinogram rows light up most of the scan line.

    Read from ``sinoOccupancy_*.bin`` (written by find_grains). These
    grains are comparable to or larger than the scanned field, so their
    *shapes* are not recoverable from these sinograms — but they are
    still real grains and must stay in the grain-ID competition.
    Excluding them was measured to be much worse (see
    :func:`..find_grains._sinogen.sinogram_occupancy`), so this is a
    warning, not a filter.
    """
    if cutoff <= 0:
        return []
    files = sorted((layer_dir / "Output").glob("sinoOccupancy_*.bin"))
    if not files:
        return []
    occ = np.fromfile(files[0], dtype=np.float64)
    flagged = [int(g) for g in np.flatnonzero(np.isfinite(occ) & (occ > cutoff))]
    if flagged:
        LOG.warning(
            "reconstruct(PF): grains %s have sinogram occupancy > %.2f "
            "(%s) — they fill or exceed the scanned field, so their "
            "reconstructed SHAPES are not trustworthy. Their grain-ID "
            "assignment is left untouched on purpose.",
            flagged, cutoff,
            ", ".join(f"g{g}={occ[g]:.2f}" for g in flagged),
        )
    return flagged


def run(ctx: StageContext) -> StageResult:
    if ctx.is_ff:
        return stub_run("reconstruct", ctx)

    cfg = ctx.config
    if not cfg.recon.do_tomo:
        LOG.info("reconstruct: do_tomo=False → no-op")
        return stub_run("reconstruct", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    recons_dir = layer_dir / "Recons"
    recons_dir.mkdir(parents=True, exist_ok=True)
    method = cfg.recon.method

    # Soft-skip when upstream find_grains hasn't emitted sinos and the
    # backend needs them. voxelmap is the only path that doesn't read
    # sinos — it consumes IndexBest_all.bin directly.
    sino_required = method != "voxelmap"
    if sino_required:
        any_sino = any(
            (layer_dir / "Output").glob(f"sinos_{cfg.recon.sino_type}_*.bin")
        )
        if not any_sino:
            LOG.info("reconstruct(PF): no sinos on disk → skip.")
            return stub_run("reconstruct", ctx)
    elif not (layer_dir / "Output" / "IndexBest_all.bin").exists():
        LOG.info("reconstruct(PF/voxelmap): no IndexBest_all.bin → skip.")
        return stub_run("reconstruct", ctx)

    LOG.info("reconstruct(PF): method=%s sino_type=%s n_scans=%d",
             method, cfg.recon.sino_type, cfg.scan.n_scans)
    out_of_field = _out_of_field_grains(
        layer_dir, getattr(cfg.recon, "out_of_field_occupancy", 0.0),
    )

    n_scans = int(cfg.scan.n_scans)

    if method == "voxelmap":
        from ..recon.voxelmap import voxelmap_recon
        all_recons = voxelmap_recon(
            topdir=layer_dir,
            sgnum=_read_space_group(layer_dir),
            nScans=n_scans,
            nGrs=_count_grains(layer_dir),
            max_ang_deg=cfg.fusion.max_ang_deg,
            min_conf=cfg.fusion.min_conf,
        )
    elif method in ("fbp", "bayesian"):
        from ..recon.fbp import fbp_recon_per_grain
        sinos, omegas, nr_hkls = _read_sinograms(layer_dir, cfg.recon.sino_type)
        all_recons = fbp_recon_per_grain(
            sinos_by_grain=sinos,
            omegas_by_grain=omegas,
            n_hkls_per_grain=nr_hkls,
            n_scans=n_scans,
            workingdir=recons_dir / "_fbp_tmp",
            num_cpus=cfg.n_cpus,
            do_cleanup=1,
            use_gpu=False,
        )
    elif method in ("mlem", "osem"):
        from ..recon.mlem import mlem_recon, osem_recon
        sinos, omegas, nr_hkls = _read_sinograms(layer_dir, cfg.recon.sino_type)
        n_grs = sinos.shape[0]
        all_recons = np.zeros((n_grs, n_scans, n_scans), dtype=np.float32)
        fn = mlem_recon if method == "mlem" else osem_recon
        for g in range(n_grs):
            n_hkl = int(nr_hkls[g])
            if n_hkl == 0:
                continue
            sino_g = sinos[g, :n_hkl, :]
            theta_g = omegas[g, :n_hkl]
            kwargs = {"n_iter": cfg.recon.mlem_iter}
            if method == "osem":
                kwargs["n_subsets"] = cfg.recon.osem_subsets
            recon = fn(sino_g, theta_g, n_pixels=n_scans, **kwargs)
            if hasattr(recon, "detach"):
                recon = recon.detach().cpu().numpy()
            all_recons[g] = np.asarray(recon, dtype=np.float32)
    else:
        raise ValueError(f"reconstruct: unknown method {method!r}")

    # Emit per-grain TIFs + a max-projection grain-ID map.
    try:
        import tifffile
    except ImportError:                                  # pragma: no cover
        tifffile = None

    per_grain_paths: list[str] = []
    if tifffile is not None:
        for g, rec in enumerate(all_recons):
            path = recons_dir / f"recon_grNr_{g:04d}.tif"
            tifffile.imwrite(str(path), rec.astype(np.float32))
            per_grain_paths.append(str(path))
        max_proj_grid = np.argmax(all_recons, axis=0).astype(np.int32)
        # -1 where no grain has any density.
        max_proj_grid = np.where(all_recons.max(axis=0) > 0,
                                 max_proj_grid, -1)
        max_proj_path = recons_dir / "Full_recon_max_project_grID.tif"
        tifffile.imwrite(str(max_proj_path), max_proj_grid)
    else:
        max_proj_path = recons_dir / "Full_recon_max_project_grID.tif"

    finished = time.time()
    return ReconResult(
        stage_name="reconstruct",
        started_at=started, finished_at=finished, duration_s=finished - started,
        method=method,
        per_grain_tifs=per_grain_paths,
        full_recon_max_project_grid_tif=str(max_proj_path),
        outputs={p: "" for p in per_grain_paths},
        metrics={"n_grains": int(all_recons.shape[0]),
                 "method": method, "sino_type": cfg.recon.sino_type,
                 "out_of_field_grains": out_of_field,
                 "n_out_of_field": len(out_of_field)},
    )


def _read_space_group(layer_dir: Path) -> int:
    p = layer_dir / "paramstest.txt"
    if not p.exists():
        return 225
    for line in p.read_text().splitlines():
        toks = line.split()
        if len(toks) >= 2 and toks[0] == "SpaceGroup":
            try:
                return int(toks[1])
            except ValueError:
                continue
    return 225


def _count_grains(layer_dir: Path) -> int:
    """Read UniqueOrientations.csv row count to size the recon stack."""
    p = layer_dir / "UniqueOrientations.csv"
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip())

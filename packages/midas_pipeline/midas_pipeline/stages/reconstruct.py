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
  all       → fbp, mlem and voxelmap, each with a quality row in
              ``Recons/ReconQuality.json`` (half-split, agreement with the
              per-voxel map, majority null, grid-convention check) and its
              label map ``Recons/labels_<method>.npy``; the per-grain TIFs and
              the max-projection map are FBP's. Which method is better depends
              on the data (recon/quality.py), so this reports all of them.

``ReconConfig.sample_mask`` (an (n_scans, n_scans) .npy/.tif, nonzero = sample;
build one from a tomogram with ``python -m midas_pipeline.recon.sample_mask``)
zeroes every reconstruction outside the sample, so vacuum voxels get label -1
instead of a grain, and restricts the quality report to it.

FF mode is a no-op (FF has no tomographic recon step).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

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
    from ._voxel_grid import require_positions_grid
    require_positions_grid(ctx.layer_dir, "reconstruct")

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

    mask = None
    mask_path = getattr(cfg.recon, "sample_mask", None)
    if mask_path:
        from ..recon.sample_mask import load_sample_mask
        mask = load_sample_mask(mask_path, n_scans)
        LOG.info("reconstruct(PF): sample mask %s (%d of %d voxels)", mask_path, int(mask.sum()), mask.size)

    quality = None
    if method == "all":
        all_recons, quality = _reconstruct_all(layer_dir, cfg, n_scans, recons_dir, mask)
    else:
        all_recons = _reconstruct(method, layer_dir, cfg, n_scans, recons_dir)
        if mask is not None:
            all_recons = np.where(mask[None], all_recons, 0).astype(all_recons.dtype)

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
                 "n_out_of_field": len(out_of_field),
                 "sample_mask": str(mask_path) if mask_path else "",
                 "quality": quality or {}},
    )



def _reconstruct(method: str, layer_dir: Path, cfg, n_scans: int, recons_dir: Path,
                 sino: Optional[tuple] = None) -> np.ndarray:
    """One backend's (n_grains, n_scans, n_scans) stack. ``sino`` = (sinos, omegas, nr_hkls) to reuse
    (and to reconstruct a half-split subset); read from Output/ when None."""
    if method == "voxelmap":
        from ..recon.voxelmap import voxelmap_recon
        all_recons = voxelmap_recon(
            topdir=layer_dir,
            sgnum=_read_space_group(layer_dir),
            n_scans=n_scans,
            n_grains=_count_grains(layer_dir),
            max_ang_deg=cfg.fusion.max_ang_deg,
            min_conf=cfg.fusion.min_conf,
        )
    elif method in ("fbp", "bayesian"):
        from ..recon.fbp import fbp_recon_per_grain
        sinos, omegas, nr_hkls = sino if sino is not None else _read_sinograms(layer_dir, cfg.recon.sino_type)
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
        sinos, omegas, nr_hkls = sino if sino is not None else _read_sinograms(layer_dir, cfg.recon.sino_type)
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
            # The image side is the sinogram width (n_scans); mlem_recon/osem_recon
            # take no size argument. Passing one (`n_pixels=`) crashed every
            # method="mlem"/"osem" run with a TypeError.
            recon = fn(sino_g, theta_g, **kwargs)
            if hasattr(recon, "detach"):
                recon = recon.detach().cpu().numpy()
            all_recons[g] = np.asarray(recon, dtype=np.float32)
    else:
        raise ValueError(f"reconstruct: unknown method {method!r}")
    return all_recons


def _reconstruct_all(layer_dir: Path, cfg, n_scans: int, recons_dir: Path, mask) -> tuple:
    """fbp + mlem (+ half-splits) + voxelmap; write labels_<m>.npy and ReconQuality.json. Returns (FBP stack, report)."""
    import json
    from ..recon.quality import half_rows, labels_from_stack, quality_entry
    sino = _read_sinograms(layer_dir, cfg.recon.sino_type)
    masked = (lambda R: np.where(mask[None], R, 0).astype(R.dtype)) if mask is not None else (lambda R: R)
    labs, halves, stacks = {}, {}, {}
    for m in ("fbp", "mlem"):
        R_raw = _reconstruct(m, layer_dir, cfg, n_scans, recons_dir / f"_{m}", sino=sino)
        R = masked(R_raw)
        if m == "mlem":
            mlem_unmasked = R_raw                          # the own-grain density check must see the vacuum, so it uses this
        stacks[m] = R; labs[m] = labels_from_stack(R)
        halves[m] = tuple(labels_from_stack(masked(_reconstruct(m, layer_dir, cfg, n_scans, recons_dir / f"_{m}_h{par}",
                                                                 sino=half_rows(*sino, par)))) for par in (0, 1))
        LOG.info("reconstruct(PF/all): %s done", m)
    pbp = None
    if (layer_dir / "Output" / "IndexBest_all.bin").exists():
        try:
            pbp = labels_from_stack(masked(np.asarray(_reconstruct("voxelmap", layer_dir, cfg, n_scans, recons_dir))))
            labs["voxelmap"] = pbp
        except Exception as e:                            # noqa: BLE001 - report without the PBP reference
            LOG.warning("reconstruct(PF/all): voxelmap (PBP reference) failed: %s", e)
    if mask is not None:
        support, support_is = mask, "sample mask"
    elif pbp is not None:
        support, support_is = pbp >= 0, "voxels the per-voxel (PBP) map solved - no sample mask given"
    else:
        support, support_is = labs["fbp"] >= 0, "voxels FBP assigned - no mask and no PBP map (includes vacuum)"
    rep = {"support": support_is, "methods": {}}
    for m, L in labs.items():
        rep["methods"][m] = quality_entry(L, support, pbp=pbp if m != "voxelmap" else None, halves=halves.get(m))
        np.save(recons_dir / f"labels_{m}.npy", L)
    (recons_dir / "OwnGrainDensity.npy").unlink(missing_ok=True)   # never leave a previous run's file next to a report without the key
    vg = layer_dir / "Output" / "voxel_grid.csv"
    if vg.exists():
        try:
            from ..recon.own_grain_density import own_grain_density
            # find_grains' own voxel -> grain assignment (the map the sinograms were built for), NOT the min_conf-filtered
            # voxel-map labels, and the UNMASKED MLEM stack: a sample mask would zero exactly the voxels this looks for.
            gid = np.loadtxt(vg, skiprows=1, usecols=4).astype(np.int64)
            if gid.size != n_scans * n_scans:
                raise ValueError(f"voxel_grid.csv has {gid.size} voxels, expected {n_scans * n_scans}")
            d = own_grain_density(mlem_unmasked, gid.reshape(n_scans, n_scans), sino[2])
            np.save(recons_dir / "OwnGrainDensity.npy", d["rho"].reshape(n_scans, n_scans))
            rep["own_grain_density"] = {k: d[k] for k in ("tau", "min_rows", "scored_voxels", "flagged_voxels",
                                                        "unscored_solved_voxels", "grains_below_min_rows", "per_grain")}
            rep["own_grain_density"]["note"] = ("flag = NOT-THIS-GRAIN (vacuum inheriting a grain, or material without density of its own grain); "
                                                "not proof of vacuum; grains with < min_rows rows are not scored; phantom-calibrated only")
            LOG.info("reconstruct(PF/all): own-grain density: %d of %d scored voxels below rho %.2f (%d solved voxels in grains "
                     "with < %d rows are not scored)", d["flagged_voxels"], d["scored_voxels"], d["tau"],
                     d["unscored_solved_voxels"], d["min_rows"])
        except Exception as e:                            # noqa: BLE001 - a diagnostic never stops the run
            LOG.warning("reconstruct(PF/all): own-grain density skipped: %s", e)
    (recons_dir / "ReconQuality.json").write_text(json.dumps(rep, indent=1))
    for m, e in rep["methods"].items():
        LOG.info("reconstruct(PF/all): %-8s half-split %s  vs PBP %s  majority null %.3f  (%s)", m,
                 f"{e['half_split']:.3f}" if "half_split" in e else "  -  ",
                 f"{e['agreement_vs_pbp']:.3f}" if "agreement_vs_pbp" in e else "  -  ", e["majority_null"], support_is)
        if not e.get("grid_convention_ok", True):
            LOG.warning("reconstruct(PF/all): %s agrees better with the PBP map under grid transform %d than as is "
                        "(%s): a grid-convention (transpose/flip) error", m, e["best_transform"],
                        [round(x, 3) for x in e["agreement_by_transform"]])
    return stacks["fbp"], rep

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
    """Row count of the find_grains grain list, to size the recon stack."""
    from .._grain_list import grain_list_path
    p = grain_list_path(layer_dir)
    if not p.exists():
        return 0
    # The grain list has a '# GrainID RowNr ...' header; counting it made n_grains one too many and
    # voxelmap_recon (whose genfromtxt skips comments) index past the end of the list.
    return sum(1 for line in p.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#"))

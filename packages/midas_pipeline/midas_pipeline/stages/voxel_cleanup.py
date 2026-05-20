"""Stage: voxel_cleanup (PF only) — missing-spot directionality cleanup.

Optional, OFF by default. Removes/reassigns mis-indexed voxels (lone juts,
orphans, small fragments) in ``Output/voxel_grid.csv`` using the directional
missing-spot signal. See ``midas_pipeline.voxel_cleanup`` and
``dev/paper/MISSING_SPOT_DIRECTIONALITY_CLEANUP.md``.

Validated for small/compact/tightly-supported grains; a near no-op (safe) on
large spread grains. Runs after find_grains so the cleaned ``voxel_grid.csv``
feeds the V-map path.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .._logging import LOG
from ..results import StageResult
from ._base import StageContext
from ._stub import stub_run


def _load_obs_by_ring(spots_bin: Path, positions: np.ndarray):
    """Spots.bin cols [x y ome int id ring eta th ds scan] -> per-ring
    (omega_deg, eta_deg, scan_position_um)."""
    sb = np.fromfile(spots_bin, dtype=np.float64).reshape(-1, 10)
    ring = sb[:, 5].astype(int)
    scanpos = positions[sb[:, 9].astype(int)]
    out = {}
    for r in np.unique(ring):
        m = ring == r
        out[int(r)] = (sb[m, 2], sb[m, 6], scanpos[m])
    return out


def run(ctx: StageContext) -> StageResult:
    cfg = getattr(ctx.config, "voxel_cleanup", None)
    if cfg is None or not cfg.run or ctx.is_ff:
        return stub_run("voxel_cleanup", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    out_dir = layer_dir / "Output"
    vg_path = out_dir / "voxel_grid.csv"
    paramstest = layer_dir / "paramstest.txt"
    uo_path = out_dir / "UniqueOrientations.csv"
    positions_csv = layer_dir / "positions.csv"
    spots_bin = layer_dir / "Spots.bin"
    if not spots_bin.exists():
        spots_bin = out_dir / "Spots.bin"
    for need in (vg_path, paramstest, uo_path, positions_csv, spots_bin):
        if not need.exists():
            LOG.info("voxel_cleanup: missing %s → skip.", need.name)
            return stub_run("voxel_cleanup", ctx)

    import os
    import torch
    from ..voxel_cleanup import cleanup_voxel_grid

    # --- forward adapter (configured from paramstest) ---------------------
    from midas_index.indexer import Indexer
    from midas_index.pipeline import IndexerContext
    ind = Indexer.from_param_file(paramstest, device="cpu", dtype="float64")
    cwd0 = Path.cwd()
    os.chdir(layer_dir)               # hkls.csv etc. resolve relative to cwd
    try:
        ind.load_observations(cwd=layer_dir)
        obs = ind._observations
        ictx = IndexerContext(
            params=ind.params, hkls_real=obs["hkls_real"],
            hkls_int=obs["hkls_int"], obs=obs["spots"],
            bin_data=obs["bin_data"], bin_ndata=obs["bin_ndata"],
            device="cpu", dtype=torch.float64,
        )
        adapter = ictx.adapter
    finally:
        os.chdir(cwd0)

    # --- layer artifacts ---------------------------------------------------
    vg = np.loadtxt(vg_path, skiprows=1)
    if vg.ndim == 1:
        vg = vg.reshape(1, -1)
    vx, vy, grain = vg[:, 1], vg[:, 2], vg[:, 4].astype(np.int64)
    positions = np.sort(np.loadtxt(positions_csv).ravel())
    obs_by_ring = _load_obs_by_ring(spots_bin, positions)

    uo = np.loadtxt(uo_path)
    if uo.ndim == 1:
        uo = uo[None, :]
    grain_OM = {gid: uo[gid, 5:14].reshape(3, 3) for gid in range(uo.shape[0])}
    grains = sorted(int(g) for g in set(grain[grain >= 0].tolist())
                    if int(g) in grain_OM)

    def predict_fn(g, vox_ids):
        R = torch.tensor(grain_OM[g], dtype=torch.float64).view(1, 3, 3)
        R = R.expand(len(vox_ids), 3, 3).contiguous()
        pos = torch.tensor(
            np.column_stack([vx[vox_ids], vy[vox_ids], np.zeros(len(vox_ids))]),
            dtype=torch.float64,
        )
        theor, valid = adapter.simulate(R, pos, lattice=None)
        return (theor[..., 6].numpy(), theor[..., 7].numpy(),
                theor[..., 9].numpy().astype(int), valid.numpy())

    pitch = float(np.median(np.diff(np.unique(vy))))
    res = cleanup_voxel_grid(
        predict_fn=predict_fn, vx=vx, vy=vy, grain=grain, grains=grains,
        obs_by_ring=obs_by_ring, pitch=pitch,
        margin_ome=float(ind.params.MarginOme),
        margin_eta=float(ind.params.MarginEta),
        scan_tol=float(ind.params.scan_pos_tol_um) or pitch / 2.0,
        score_threshold=cfg.score_threshold,
        max_same_neighbours=cfg.max_same_neighbours,
        max_iters=cfg.max_iters,
        occ_min_count=cfg.occ_min_count,
        action=cfg.action,
    )

    # --- write cleaned voxel_grid.csv (back up original) + sidecar --------
    n_acted = int(res.flagged.sum())
    if n_acted > 0:
        vg_path.replace(out_dir / "voxel_grid_precleanup.csv")
        out = vg.copy()
        out[:, 4] = res.new_grain
        np.savetxt(vg_path, out, header="voxel_idx x_um y_um z_um grain_id",
                   fmt=["%d", "%.4f", "%.4f", "%.4f", "%d"], comments="")
    np.savetxt(
        out_dir / "voxel_cleanup.csv",
        np.column_stack([np.arange(grain.size), grain, res.new_grain,
                         res.flagged.astype(int), res.directional, res.scalar]),
        header="voxel_idx grain_before grain_after flagged directional_score scalar_incompleteness",
        fmt=["%d", "%d", "%d", "%d", "%.4f", "%.4f"], comments="",
    )

    finished = time.time()
    LOG.info("voxel_cleanup: %d voxels acted on (%s) over %d passes %s; "
             "wrote cleaned voxel_grid.csv", n_acted, cfg.action,
             res.n_passes, res.per_pass_flagged)
    return StageResult(
        stage_name="voxel_cleanup",
        started_at=started, finished_at=finished, duration_s=finished - started,
        outputs={str(vg_path): "", str(out_dir / "voxel_cleanup.csv"): ""},
        metrics={"n_acted": n_acted, "n_passes": res.n_passes,
                 "action": cfg.action},
    )

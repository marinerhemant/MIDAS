"""Stage: fuse (PF only, optional).

Runs Bayesian fusion of per-grain tomographic recon shape × per-voxel
indexer orientation likelihood. Gated on
``cfg.recon.method == "bayesian"`` or ``cfg.fusion.enable_bayesian``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .._logging import LOG
from ..results import FuseResult, StageResult
from ._base import StageContext
from ._stub import stub_run


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


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.config
    if ctx.is_ff or not (
        cfg.fusion.enable_bayesian or cfg.recon.method == "bayesian"
    ):
        return stub_run("fuse", ctx)
    from ._voxel_grid import require_positions_grid
    require_positions_grid(ctx.layer_dir, "fuse")

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    recons_dir = layer_dir / "Recons"

    import tifffile
    tif_paths = sorted(recons_dir.glob("recon_grNr_*.tif"))
    if not tif_paths:
        raise FileNotFoundError(
            "fuse: no recon_grNr_*.tif in Recons/. Run reconstruct first."
        )
    all_recons = np.stack([tifffile.imread(str(p)) for p in tif_paths], axis=0)
    n_grs = all_recons.shape[0]

    LOG.info("fuse(PF): n_grains=%d max_ang=%.2f° min_conf=%.2f",
             n_grs, cfg.fusion.max_ang_deg, cfg.fusion.min_conf)

    from ..fuse import bayesian_fusion
    posterior = bayesian_fusion(
        all_recons=all_recons,
        topdir=layer_dir,
        sgnum=_read_space_group(layer_dir),
        n_grains=n_grs,
        max_ang_deg=cfg.fusion.max_ang_deg,
        min_conf=cfg.fusion.min_conf,
    )
    posterior_path = recons_dir / "posterior.npy"
    np.save(str(posterior_path), posterior)
    finished = time.time()
    return FuseResult(
        stage_name="fuse",
        started_at=started, finished_at=finished, duration_s=finished - started,
        posterior_path=str(posterior_path),
        outputs={str(posterior_path): ""},
        metrics={"n_grains": int(n_grs),
                 "max_ang_deg": cfg.fusion.max_ang_deg,
                 "min_conf": cfg.fusion.min_conf},
    )

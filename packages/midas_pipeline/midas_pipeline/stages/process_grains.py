"""Stage: process_grains — FF grain consolidation.

FF mode shells out to the standalone ``midas-process-grains`` (the same kernel
``midas-ff-pipeline`` uses): Stage-1 clustering + PassA dedup + confidence
filter + Kenesei strain, writing ``Grains.csv`` / ``SpotMatrix.csv`` /
``GrainIDsKey.csv`` into the layer dir. PF consolidation is handled elsewhere
(``find_grains`` / fuse), so this stage is FF-only.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from .._logging import LOG
from ..results import StageResult
from ._base import StageContext
from ._stub import stub_run


def run(ctx: StageContext) -> StageResult:
    if not ctx.is_ff:
        # PF path consolidates via find_grains/fuse, not here.
        return stub_run("process_grains", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    paramstest = layer_dir / "paramstest.txt"
    # refinement writes OrientPosFit.bin into Results/ (c-omp) or the layer dir
    # (python); process-grains reads it via the same paramstest folders.
    opf = layer_dir / "Results" / "OrientPosFit.bin"
    if not opf.exists():
        opf = layer_dir / "OrientPosFit.bin"
    if not paramstest.exists() or not opf.exists():
        LOG.info("process_grains(FF): missing paramstest or OrientPosFit.bin "
                 "→ skip.")
        return stub_run("process_grains", ctx)

    pg_paramstest = paramstest
    if ctx.config.indexer_backend == "c-omp":
        from ._comp_params import comp_backend_paramstest
        pg_paramstest = comp_backend_paramstest(paramstest, layer_dir)

    cmd = [
        sys.executable, "-m", "midas_process_grains",
        str(pg_paramstest),
        str(ctx.config.n_cpus),
        "--mode", ctx.config.process_grains_mode,
        "--device", ctx.config.device,
        "--dtype", ctx.config.dtype,
    ]
    LOG.info("process_grains(FF): %s", " ".join(cmd))
    log_dir = Path(ctx.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "process_grains_out.csv").open("w") as out_fp, \
         (log_dir / "process_grains_err.csv").open("w") as err_fp:
        subprocess.run(cmd, cwd=str(layer_dir), check=True,
                       stdout=out_fp, stderr=err_fp)

    finished = time.time()
    grains_csv = layer_dir / "Grains.csv"
    n_grains = 0
    if grains_csv.exists():
        for ln in grains_csv.open():
            if ln.startswith("%NumGrains"):
                n_grains = int(ln.split()[1])
                break
    LOG.info("process_grains(FF): %d grains → %s", n_grains, grains_csv)
    return StageResult(
        stage_name="process_grains",
        started_at=started, finished_at=finished, duration_s=finished - started,
        outputs={str(grains_csv): "",
                 str(layer_dir / "SpotMatrix.csv"): ""},
        metrics={"scan_mode": "ff", "n_grains": n_grains,
                 "mode": ctx.config.process_grains_mode},
    )

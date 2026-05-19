"""Stage: sinogen (PF only).

Sinogram generation already happens inside ``find_grains_single``
(see :mod:`midas_pipeline.find_grains`); this stage is a thin
verifier that the find_grains stage actually emitted the sino
binaries the reconstruct stage will consume.

If ``do_tomo=False`` or the sinograms aren't present (e.g. when
``find_grains_multiple`` was used), this stage is a clean skip.
"""

from __future__ import annotations

import time
from pathlib import Path

from .._logging import LOG
from ..results import SinogenResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def run(ctx: StageContext) -> StageResult:
    if ctx.is_ff or not ctx.config.recon.do_tomo:
        return stub_run("sinogen", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    output_dir = layer_dir / "Output"

    sinos_paths = {
        variant: sorted(output_dir.glob(f"sinos_{variant}_*.bin"))
        for variant in ("raw", "norm", "abs", "normabs", "softsum")
    }
    omegas = sorted(output_dir.glob("omegas_*.bin"))
    nrhkls = sorted(output_dir.glob("nrHKLs_*.bin"))

    # find_grains_single emits all four sino variants + omegas + nrHKLs
    # in one shot. If none are present we treat this as a skip (the
    # multi-solution path doesn't produce them).
    if not any(sinos_paths[v] for v in sinos_paths) or not omegas or not nrhkls:
        LOG.info("sinogen(PF): no sino binaries on disk — skipping.")
        return stub_run("sinogen", ctx)

    LOG.info("sinogen(PF): verified %d sino-variant binaries under %s",
             sum(len(v) for v in sinos_paths.values()), output_dir)

    primary = sinos_paths[ctx.config.recon.sino_type] or sinos_paths["raw"]
    primary_path = str(primary[0]) if primary else ""
    # Decode (n_grains, max_n_hkls, n_scans) from the binary's
    # filename suffix (consistent with the find_grains writer).
    parts = primary[0].stem.split("_") if primary else []
    n_grs = int(parts[-3]) if len(parts) >= 3 else 0
    max_n_hkls = int(parts[-2]) if len(parts) >= 3 else 0

    finished = time.time()
    return SinogenResult(
        stage_name="sinogen",
        started_at=started, finished_at=finished, duration_s=finished - started,
        sinos_paths={k: str(v[0]) for k, v in sinos_paths.items() if v},
        omegas_path=str(omegas[0]) if omegas else "",
        nr_hkls_path=str(nrhkls[0]) if nrhkls else "",
        n_grains=n_grs,
        max_n_hkls=max_n_hkls,
        outputs={primary_path: ""} if primary_path else {},
        metrics={"n_grains": n_grs, "max_n_hkls": max_n_hkls,
                 "sino_type": ctx.config.recon.sino_type},
    )

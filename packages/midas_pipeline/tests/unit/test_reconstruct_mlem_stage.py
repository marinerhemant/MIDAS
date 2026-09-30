"""The reconstruct stage's MLEM/OSEM branch runs end to end.

It crashed on every call: ``stages/reconstruct.py`` passed ``n_pixels=`` to
``recon.mlem.mlem_recon`` / ``osem_recon``, which take no size argument, so a
``method="mlem"`` or ``"osem"`` pipeline died with a TypeError. Nothing caught it
because the unit tests call ``mlem_recon`` directly and the stage tests only
exercised the skip paths (found on ESRF ma5608, 2026-09-22).

A two-grain phantom: each grain's sinogram is the forward projection of a disc,
so the stage must reconstruct the disc where it was and write one TIF per grain.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.config import PipelineConfig, ReconConfig, ScanGeometry
from midas_pipeline.recon.mlem import forward_project
from midas_pipeline.stages import reconstruct
from midas_pipeline.stages._base import StageContext

N = 24


def _ctx(tmp_path: Path, method: str) -> StageContext:
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"), params_file=str(params),
        scan=ScanGeometry.pf_uniform(n_scans=N, scan_step_um=1.0, beam_size_um=1.0),
        device="cpu", dtype="float64",
        recon=ReconConfig(do_tomo=True, method=method, mlem_iter=30, sino_type="raw"),
    )
    layer = tmp_path / "Layer1"
    (layer / "midas_log").mkdir(parents=True)
    return StageContext(config=cfg, layer_nr=1, layer_dir=layer,
                        log_dir=layer / "midas_log")


def _write_two_grain_sinos(layer: Path) -> list[np.ndarray]:
    out = layer / "Output"; out.mkdir()
    yy, xx = np.mgrid[0:N, 0:N]
    discs = [((yy - 7) ** 2 + (xx - 8) ** 2 < 16).astype(float),
             ((yy - 16) ** 2 + (xx - 15) ** 2 < 16).astype(float)]
    n_h = 12
    ang = np.linspace(-80, 85, n_h)
    sinos = np.stack([forward_project(d, ang) for d in discs])          # (2, n_h, N)
    sinos.astype(np.float64).tofile(out / f"sinos_raw_2_{n_h}_{N}.bin")
    np.stack([ang, ang]).astype(np.float64).tofile(out / f"omegas_2_{n_h}.bin")
    np.array([n_h, n_h], dtype=np.int32).tofile(out / "nrHKLs_2.bin")
    return discs


@pytest.mark.parametrize("method", ["mlem", "osem"])
def test_reconstruct_stage_mlem_osem_runs_and_recovers_the_discs(tmp_path, method):
    pytest.importorskip("tifffile")
    ctx = _ctx(tmp_path, method)
    discs = _write_two_grain_sinos(Path(ctx.layer_dir))
    if hasattr(ctx.config, "recon") and hasattr(ctx.config.recon, "osem_subsets"):
        ctx.config.recon.osem_subsets = 3
    result = reconstruct.run(ctx)
    assert not getattr(result, "skipped", False)
    import tifffile
    recs = [tifffile.imread(p) for p in sorted((Path(ctx.layer_dir) / "Recons").glob("recon_grNr_*.tif"))]
    assert len(recs) == 2 and all(r.shape == (N, N) for r in recs)
    for rec, disc in zip(recs, discs):
        inside, outside = rec[disc > 0].mean(), rec[disc == 0].mean()
        assert inside > 3 * max(outside, 1e-12), f"{method}: disc not reconstructed where it was"

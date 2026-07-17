"""Smoke tests for the PF stage-wiring (P-stage dispatchers).

Each stage now dispatches on ``ctx.is_ff`` / ``ctx.is_pf`` plus
config gates. We verify two contracts per stage:

1. FF mode (``scan_mode='ff'``) returns the legacy skipped stub
   regardless of other config — ensures we don't accidentally invoke
   PF compute paths when the user is running an FF pipeline.
2. PF mode with no on-disk inputs raises a clear ``FileNotFoundError``
   pointing at the missing artefact, so the orchestrator surfaces a
   helpful message instead of a cryptic deep-stack crash.

Heavy end-to-end paths (actual indexing / refinement / recon /
fusion) require real inputs and live in integration tests; this
suite stays fast (~1s) and covers the dispatch / error-shape
contracts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.config import PipelineConfig, ScanGeometry
from midas_pipeline.stages._base import StageContext
from midas_pipeline.stages import (
    em_refine, find_grains_stage, fuse, indexing, potts,
    reconstruct, refinement, sinogen,
)


# ---------------------------------------------------------------------------
# Context fixtures
# ---------------------------------------------------------------------------


def _ctx(tmp_path: Path, *, scan_mode: str = "pf",
         n_scans: int = 4, **cfg_overrides) -> StageContext:
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    if scan_mode == "ff":
        scan = ScanGeometry.ff()
    else:
        scan = ScanGeometry.pf_uniform(
            n_scans=n_scans, scan_step_um=2.0, beam_size_um=4.0,
        )
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=scan,
        device="cpu", dtype="float64",
        **cfg_overrides,
    )
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    return StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir,
                        log_dir=log_dir)


# ---------------------------------------------------------------------------
# FF mode — every dispatcher must return a skipped stub
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", [
    indexing, refinement, find_grains_stage, reconstruct,
    fuse, potts, em_refine, sinogen,
])
def test_ff_mode_returns_skipped(stage, tmp_path: Path):
    ctx = _ctx(tmp_path, scan_mode="ff")
    result = stage.run(ctx)
    assert result.skipped is True, (
        f"{stage.__name__}: FF mode should skip, got skipped={result.skipped}"
    )


# ---------------------------------------------------------------------------
# PF mode with missing upstream artefacts — soft skip (orchestrator-friendly)
# ---------------------------------------------------------------------------


def test_indexing_pf_missing_positions_is_hard_error(tmp_path: Path):
    """P0-2: missing positions.csv in PF mode is a HARD error (it made
    whole runs exit 0 doing nothing); missing paramstest still skips
    cleanly (smoke-test contract for partial pipelines)."""
    ctx = _ctx(tmp_path, scan_mode="pf")
    with pytest.raises(RuntimeError, match="positions.csv"):
        indexing.run(ctx)


def test_indexing_pf_missing_paramstest_skips(tmp_path: Path):
    ctx = _ctx(tmp_path, scan_mode="pf")
    (ctx.layer_dir / "positions.csv").write_text("0.0\n5.0\n")
    result = indexing.run(ctx)
    assert result.skipped is True


def test_refinement_pf_missing_index_best_all_skips(tmp_path: Path):
    ctx = _ctx(tmp_path, scan_mode="pf")
    (ctx.layer_dir / "positions.csv").write_text("0.0\n5.0\n")
    result = refinement.run(ctx)
    assert result.skipped is True


def test_find_grains_pf_missing_index_best_all_skips(tmp_path: Path):
    ctx = _ctx(tmp_path, scan_mode="pf")
    result = find_grains_stage.run(ctx)
    assert result.skipped is True


def test_reconstruct_pf_no_sinos_skips_cleanly(tmp_path: Path):
    """fbp + no sinos → skip rather than raise (orchestrator-friendly)."""
    ctx = _ctx(tmp_path, scan_mode="pf")
    result = reconstruct.run(ctx)
    assert result.skipped is True


def test_fuse_pf_disabled_by_default_returns_skip(tmp_path: Path):
    """fuse runs only when fusion.enable_bayesian or recon.method='bayesian'."""
    ctx = _ctx(tmp_path, scan_mode="pf")
    result = fuse.run(ctx)
    assert result.skipped is True


def test_potts_pf_disabled_by_default_returns_skip(tmp_path: Path):
    """potts runs only when fusion.cw_potts_lambda > 0."""
    ctx = _ctx(tmp_path, scan_mode="pf")
    result = potts.run(ctx)
    assert result.skipped is True


def test_em_refine_pf_disabled_by_default_returns_skip(tmp_path: Path):
    """em_refine runs only when em.enable=True."""
    ctx = _ctx(tmp_path, scan_mode="pf")
    result = em_refine.run(ctx)
    assert result.skipped is True


def test_sinogen_pf_no_sinos_skips_cleanly(tmp_path: Path):
    """sinogen with no find_grains output on disk → clean skip."""
    ctx = _ctx(tmp_path, scan_mode="pf")
    result = sinogen.run(ctx)
    assert result.skipped is True


def test_reconstruct_pf_skips_when_do_tomo_false(tmp_path: Path):
    """do_tomo=False short-circuits to a skipped stub."""
    from midas_pipeline.config import ReconConfig
    ctx = _ctx(tmp_path, scan_mode="pf",
               recon=ReconConfig(do_tomo=False))
    result = reconstruct.run(ctx)
    assert result.skipped is True

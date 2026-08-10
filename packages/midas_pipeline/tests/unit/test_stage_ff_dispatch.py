"""FF-mode dispatch tests for indexing + refinement stages.

The single-source contract: ``midas-pipeline run --scan-mode ff`` invokes
``python -m midas_index`` (indexing) and ``python -m midas_fit_grain``
(refinement) — the same kernels ``midas-ff-pipeline`` uses. These tests
mock ``subprocess.run`` and assert the command line + cwd are right;
they don't actually run the subprocess.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midas_pipeline.config import (
    PipelineConfig, ScanGeometry, RefinementConfig,
)
from midas_pipeline.stages._base import StageContext
from midas_pipeline.stages import indexing, refinement


def _ff_ctx(tmp_path: Path, *, has_files: bool, n_seeds: int = 3,
            indexer_backend: str = "python") -> StageContext:
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.ff(),
        device="cpu", dtype="float64",
        n_cpus=4,
        # Pin the python backend explicitly: the default is "c-omp", which
        # dispatches the installed C binary instead of `python -m midas_index`.
        # Without this pin the assertion below depends on whether the C
        # binary happens to be built on the test machine.
        indexer_backend=indexer_backend,
        refinement=RefinementConfig(solver="lbfgs", loss="angular", mode="all_at_once"),
    )
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    if has_files:
        (layer_dir / "paramstest.txt").write_text("RingNumbers 1\n")
        (layer_dir / "SpotsToIndex.csv").write_text(
            "\n".join(str(i) for i in range(n_seeds)) + "\n"
        )
    return StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir,
                        log_dir=log_dir)


def _write_stub_seeds(layer_dir: Path, n_seeds: int = 5) -> None:
    """Give the mocked indexer an output file, as a real one would.

    These tests assert on the COMMAND LINE, so they never cared what the
    subprocess wrote. The stage now raises when the indexer exits 0 having
    produced no recognisable seed file at all (issues/68 -- that silence used
    to surface as `cannot mmap an empty file` two stages later), so the mock
    has to model a successful indexer rather than a broken one.
    """
    import numpy as np
    out = layer_dir / "Output"
    out.mkdir(parents=True, exist_ok=True)
    rec = np.zeros((n_seeds, 15), dtype=np.float64)
    rec[:, 14] = 10.0                      # n_observed > 0 == "indexed"
    rec.tofile(out / "IndexBest.bin")


def test_indexing_ff_skips_when_artifacts_missing(tmp_path: Path):
    """No paramstest/SpotsToIndex → soft skip (smoke / partial-run path)."""
    result = indexing.run(_ff_ctx(tmp_path, has_files=False))
    assert result.skipped is True


def test_indexing_ff_invokes_midas_index_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """When inputs are present, dispatch shells to ``python -m midas_index``."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        _write_stub_seeds(Path(kwargs.get("cwd")))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    ctx = _ff_ctx(tmp_path, has_files=True, n_seeds=5)
    indexing.run(ctx)

    assert seen["cmd"][1:3] == ["-m", "midas_index"]
    # Positional args: paramstest, block_nr, n_blocks, n_seeds, n_cpus
    assert seen["cmd"][3].endswith("paramstest.txt")
    assert seen["cmd"][4:7] == ["0", "1", "5"]
    assert seen["cmd"][7] == "4"  # n_cpus
    assert seen["cwd"] == str(ctx.layer_dir)


def test_indexing_ff_comp_invokes_c_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Default backend (c-omp): dispatch shells to the C ``midas_indexer``."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0)

    # Pretend the C binary is built and points at a known path.
    fake_bin = tmp_path / "midas_indexer"
    monkeypatch.setattr("midas_index.backend_c.available", lambda: True)
    monkeypatch.setattr("midas_index.backend_c.binary_path", lambda: fake_bin)
    monkeypatch.setattr("subprocess.run", fake_run)
    _orig = fake_run

    def fake_run_writing(cmd, **kwargs):
        out = _orig(cmd, **kwargs)
        _write_stub_seeds(Path(kwargs.get("cwd")))
        return out

    monkeypatch.setattr("subprocess.run", fake_run_writing)

    ctx = _ff_ctx(tmp_path, has_files=True, n_seeds=5, indexer_backend="c-omp")
    indexing.run(ctx)

    assert seen["cmd"][0] == str(fake_bin)
    # c-omp is handed the backend-aware comp paramstest (OutputFolder/ResultFolder
    # rewritten to <layer_dir>/Output|Results so the C binary finds its binned
    # inputs); see stages/_comp_params.comp_backend_paramstest.
    assert seen["cmd"][1].endswith("paramstest_comp.txt")
    # Positional args: paramstest, block_nr, n_blocks, n_seeds, n_cpus
    assert seen["cmd"][2:5] == ["0", "1", "5"]
    assert seen["cmd"][5] == "4"  # n_cpus
    assert seen["cwd"] == str(ctx.layer_dir)


def test_refinement_ff_skips_when_artifacts_missing(tmp_path: Path):
    result = refinement.run(_ff_ctx(tmp_path, has_files=False))
    assert result.skipped is True


def test_refinement_ff_invokes_midas_fit_grain_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """When inputs are present, dispatch shells to ``python -m midas_fit_grain``."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    ctx = _ff_ctx(tmp_path, has_files=True, n_seeds=7)
    refinement.run(ctx)

    assert seen["cmd"][1:3] == ["-m", "midas_fit_grain"]
    assert seen["cmd"][3].endswith("paramstest.txt")
    assert seen["cmd"][4:7] == ["0", "1", "7"]
    assert seen["cmd"][7] == "4"  # n_cpus
    assert "--solver" in seen["cmd"]
    assert "lbfgs" in seen["cmd"]
    assert "--loss" in seen["cmd"]
    assert "angular" in seen["cmd"]
    assert seen["cwd"] == str(ctx.layer_dir)


def test_refinement_ff_swaps_pixel_to_angular_for_multidet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Multi-detector paramstest → pixel loss swapped to angular."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    ctx = _ff_ctx(tmp_path, has_files=True)
    (ctx.layer_dir / "paramstest.txt").write_text(
        "RingNumbers 1\nDetParams 0\n"
    )
    refinement.run(ctx)
    # pixel → angular swap
    loss_idx = seen["cmd"].index("--loss")
    assert seen["cmd"][loss_idx + 1] == "angular"


# ---------------------------------------------------------------------------
# FF + consolidated seeds — github.com/marinerhemant/MIDAS issues/68
# ---------------------------------------------------------------------------

def _write_consolidated_seeds(layer_dir: Path, n_vox: int = 5,
                              n_solved: int = 4) -> None:
    """What BOTH modern indexer backends actually write in FF mode."""
    import struct
    import numpy as np
    out = layer_dir / "Output"
    out.mkdir(parents=True, exist_ok=True)
    recs = np.zeros((n_vox, 16), dtype=np.float64)
    recs[:, 14] = 100.0                                   # n_expected
    recs[:n_solved, 15] = 60.0                            # n_observed
    with (out / "IndexBest_all.bin").open("wb") as f:
        f.write(struct.pack("<i", n_vox))
        f.write(np.ones(n_vox, dtype=np.int32).tobytes())
        f.write(np.arange(n_vox, dtype=np.int64).tobytes())
        f.write(recs.tobytes())
    for name in ("IndexKey_all.bin", "IndexBest_IDs_all.bin",
                 "IndexBest_weights_all.bin"):
        (out / name).write_bytes(b"\x00" * 8)


def test_ff_comp_reports_the_consolidated_seed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The FF stage must advertise the seed file it actually has.

    It used to hardcode the legacy pair: `index_best_bin` named an
    IndexBest.bin that was never written, `index_best_all_bin` was empty for
    the file that was, and `outputs` listed two non-existent paths. Anything
    reading the manifest got fiction (issues/68).
    """
    def fake_run(cmd, **kwargs):
        _write_consolidated_seeds(Path(kwargs.get("cwd")))
        return SimpleNamespace(returncode=0)

    fake_bin = tmp_path / "midas_indexer"
    monkeypatch.setattr("midas_index.backend_c.available", lambda: True)
    monkeypatch.setattr("midas_index.backend_c.binary_path", lambda: fake_bin)
    monkeypatch.setattr("subprocess.run", fake_run)

    ctx = _ff_ctx(tmp_path, has_files=True, n_seeds=5, indexer_backend="c-omp")
    res = indexing.run(ctx)

    assert res.n_seeds_indexed == 4, "the consolidated family must be counted"
    assert res.index_best_all_bin.endswith("IndexBest_all.bin")
    assert res.index_best_bin == "", "there is no legacy file to point at"
    assert res.metrics["seed_format"] == "consolidated"
    for p in res.outputs:
        assert Path(p).exists(), f"advertised a file that does not exist: {p}"


def test_ff_legacy_backend_still_reports_the_legacy_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    def fake_run(cmd, **kwargs):
        _write_stub_seeds(Path(kwargs.get("cwd")))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    res = indexing.run(_ff_ctx(tmp_path, has_files=True, n_seeds=5))

    assert res.index_best_bin.endswith("IndexBest.bin")
    assert res.index_best_all_bin == ""
    assert res.metrics["seed_format"] == "legacy"
    for p in res.outputs:
        assert Path(p).exists()


def test_ff_raises_when_the_indexer_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Exit 0 with no seed file is a broken contract, not a zero-seed result."""
    monkeypatch.setattr("subprocess.run",
                        lambda cmd, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="no recognisable seed file"):
        indexing.run(_ff_ctx(tmp_path, has_files=True, n_seeds=5))

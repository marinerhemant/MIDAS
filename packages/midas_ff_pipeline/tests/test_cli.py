"""CLI argv parsing — smoke checks each subcommand parses without erroring."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from midas_ff_pipeline.cli import _build_parser


def test_run_parser_accepts_minimum(tmp_path: Path):
    parser = _build_parser()
    args = parser.parse_args([
        "run",
        "--params", "p.txt",
        "--result", str(tmp_path),
    ])
    assert args.cmd == "run"
    assert args.layers == "1-1"
    assert args.device == "cuda"
    assert args.solver == "lbfgs"


def test_run_parser_propagates_overrides():
    parser = _build_parser()
    args = parser.parse_args([
        "run",
        "--params", "p.txt", "--result", "/tmp/x",
        "--layers", "2-7",
        "--device", "cpu", "--dtype", "float32",
        "--solver", "lm", "--loss", "angular", "--mode", "iterative",
        "--group-size", "8", "--pg-mode", "legacy",
        "--only", "indexing", "--only", "refinement",
    ])
    assert args.layers == "2-7"
    assert args.device == "cpu" and args.dtype == "float32"
    assert args.solver == "lm" and args.loss == "angular" and args.mode == "iterative"
    # ``--group-size`` is now string-typed at the argparse layer to allow
    # the ``auto`` sentinel; CLI resolver converts to int before
    # PipelineConfig sees it.
    assert args.group_size == "8" and args.pg_mode == "legacy"
    assert args.only == ["indexing", "refinement"]


def test_status_parser():
    parser = _build_parser()
    args = parser.parse_args(["status", "/tmp/run", "--json"])
    assert args.cmd == "status"
    assert args.json is True


def test_resume_parser_requires_from():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["resume", "/tmp/run"])
    args = parser.parse_args(["resume", "/tmp/run", "--from", "indexing"])
    assert args.from_stage == "indexing"


def test_inspect_parser():
    parser = _build_parser()
    args = parser.parse_args(["inspect", "/tmp/run/LayerNr_1", "--json"])
    assert args.layer_dir == "/tmp/run/LayerNr_1"
    assert args.json is True


def test_simulate_parser():
    parser = _build_parser()
    args = parser.parse_args([
        "simulate", "--out", "/tmp/sim",
        "--params", "Parameters.txt",
        "--n-grains", "100",
    ])
    assert args.n_grains == 100


def test_status_runs_on_empty_dir(tmp_path: Path, capsys):
    """status prints something useful even when no LayerNr_* exists."""
    from midas_ff_pipeline.cli import _cmd_status

    class _A:
        result_dir = str(tmp_path)
        layers = None
        json = False
    rc = _cmd_status(_A())
    assert rc == 2  # no LayerNr_* dirs


# ---- auto-resolver tests --------------------------------------------------

def test_resolve_dtype_auto():
    from midas_ff_pipeline.cli import _resolve_dtype
    assert _resolve_dtype("cuda", "auto") == "float32"
    assert _resolve_dtype("mps", "auto") == "float32"
    assert _resolve_dtype("cpu", "auto") == "float64"
    # explicit values pass through
    assert _resolve_dtype("cuda", "float64") == "float64"
    assert _resolve_dtype("cpu", "float32") == "float32"


def test_resolve_shard_gpus():
    from midas_ff_pipeline.cli import _resolve_shard_gpus
    assert _resolve_shard_gpus("cpu", "auto") is None
    assert _resolve_shard_gpus("cuda", "none") is None
    assert _resolve_shard_gpus("cuda", "") is None
    # explicit comma list
    assert _resolve_shard_gpus("cuda", "0,1") == "0,1"


def test_resolve_group_size_explicit():
    from midas_ff_pipeline.cli import _resolve_group_size
    assert _resolve_group_size("cuda", None, "1") == 1
    assert _resolve_group_size("cuda", "0,1", "16") == 16
    # CPU device → falls back to 4 regardless
    assert _resolve_group_size("cpu", None, "auto") == 4


def test_count_dataset_density(tmp_path: Path):
    from midas_ff_pipeline.cli import _count_dataset_density
    # Ti-7Al-shaped paramstest: 12 RingNumbers, 1440 omega steps.
    p = tmp_path / "ti7al.txt"
    p.write_text(
        "# header\n"
        "OmegaStart -180.0\n"
        "OmegaEnd 180.0\n"
        "OmegaStep 0.25\n"
        + "".join(f"RingNumbers {r}\n" for r in range(4, 16))
    )
    n_rings, n_omega = _count_dataset_density(str(p))
    assert n_rings == 12
    assert n_omega == 1440

    # Missing omega keys → n_omega=0, but rings still counted.
    p2 = tmp_path / "no_omega.txt"
    p2.write_text("RingNumbers 1\nRingNumbers 2\n")
    assert _count_dataset_density(str(p2)) == (2, 0)

    # Pipeline-style params.txt uses `RingThresh <ring> <thresh>` instead.
    p3 = tmp_path / "ringthresh.txt"
    p3.write_text(
        "OmegaStart -180.0\nOmegaEnd 180.0\nOmegaStep 0.25\n"
        + "".join(f"RingThresh {r} 100\n" for r in range(4, 16))
    )
    assert _count_dataset_density(str(p3)) == (12, 1440)

    # No params_file → (0, 0).
    assert _count_dataset_density(None) == (0, 0)
    assert _count_dataset_density(str(tmp_path / "missing.txt")) == (0, 0)


def test_resolve_group_size_density_scales_down(tmp_path: Path, monkeypatch):
    """Dense datasets (Ti-7Al-class) get smaller groups than the baseline."""
    from midas_ff_pipeline import cli

    # Stub torch to report an A6000-class 48 GB GPU (would normally pick gs=4).
    class _StubProps:
        total_memory = 48 * 1_000_000_000
    class _StubCuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 1
        @staticmethod
        def get_device_properties(i): return _StubProps()
    class _StubTorch:
        cuda = _StubCuda
    monkeypatch.setitem(__import__("sys").modules, "torch", _StubTorch)

    # Park22-shaped paramstest: 8 rings, 720 omega steps → density 1 → unchanged.
    p_baseline = tmp_path / "park22.txt"
    p_baseline.write_text(
        "OmegaStart -180.0\nOmegaEnd 180.0\nOmegaStep 0.5\n"
        + "".join(f"RingNumbers {r}\n" for r in range(1, 9))
    )
    assert cli._resolve_group_size("cuda", None, "auto",
                                   params_file=str(p_baseline)) == 4

    # Ti-7Al-shaped: 12 rings, 1440 steps → density 3.0 → 4 / 3 → 1.
    p_dense = tmp_path / "ti7al.txt"
    p_dense.write_text(
        "OmegaStart -180.0\nOmegaEnd 180.0\nOmegaStep 0.25\n"
        + "".join(f"RingNumbers {r}\n" for r in range(4, 16))
    )
    assert cli._resolve_group_size("cuda", None, "auto",
                                   params_file=str(p_dense)) == 1

    # No params_file → falls back to baseline tier (gs=4 for 48 GB).
    assert cli._resolve_group_size("cuda", None, "auto") == 4

    # Explicit override wins even when density would scale down.
    assert cli._resolve_group_size("cuda", None, "8",
                                   params_file=str(p_dense)) == 8


def test_resolve_group_size_density_never_scales_up(tmp_path: Path, monkeypatch):
    """Sparse datasets must not push the group size above the memory tier."""
    from midas_ff_pipeline import cli

    class _StubProps:
        total_memory = 48 * 1_000_000_000
    class _StubCuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 1
        @staticmethod
        def get_device_properties(i): return _StubProps()
    class _StubTorch:
        cuda = _StubCuda
    monkeypatch.setitem(__import__("sys").modules, "torch", _StubTorch)

    # 4 rings, 360 steps → density would be 0.25 but is clamped to 1.0.
    p_sparse = tmp_path / "sparse.txt"
    p_sparse.write_text(
        "OmegaStart -180.0\nOmegaEnd 180.0\nOmegaStep 1.0\n"
        + "".join(f"RingNumbers {r}\n" for r in range(1, 5))
    )
    assert cli._resolve_group_size("cuda", None, "auto",
                                   params_file=str(p_sparse)) == 4


def test_resolve_group_size_rings_only_when_omega_missing(tmp_path: Path, monkeypatch):
    """Indexer-style paramstest (OmegaRange line, no OmegaStep) still gets a
    rings-only down-scale — should not silently revert to the unscaled tier."""
    from midas_ff_pipeline import cli

    class _StubProps:
        total_memory = 48 * 1_000_000_000
    class _StubCuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 1
        @staticmethod
        def get_device_properties(i): return _StubProps()
    class _StubTorch:
        cuda = _StubCuda
    monkeypatch.setitem(__import__("sys").modules, "torch", _StubTorch)

    # 12 rings, omega not parseable → rings-only density = 12/8 = 1.5 → gs=2.
    p = tmp_path / "indexer_style.txt"
    p.write_text(
        "OmegaRange -180.0 180.0;\nOmeBinSize 0.1;\n"
        + "".join(f"RingNumbers {r}\n" for r in range(4, 16))
    )
    assert cli._resolve_group_size("cuda", None, "auto",
                                   params_file=str(p)) == 2

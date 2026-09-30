"""Tests for from_cache: loading pre-computed seed-orientation files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_nf_preprocess.seed_orientations import (
    DEFAULT_SEED_DIR,
    SeedCacheNotFound,
    load_seeds_for_lookup_type,
    load_seeds_for_space_group,
)


# ----- Helpers --------------------------------------------------------------


def _write_csv(path: Path, quats: np.ndarray) -> None:
    np.savetxt(path, quats, delimiter=",")


def _write_master_lookup(seed_dir: Path, lookup_type: str, master: np.ndarray, indices: np.ndarray) -> None:
    master.astype(np.float64).tofile(seed_dir / "orientations_master.bin")
    indices.astype(np.int32).tofile(seed_dir / f"lookup_{lookup_type}.bin")


# ----- CSV path -------------------------------------------------------------


def test_load_from_csv(tmp_path):
    quats = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    _write_csv(tmp_path / "seed_cubic_high.csv", quats)
    out = load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)
    assert out.shape == (3, 4)
    assert torch.allclose(out, torch.from_numpy(quats))


def test_load_from_csv_matches_space_group_route(tmp_path):
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    _write_csv(tmp_path / "seed_cubic_high.csv", quats)
    out_lt = load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)
    out_sg = load_seeds_for_space_group(225, seed_dir=tmp_path)
    assert torch.equal(out_lt, out_sg)


def test_load_from_csv_dtype_float32(tmp_path):
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    _write_csv(tmp_path / "seed_cubic_high.csv", quats)
    out = load_seeds_for_lookup_type(
        "cubic_high", seed_dir=tmp_path, dtype="fp32"
    )
    assert out.dtype == torch.float32


# ----- Master + lookup binary path ------------------------------------------


def test_load_from_master_lookup(tmp_path):
    """When CSV is absent, fall back to orientations_master.bin + lookup_*.bin."""
    master = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    indices = np.array([0, 2, 3], dtype=np.int32)  # pick rows 0, 2, 3
    _write_master_lookup(tmp_path, "cubic_high", master, indices)
    out = load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)
    expected = master[indices]
    assert out.shape == (3, 4)
    assert torch.allclose(out, torch.from_numpy(expected))


def test_csv_takes_precedence_over_master(tmp_path):
    """If both forms are present, the CSV should win (it's faster to load)."""
    csv_quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    _write_csv(tmp_path / "seed_cubic_high.csv", csv_quats)
    # Master with different content
    master = np.array([[0.5, 0.5, 0.5, 0.5]])
    indices = np.array([0], dtype=np.int32)
    _write_master_lookup(tmp_path, "cubic_high", master, indices)
    out = load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)
    assert torch.allclose(out, torch.from_numpy(csv_quats))


# ----- Errors ---------------------------------------------------------------


def test_missing_seed_dir_raises(tmp_path):
    with pytest.raises(SeedCacheNotFound):
        load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path / "missing")


def test_missing_files_in_seed_dir_raises(tmp_path):
    with pytest.raises(SeedCacheNotFound, match="--build-cache"):
        load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)


# ----- Clean checkout: no cache anywhere (issue #2) --------------------------


@pytest.fixture
def clean_install(tmp_path, monkeypatch):
    """No NF_HEDM/seedOrientations, no user cache, no MIDAS_NF_SEED_DIR."""
    from midas_nf_preprocess.seed_orientations import from_cache

    monkeypatch.delenv("MIDAS_NF_SEED_DIR", raising=False)
    monkeypatch.setattr(from_cache, "DEFAULT_SEED_DIR", tmp_path / "repo_missing")
    monkeypatch.setattr(from_cache, "USER_SEED_DIR", tmp_path / "user_cache")
    # Coarse grid keeps the tests fast; the real default is density-matched.
    monkeypatch.setitem(from_cache.CACHE_EQUIVALENT_RESOLUTION_DEG, "cubic", 12.0)
    return from_cache


def test_clean_install_error_names_file_and_command(clean_install, tmp_path):
    with pytest.raises(SeedCacheNotFound) as ei:
        load_seeds_for_space_group(225)
    msg = str(ei.value)
    assert "seed_cubic_high.csv" in msg
    assert str(tmp_path / "repo_missing") in msg
    assert str(tmp_path / "user_cache") in msg
    assert ("midas-nf-preprocess seed-orientations --method cache "
            "--space-group 225 --build-cache") in msg


def test_build_seed_cache_is_deterministic(clean_install, tmp_path):
    a = clean_install.build_seed_cache("cubic_high", seed_dir=tmp_path / "a")
    b = clean_install.build_seed_cache("cubic_high", seed_dir=tmp_path / "b")
    assert a.name == "seed_cubic_high.csv"
    assert a.read_bytes() == b.read_bytes()
    q = load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path / "a")
    assert q.shape[0] > 100 and q.shape[1] == 4
    assert torch.allclose(q.norm(dim=1), torch.ones(q.shape[0], dtype=q.dtype), atol=1e-6)


def test_build_if_missing_writes_user_cache_then_reuses_it(clean_install, tmp_path):
    q1 = load_seeds_for_space_group(225, build_if_missing=True)
    csv = tmp_path / "user_cache" / "seed_cubic_high.csv"
    assert csv.is_file()
    mtime = csv.stat().st_mtime_ns
    q2 = load_seeds_for_space_group(225)  # found now, no rebuild
    assert torch.equal(q1, q2)
    assert csv.stat().st_mtime_ns == mtime


def test_cli_cache_clean_install_fails_cleanly_then_builds(clean_install, tmp_path):
    from midas_nf_preprocess.seed_orientations.cli import main

    out = tmp_path / "seeds.csv"
    base = ["--method", "cache", "--space-group", "225", "--output", str(out)]
    with pytest.raises(SystemExit) as ei:
        main(base)
    assert "--build-cache" in str(ei.value.code)
    assert not out.exists()

    assert main(base + ["--build-cache", "--device", "cpu"]) == 0
    assert out.is_file()
    assert (tmp_path / "user_cache" / "seed_cubic_high.csv").is_file()


def test_env_var_seed_dir(tmp_path, monkeypatch):
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    _write_csv(tmp_path / "seed_cubic_high.csv", quats)
    monkeypatch.setenv("MIDAS_NF_SEED_DIR", str(tmp_path))
    out = load_seeds_for_lookup_type("cubic_high", seed_dir=None)
    assert out.shape == (1, 4)


# ----- Real cached files (skipped when not present) -------------------------


@pytest.mark.skipif(
    not (DEFAULT_SEED_DIR / "seed_cubic_high.csv").resolve().exists(),
    reason="bundled seed cache not present",
)
def test_load_real_cubic_high_cache():
    out = load_seeds_for_lookup_type("cubic_high")
    # The bundled cubicSeed.txt has ~243k entries.
    assert out.shape[0] > 100_000
    assert out.shape[1] == 4
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

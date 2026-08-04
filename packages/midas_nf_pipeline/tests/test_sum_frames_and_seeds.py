"""Seed generation and EndNr derivation.

SumFrames conversion is no longer here at all: the parameter file states the
experiment and every stage derives its own post-sum quantities, so there is
nothing to convert and nothing rewritten on disk. See
midas_nf_preprocess/tests/process_images/test_sum_frames_internal.py.

Seed orientations. The lookup cache lives in the source tree and is not
   shipped in the wheel, so a plain `pip install midas-suite` had no cache and
   the stage raised. Orientations are derivable, so they get derived.
"""
import logging
import pathlib
import os

import pytest

from midas_nf_pipeline import stages
from midas_nf_pipeline.params import parse_parameters


def _scan(tmp_path, n_files, start=5043, stem="nf_scan"):
    d = tmp_path / stem
    d.mkdir()
    for i in range(n_files):
        (d / f"{stem}_{start + i:06d}.tif").write_bytes(b"")
    return d


def _params(tmp_path, *, n_raw_files, nfd, sum_frames, n_dist=2, start=5043):
    _scan(tmp_path, n_raw_files, start=start)
    f = tmp_path / "p.txt"
    f.write_text(
        f"DataDirectory {tmp_path}/nf_scan\nOrigFileName nf_scan\nextOrig tif\n"
        f"StartNr {start}\nEndNr {start + nfd - 1}\nRawStartNr {start}\n"
        f"NrFilesPerDistance {nfd}\nnDistances {n_dist}\n"
        f"SumFrames {sum_frames}\nOmegaStep -0.1\n")
    return f







@pytest.fixture
def no_cache(monkeypatch, tmp_path):
    """Hide the source-tree cache.

    DEFAULT_SEED_DIR resolves relative to the package source, so in a checkout
    it finds the real cache and a test that only passes install_dir would load
    from it and never touch the generation path -- passing while proving
    nothing. This reproduces the pip-only install.
    """
    monkeypatch.setattr(
        "midas_nf_preprocess.seed_orientations.from_cache.DEFAULT_SEED_DIR",
        tmp_path / "no_such_cache")
    return tmp_path


def test_seeds_generated_when_cache_is_absent(no_cache, tmp_path, caplog):
    """pip-only install: no cache anywhere, seeds must still appear."""
    out = tmp_path / "seeds.csv"
    p = {"SpaceGroup": 225, "SeedOrientations": str(out)}
    with caplog.at_level(logging.INFO):
        stages.run_seed_orientations_from_cache(
            p, install_dir=str(tmp_path / "nonexistent"))
    assert "generated" in " ".join(r.getMessage() for r in caplog.records), (
        "fell back to the cache; the generation path was never exercised")
    assert out.exists(), "no seed file produced"
    n = sum(1 for _ in open(out))
    assert n > 10_000, f"implausibly few seeds: {n}"
    first = open(out).readline().strip().split(",")
    assert len(first) == 4, "seed CSV must be 4-column quaternions"
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "generating seeds" in text.lower() or "generated" in text.lower()


def test_generated_seed_count_is_near_the_cache_density(no_cache, tmp_path):
    """Generating at the sampler default would cost ~2.8x the fit time."""
    out = tmp_path / "seeds.csv"
    stages.run_seed_orientations_from_cache(
        {"SpaceGroup": 225, "SeedOrientations": str(out)},
        install_dir=str(tmp_path / "nope"))
    n = sum(1 for _ in open(out))
    assert 200_000 < n < 320_000, (
        f"{n} seeds; the shipped cubic cache has 243,129 and the resolution "
        f"was chosen to match it")


# ---------------------------------------------------------------------------
#  EndNr is derivable, so it should not have to be given
# ---------------------------------------------------------------------------

from midas_nf_pipeline.workflows import _derive_end_nr


def _params_no_endnr(tmp_path, *, nfd=1800, start=5043, n_files=3600):
    _scan(tmp_path, n_files, start=start)
    f = tmp_path / "p.txt"
    f.write_text(
        f"DataDirectory {tmp_path}/nf_scan\nOrigFileName nf_scan\nextOrig tif\n"
        f"StartNr {start}\nRawStartNr {start}\n"
        f"NrFilesPerDistance {nfd}\nnDistances 2\nSumFrames 1\nOmegaStep -0.1\n")
    return f


def test_end_nr_is_derived_when_absent(tmp_path, caplog):
    f = _params_no_endnr(tmp_path)
    p = parse_parameters(str(f))
    assert p.get("EndNr") is None
    with caplog.at_level(logging.INFO):
        _derive_end_nr(p, str(f))
    assert int(parse_parameters(str(f))["EndNr"]) == 5043 + 1800 - 1
    assert "derived" in " ".join(r.getMessage() for r in caplog.records)


def test_given_end_nr_is_not_overwritten(tmp_path):
    """A supplied value stays put -- deriving over it would hide a real mistake
    rather than let the consistency check report it."""
    f = _params(tmp_path, n_raw_files=3600, nfd=1800, sum_frames=1)
    p = parse_parameters(str(f))
    before = p["EndNr"]
    _derive_end_nr(p, str(f))
    assert parse_parameters(str(f))["EndNr"] == before



"""Two things the user should not have to know about.

1. SumFrames coupling. NrFilesPerDistance / EndNr / OmegaStep are POST-SUM,
   RawStartNr indexes RAW files. Raising SumFrames alone used to make the loader
   ask for NrFilesPerDistance x SumFrames raw files per distance and die on a
   filename hundreds past the end of the scan -- an error that says nothing
   about the real mistake. The parameter file may now describe the scan as
   acquired; the pipeline converts and logs it.

2. Seed orientations. The lookup cache lives in the source tree and is not
   shipped in the wheel, so a plain `pip install midas-suite` had no cache and
   the stage raised. Orientations are derivable, so they get derived.
"""
import logging
import os

import pytest

from midas_nf_pipeline import stages
from midas_nf_pipeline.workflows import _normalise_sum_frames
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


def test_raw_counts_are_converted(tmp_path, caplog):
    """1800 raw/distance x 2 with SumFrames 3 -> 600 post-sum, step -0.3."""
    f = _params(tmp_path, n_raw_files=3600, nfd=1800, sum_frames=3)
    p = parse_parameters(str(f))
    with caplog.at_level(logging.INFO):
        _normalise_sum_frames(p, str(f))
    out = parse_parameters(str(f))
    assert out["NrFilesPerDistance"] == 600
    assert int(out["EndNr"]) == 5043 + 599
    assert float(out["OmegaStep"]) == pytest.approx(-0.3)
    assert out["RawStartNr"] == 5043, "RawStartNr must keep indexing raw files"
    assert "RAW scan" in " ".join(r.getMessage() for r in caplog.records)


def test_already_post_sum_is_left_alone(tmp_path):
    """A file that already holds post-sum values must not be halved again."""
    f = _params(tmp_path, n_raw_files=3600, nfd=600, sum_frames=3)
    p = parse_parameters(str(f))
    _normalise_sum_frames(p, str(f))
    out = parse_parameters(str(f))
    assert out["NrFilesPerDistance"] == 600
    assert float(out["OmegaStep"]) == pytest.approx(-0.1)


def test_sum_frames_one_is_a_noop(tmp_path):
    f = _params(tmp_path, n_raw_files=3600, nfd=1800, sum_frames=1)
    p = parse_parameters(str(f))
    _normalise_sum_frames(p, str(f))
    assert parse_parameters(str(f))["NrFilesPerDistance"] == 1800


def test_indivisible_sum_frames_raises_with_the_real_reason(tmp_path):
    f = _params(tmp_path, n_raw_files=2000, nfd=1000, sum_frames=3)
    p = parse_parameters(str(f))
    with pytest.raises(ValueError, match="does not divide"):
        _normalise_sum_frames(p, str(f))


def test_short_scan_is_left_for_the_loader_to_report(tmp_path):
    """Neither interpretation fits: don't silently rewrite, let the loader name
    the missing file."""
    f = _params(tmp_path, n_raw_files=100, nfd=1800, sum_frames=3)
    p = parse_parameters(str(f))
    _normalise_sum_frames(p, str(f))
    assert parse_parameters(str(f))["NrFilesPerDistance"] == 1800


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

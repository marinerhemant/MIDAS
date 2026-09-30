"""The c-omp refiner's spot-weight options (RelFitRMSEWeightR0, SpotWeightsDirectional) and their files.

Both weight files are read right after ExtraInfo.bin is mapped, so a tiny staged layer is enough: a set option with a
missing, short or invalid file must STOP the refiner with a clear message (a silently unweighted run would pass for a
weighted one), and a valid file must be reported. Refinement itself is not exercised here; the effect of the weights
was measured end to end against EBSD (LSHR layer 1) and on repeated load steps, and "off" was shown bit-identical to
the unpatched refiner on three datasets.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from midas_fit_grain import backend_c


def _has_options() -> bool:
    if not backend_c.available():
        return False
    blob = Path(backend_c.binary_path()).read_bytes()
    return b"RelFitRMSEWeightR0" in blob and b"SpotWeightsDirectional" in blob


pytestmark = pytest.mark.skipif(
    not _has_options(), reason="midas_fitgrain binary missing or built before the spot-weight options")

N_ROWS = 4


def _stage(d: Path, extra: list[str]) -> Path:
    (d / "Results").mkdir(parents=True)
    (d / "Output").mkdir()
    lines = [
        "Wavelength 0.18", "Lsd 1000000", "px 200",
        "LatticeParameter 3.6 3.6 3.6 90 90 90", "SpaceGroup 225",
        "RingNumbers 1", "RingRadii 500",
        f"OutputFolder {d / 'Output'}", f"ResultFolder {d / 'Results'}",
    ] + extra
    (d / "paramstest.txt").write_text("\n".join(lines) + "\n")
    np.zeros((N_ROWS, 16)).tofile(d / "ExtraInfo.bin")
    return d


def _run(d: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(backend_c.binary_path()), str(d / "paramstest.txt"), "0", "1", "1", "1"],
        cwd=d, capture_output=True, timeout=120)


def _out(p: subprocess.CompletedProcess) -> str:
    return (p.stdout + p.stderr).decode(errors="replace")


def test_rel_weight_without_file_is_refused(tmp_path):
    p = _run(_stage(tmp_path, ["RelFitRMSEWeightR0 0.69"]))
    assert p.returncode != 0 and "RelFitRMSE.bin" in _out(p)


def test_rel_weight_wrong_length_is_refused(tmp_path):
    d = _stage(tmp_path, ["RelFitRMSEWeightR0 0.69"])
    np.zeros(N_ROWS - 1).tofile(d / "RelFitRMSE.bin")
    p = _run(d)
    assert p.returncode != 0 and "does not hold exactly" in _out(p)


def test_rel_weight_file_is_loaded_and_reported(tmp_path):
    d = _stage(tmp_path, ["RelFitRMSEWeightR0 0.69"])
    np.array([0.0, 0.69, np.nan, 1.38]).tofile(d / "RelFitRMSE.bin")
    out = _out(_run(d))
    # w = 1, 0.5, 1 (unknown), 1/3 -> mean 0.7083; one row without a value
    assert "RelFitRMSE weights: r0 0.69, 4 rows, 1 without a value" in out
    assert "mean w 0.7083" in out


def test_directional_without_file_is_refused(tmp_path):
    p = _run(_stage(tmp_path, ["SpotWeightsDirectional 1"]))
    assert p.returncode != 0 and "SpotWeights.bin" in _out(p)


def test_directional_nonpositive_weight_is_refused(tmp_path):
    d = _stage(tmp_path, ["SpotWeightsDirectional 1"])
    w = np.ones((N_ROWS, 3)); w[2, 1] = 0.0
    w.tofile(d / "SpotWeights.bin")
    p = _run(d)
    assert p.returncode != 0 and "non-positive" in _out(p)


def test_directional_file_is_loaded_and_both_options_announce_stage_specific(tmp_path):
    d = _stage(tmp_path, ["SpotWeightsDirectional 1", "RelFitRMSEWeightR0 0.69"])
    w = np.ones((N_ROWS, 3)); w[:, 1] = 0.5
    w.tofile(d / "SpotWeights.bin")
    np.zeros(N_ROWS).tofile(d / "RelFitRMSE.bin")
    out = _out(_run(d))
    assert "mean w_rad 1.0000 w_tan 0.5000 w_ome 1.0000" in out
    assert "stage-specific" in out

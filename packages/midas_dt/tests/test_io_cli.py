"""Result writing with provenance, and CLI wiring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from midas_dt.branches import BranchResult
from midas_dt.channels import Channel
from midas_dt.conventions import ScanKnownLimits
from midas_dt.io import read_legacy_reconstruction, write_result


def _result():
    return BranchResult(
        maps={"RMEAN": np.full((4, 4), 118.0),
              "TotalIntensity": np.full((4, 4), 900.0)},
        branch="fit-then-recon[intensity]", channel=Channel(105, 125),
        limits=ScanKnownLimits(snake_corrected=True, omega_negated=True),
        linearity={"RMEAN": "weighted-moment", "TotalIntensity": "exact"},
    )


def test_write_result_saves_maps_and_provenance(tmp_path):
    d = write_result(_result(), tmp_path / "out")
    assert (d / "RMEAN.npy").is_file()
    np.testing.assert_allclose(np.load(d / "RMEAN.npy"), 118.0)
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["branch"] == "fit-then-recon[intensity]"
    assert prov["linearity"]["RMEAN"] == "weighted-moment"


def test_provenance_carries_the_approximation_and_the_limits(tmp_path):
    """A map must not be separable from its caveats by copying a file."""
    d = write_result(_result(), tmp_path / "out")
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["approximate_outputs"] == ["RMEAN"]
    assert any("Self-absorption" in w for w in prov["known_limits"])
    assert prov["snake_corrected"] is True


def test_npy_carries_its_own_shape(tmp_path):
    """The legacy .bin encoded shape in the filename, so a rename lost it."""
    d = write_result(_result(), tmp_path / "out")
    assert np.load(d / "RMEAN.npy").shape == (4, 4)


def test_hdf5_output_tags_each_map_with_its_linearity(tmp_path):
    h5py = pytest.importorskip("h5py")
    from midas_dt.io import write_maps_hdf5

    p = write_maps_hdf5(_result(), tmp_path / "maps.h5")
    with h5py.File(p, "r") as hf:
        assert hf["maps/RMEAN"].attrs["linearity"] == "weighted-moment"
        assert hf["maps/TotalIntensity"].attrs["linearity"] == "exact"
        assert "Self-absorption" in hf.attrs["known_limits"]


# ------------------------------------------------------- legacy reader
def test_read_legacy_reconstruction_shape(tmp_path):
    size, n_out = 8, 12
    data = np.arange(size * size * n_out, dtype=np.float64)
    p = tmp_path / "PeakFitResult.bin"
    data.tofile(p)
    arr = read_legacy_reconstruction(p, size)
    assert arr.shape == (size, size, 1, 1, n_out)


def test_legacy_reader_rejects_a_shape_mismatch(tmp_path):
    p = tmp_path / "PeakFitResult.bin"
    np.arange(10, dtype=np.float64).tofile(p)
    with pytest.raises(ValueError, match="implies"):
        read_legacy_reconstruction(p, 8)


# --------------------------------------------------------------- CLI
def test_cli_requires_a_real_parameter_file(capsys):
    from midas_dt.cli import main
    rc = main(["--params", "nope.txt", "--raw-dir", ".", "--stem", "x",
               "--start", "1", "--end", "2", "--out", "/tmp/x",
               "--r-min", "105", "--r-max", "125"])
    assert rc == 2
    assert "no such parameter file" in capsys.readouterr().err


def test_cli_defaults_to_the_exact_branch():
    from midas_dt.cli import _build_parser
    args = _build_parser().parse_args(
        ["--params", "p", "--raw-dir", ".", "--stem", "x", "--start", "1",
         "--end", "2", "--out", "o", "--r-min", "105", "--r-max", "125"])
    assert args.branch == "recon-fit"
    assert args.weighting == "intensity"


def test_cli_rejects_an_unknown_branch():
    from midas_dt.cli import _build_parser
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["--params", "p", "--raw-dir", ".", "--stem", "x", "--start", "1",
             "--end", "2", "--out", "o", "--r-min", "105", "--r-max", "125",
             "--branch", "both"])

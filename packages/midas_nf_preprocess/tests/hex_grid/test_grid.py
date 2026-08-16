"""Tests for the hex_grid submodule."""

from __future__ import annotations

import math
import subprocess

import numpy as np
import pytest
import torch

from midas_nf_preprocess.hex_grid import (
    HexGrid,
    HexGridParams,
    make_hex_grid,
    n_grid_points,
    read_grid_txt,
    write_grid_txt,
)


# -----------------------------------------------------------------------------
# n_grid_points closed-form
# -----------------------------------------------------------------------------


def test_n_grid_points_matches_make_hex_grid():
    """The closed-form count must equal len(make_hex_grid(...))."""
    for grid_size, r_sample in [(1.0, 5.0), (2.0, 10.0), (1.5, 7.0), (3.0, 15.0)]:
        n = n_grid_points(grid_size, r_sample)
        g = make_hex_grid(grid_size, r_sample)
        assert g.shape == (n, 5), (
            f"n_grid_points({grid_size}, {r_sample}) = {n}, "
            f"but grid has {g.shape[0]} rows"
        )


def test_n_grid_points_formula():
    """Documented closed form: N = 6 * NrHex^2."""
    grid_size = 2.0
    r_sample = 10.0
    a_large = (2.0 * r_sample) / math.sqrt(3.0)
    nr_hex = int(math.ceil(a_large / grid_size))
    assert n_grid_points(grid_size, r_sample) == 6 * nr_hex * nr_hex


# -----------------------------------------------------------------------------
# Output structure
# -----------------------------------------------------------------------------


def test_make_hex_grid_shape():
    g = make_hex_grid(grid_size=1.0, r_sample=5.0)
    assert g.ndim == 2
    assert g.shape[1] == 5


def test_make_hex_grid_dtype_default_float64():
    g = make_hex_grid(grid_size=1.0, r_sample=5.0)
    assert g.dtype == torch.float64


def test_make_hex_grid_edge_half_constant():
    """Last column = edge_length/2 for every voxel."""
    g = make_hex_grid(grid_size=2.0, r_sample=8.0)
    assert torch.allclose(g[:, 4], torch.full_like(g[:, 4], 1.0))


def test_make_hex_grid_edge_length_default():
    """edge_length defaults to grid_size when not given (matches C L158-L159)."""
    g_default = make_hex_grid(grid_size=2.0, r_sample=8.0)
    g_explicit = make_hex_grid(grid_size=2.0, r_sample=8.0, edge_length=2.0)
    assert torch.equal(g_default, g_explicit)


def test_make_hex_grid_distinct_xy():
    """No two voxels should share both (x, y) coordinates."""
    g = make_hex_grid(grid_size=1.5, r_sample=6.0)
    xy = g[:, [2, 3]].numpy()
    n_unique = len({(round(float(x), 6), round(float(y), 6)) for x, y in xy})
    assert n_unique == g.shape[0]


def test_make_hex_grid_origin_excluded():
    """Row i=0 is skipped in the C; ensure no voxel center is exactly at (0, 0)."""
    g = make_hex_grid(grid_size=1.0, r_sample=5.0)
    has_origin = torch.any((g[:, 2] == 0.0) & (g[:, 3] == 0.0))
    assert not bool(has_origin)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def test_make_hex_grid_invalid_grid_size():
    with pytest.raises(ValueError, match="grid_size"):
        make_hex_grid(grid_size=0.0, r_sample=5.0)


def test_make_hex_grid_invalid_r_sample():
    with pytest.raises(ValueError, match="r_sample"):
        make_hex_grid(grid_size=1.0, r_sample=-1.0)


def test_make_hex_grid_invalid_edge_length():
    with pytest.raises(ValueError, match="edge_length"):
        make_hex_grid(grid_size=1.0, r_sample=5.0, edge_length=0.0)


# -----------------------------------------------------------------------------
# HexGrid wrapper
# -----------------------------------------------------------------------------


def test_hexgrid_make_and_props():
    g = HexGrid.make(grid_size=1.0, r_sample=5.0)
    assert g.n_points == len(g)
    assert g.x.shape == (len(g),)
    assert g.y.shape == (len(g),)
    assert g.dx.shape == (len(g),)
    assert g.dy.shape == (len(g),)
    assert g.edge_half.shape == (len(g),)


def test_hexgrid_filter_keeps_subset():
    g = HexGrid.make(grid_size=1.0, r_sample=5.0)
    mask = g.x > 0  # keep right half
    g_right = g.filter(mask)
    assert g_right.n_points == int(mask.sum())
    assert torch.all(g_right.x > 0)


def test_hexgrid_from_params(tmp_path):
    pf = tmp_path / "p.txt"
    pf.write_text("GridSize 1.5\nRsample 6.0\n")
    p = HexGridParams.from_paramfile(pf)
    g = HexGrid.from_params(p)
    assert g.n_points == n_grid_points(1.5, 6.0)


# -----------------------------------------------------------------------------
# I/O round-trip
# -----------------------------------------------------------------------------


def test_grid_txt_roundtrip(tmp_path):
    g = make_hex_grid(grid_size=1.0, r_sample=4.0)
    path = tmp_path / "grid.txt"
    write_grid_txt(g, path)
    g2 = read_grid_txt(path)
    # Values written with %f precision (6 decimals); compare with tolerance.
    assert g.shape == g2.shape
    assert torch.allclose(g, g2, atol=1e-5)


def test_grid_txt_header_count(tmp_path):
    g = make_hex_grid(grid_size=1.0, r_sample=4.0)
    path = tmp_path / "grid.txt"
    write_grid_txt(g, path)
    with open(path) as f:
        first = f.readline().strip()
    assert int(first) == g.shape[0]


def test_grid_txt_wrong_count_raises(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("3\n0 0 0 0 0\n")  # header lies
    with pytest.raises(ValueError, match="header says"):
        read_grid_txt(path)


# -----------------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------------


def test_params_defaults_edge_length_falls_back():
    p = HexGridParams(grid_size=2.0)
    assert p.edge_length == 2.0


def test_params_output_dir_defaults():
    p = HexGridParams(grid_size=1.0, data_directory="/x")
    assert p.output_directory == "/x"


def test_params_from_paramfile(tmp_path):
    pf = tmp_path / "ps.txt"
    pf.write_text(
        "GridSize 1.5\n"
        "EdgeLength 2.0\n"
        "Rsample 8.0\n"
        "DataDirectory /tmp/foo\n"
        "OutputDirectory /tmp/bar\n"
        "GridFileName mygrid.txt\n"
    )
    p = HexGridParams.from_paramfile(pf)
    assert p.grid_size == 1.5
    assert p.edge_length == 2.0
    assert p.r_sample == 8.0
    assert p.data_directory == "/tmp/foo"
    assert p.output_directory == "/tmp/bar"
    assert p.grid_filename == "mygrid.txt"


# -----------------------------------------------------------------------------
# C parity (skipped if the compiled MakeHexGrid binary is unavailable)
# -----------------------------------------------------------------------------


def _c_binary_runs(path) -> bool:
    """True only if the C reference actually EXECUTES.

    Existence is not enough. These binaries were linked against
    ``MIDAS/build/lib``; once that tree goes away the loader fails with
    ``Library not loaded: @rpath/libnlopt.1.dylib`` and the process aborts
    before doing anything. A skipif that tests only for the file then lets a
    dead binary through and the parity test reports a FAILURE -- implying the
    Python disagrees with C, when C never ran and nothing was compared.
    """
    import subprocess
    from pathlib import Path as _P
    if not _P(str(path)).exists():
        return False
    try:
        r = subprocess.run([str(path)], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    # A usage message on empty argv is fine (any exit code >= 0). A negative
    # returncode means killed by a signal -- the dyld abort is -6 (SIGABRT) --
    # and a loader message on stderr says the same thing explicitly.
    if r.returncode < 0:
        return False
    return b"Library not loaded" not in (r.stderr or b"")


C_BINARY = "/Users/hsharma/opt/MIDAS/NF_HEDM/bin/MakeHexGrid"


@pytest.mark.parity
@pytest.mark.skipif(
    not _c_binary_runs(C_BINARY),
    reason="compiled MakeHexGrid binary missing or does not run",
)
def test_make_hex_grid_matches_c(tmp_path):
    """Bit-comparable parity with the C MakeHexGrid for a small grid."""
    pf = tmp_path / "p.txt"
    pf.write_text(
        f"GridSize 2.0\n"
        f"EdgeLength 2.0\n"
        f"Rsample 6.0\n"
        f"DataDirectory {tmp_path}\n"
    )
    subprocess.run([C_BINARY, str(pf)], check=True, capture_output=True)
    c_grid = read_grid_txt(tmp_path / "grid.txt")
    py_grid = make_hex_grid(grid_size=2.0, r_sample=6.0)
    # The C writes %f (6 decimals); compare with appropriate tolerance.
    assert c_grid.shape == py_grid.shape
    assert torch.allclose(c_grid, py_grid, atol=1e-5)

"""The ``Grains.csv`` column header written by ``mic2grains``.

Mic2GrainsList.c wrote a PROSE header --

    %GrainID OrientMat(9) X Y Z LatC(6) 0 0 0 Radius Confidence

-- which names no columns, so every name-driven reader in the tree raised on
NF seed files even though the 24 data columns were already correctly aligned
with the ProcessGrains schema. These tests pin the two halves of the contract
that has to hold simultaneously:

* the header must name real columns, so name-driven readers work; and
* the file must still be exactly nine ``%`` lines with unchanged data rows, so
  the POSITIONAL consumers -- the FF indexer, which reads ``GrainsFile`` as
  one line + eight content-ignored skips + a positional ``sscanf``
  (``FF_HEDM/src/IndexerOMP.c:2358-2383`` and its port
  ``midas_index/io/csv.py``) -- see exactly what they saw before.

Nothing here needs the C binary, unlike ``test_mic2grains.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_nf_pipeline.mic2grains import (
    GRAINS_COLUMN_NAMES, _write_grains_csv,
)

#: A valid rotation matrix (not the identity, so a transposed or rotated read
#: would show up).
_OM = np.array([0.36, 0.48, -0.80,
                -0.80, 0.60, 0.00,
                0.48, 0.64, 0.60])
_LATTICE = [4.078, 4.078, 4.078, 90.0, 90.0, 90.0]


def _write(tmp_path, n=2):
    out = tmp_path / "GrainsLayer1.csv"
    unique = [(_OM, 10.0, 20.0, 100), (_OM, -5.0, 7.0, 50)][:n]
    _write_grains_csv(
        out, unique, sg_nr=225, max_angle_deg=5.0, min_conf=0.4,
        lattice_params=_LATTICE, do_neighbor_search=0, tri_edge_size=5.0,
        mic_file="foo.mic",
    )
    return out


def test_header_names_real_columns(tmp_path):
    out = _write(tmp_path)
    lines = out.read_text().splitlines()
    cols = lines[1].lstrip("%").split()
    assert cols == GRAINS_COLUMN_NAMES
    assert len(cols) == 24
    # No prose left anywhere in it.
    for junk in ("OrientMat(9)", "LatC(6)"):
        assert junk not in lines[1]
    # Header width == data width, on every row.
    for raw in lines[9:]:
        assert len(raw.split()) == len(cols)


def test_positional_contract_is_unchanged(tmp_path):
    """Nine '%' lines: one NumGrains + eight the FF indexer skips blind."""
    out = _write(tmp_path)
    lines = out.read_text().splitlines()
    pct = [ln for ln in lines if ln.startswith("%")]
    assert len(pct) == 9
    assert lines[0].startswith("%NumGrains")
    assert all(ln.startswith("%") for ln in lines[:9])
    assert not lines[9].startswith("%")


def test_ff_indexer_positional_reader_still_agrees(tmp_path):
    """The reader the FF indexer's mode A actually uses."""
    idx_csv = pytest.importorskip("midas_index.io.csv")
    out = _write(tmp_path)
    d = idx_csv.read_grains_csv(out)
    np.testing.assert_array_equal(d["ids"], [1, 2])
    np.testing.assert_allclose(d["orient_mat"][0].reshape(-1), _OM)
    np.testing.assert_allclose(d["positions"][0], [10.0, 20.0, 0.0])
    # Column 22 is GrainRadius; sqrt(nVox * triEdge^2 * sqrt(3)/4 / pi).
    expected_r = np.sqrt(100 * 25.0 * np.sqrt(3.0) / 4.0 / np.pi)
    np.testing.assert_allclose(d["radii"][0], expected_r, rtol=1e-6)


def test_canonical_name_driven_reader_accepts_it(tmp_path):
    """The whole point of the change."""
    io_read = pytest.importorskip("midas_process_grains.io")
    out = _write(tmp_path)
    t = io_read.read_grains_csv(out)
    assert t.n_grains == 2
    assert t.n_columns == 24
    assert t.header_token == "GrainID"
    np.testing.assert_allclose(t.orient_mat[0].reshape(-1), _OM)
    np.testing.assert_allclose(t.positions[0], [10.0, 20.0, 0.0])
    np.testing.assert_allclose(t.lattice[0], _LATTICE)
    np.testing.assert_allclose(t.confidence, [1.0, 1.0])
    # NF has no FF refinement, so the residual triple is a real zero, and the
    # blocks that only ProcessGrains writes must be absent -- not invented.
    np.testing.assert_allclose(t.diff_pos, 0.0)
    assert t.strain_fab is None
    assert t.strain_ken is None
    assert t.euler is None


def test_midas_stress_reader_accepts_it(tmp_path):
    """midas_stress is a hard dependency, so this one never skips.

    It also covers the second half of the NF-specific trap: the column header
    is line 2 of a nine-line preamble, i.e. six more '%' lines FOLLOW it. A
    reader that only skips N lines then hands the rest to a float parser
    chokes on '%SpaceGroup 225'.
    """
    from midas_stress.io import read_grains_csv
    out = _write(tmp_path)
    g = read_grains_csv(str(out))
    assert g["raw"].shape == (2, 24)
    np.testing.assert_array_equal(g["grain_ids"], [1, 2])
    np.testing.assert_allclose(g["orientations"][0].reshape(-1), _OM)
    np.testing.assert_allclose(g["confidences"], [1.0, 1.0])
    np.testing.assert_allclose(g["lattice_params"][0], _LATTICE)


def test_ff_to_pf_seeding_handoff_accepts_it(tmp_path):
    """midas_pipeline turns an NF seed file into UniqueOrientations.csv."""
    handoff = pytest.importorskip("midas_pipeline.seeding.handoff")
    out = _write(tmp_path)
    oms, ids = handoff._parse_grains_csv(out)
    assert oms.shape == (2, 9)
    assert ids == [1, 2]
    np.testing.assert_allclose(oms[0], _OM)


def test_trailing_tab_rows_are_still_readable(tmp_path):
    """The writer emits no trailing tab, but the C ProcessGrains does, and
    these files get concatenated and re-emitted by other tools. Mutate a copy
    into that form and check the readers survive it -- the omission of this
    case from every fixture in the tree is why the tab crash shipped.
    """
    io_read = pytest.importorskip("midas_process_grains.io")
    out = _write(tmp_path)
    lines = out.read_text().splitlines()
    tabbed = tmp_path / "GrainsLayer1_tabbed.csv"
    tabbed.write_text("\n".join(
        ln if ln.startswith("%") else "\t".join(ln.split()) + "\t"
        for ln in lines) + "\n")
    t = io_read.read_grains_csv(tabbed)
    assert t.n_grains == 2
    np.testing.assert_allclose(t.orient_mat[0].reshape(-1), _OM)

    from midas_stress.io import read_grains_csv
    g = read_grains_csv(str(tabbed))
    assert g["raw"].shape == (2, 24)

    plotting = pytest.importorskip("midas_plotting.grains")
    import warnings
    with warnings.catch_warnings():
        # Euler columns are absent here, so the OM/Euler cross-check derives
        # them and cannot disagree; silence anything else for robustness.
        warnings.simplefilter("ignore")
        gl = plotting.read_grains(tabbed)
    assert len(gl) == 2

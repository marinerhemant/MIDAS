"""Smoke tests for the CPFEM I/O stub: signatures import, NotImplementedError is raised cleanly."""

import numpy as np
import pytest

from midas_defect.cpfem import damask_io, fepx_io, prisms_io


_DUMMY = dict(
    OM=np.tile(np.eye(3)[None], (5, 1, 1)),
    pos=np.zeros((5, 3)),
    grain_radii=np.ones(5),
)


def test_damask_writer_raises_not_implemented_with_clear_message():
    with pytest.raises(NotImplementedError, match="midas_defect.cpfem stub"):
        damask_io.write_damask_initial_microstructure(**_DUMMY)


def test_damask_reader_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="midas_defect.cpfem stub"):
        damask_io.read_damask_grain_output("nonexistent.hdf5")


def test_fepx_writer_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="midas_defect.cpfem stub"):
        fepx_io.write_fepx_initial_microstructure(**_DUMMY)


def test_fepx_reader_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="midas_defect.cpfem stub"):
        fepx_io.read_fepx_grain_output("nonexistent.out")


def test_prisms_writer_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="midas_defect.cpfem stub"):
        prisms_io.write_prisms_initial_microstructure(**_DUMMY)


def test_prisms_reader_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="midas_defect.cpfem stub"):
        prisms_io.read_prisms_grain_output("nonexistent.txt")

"""Tests for :mod:`midas_dfxm.md_ingest` (Component-2 B4 MD ingestion)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_dfxm.md_ingest import (
    atomic_deformation_gradients,
    fcc_neighbor_template,
    find_reference_orientation,
    load_external_field,
    read_lammps_dump,
    refine_fcc_lattice_constant,
)


def _write_lammps_dump(path, positions, box_lengths, extra_col="c_epot"):
    n = positions.shape[0]
    lo = np.zeros(3)
    lines = [
        "ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", str(n),
        "ITEM: BOX BOUNDS",
        f"{lo[0]:.6f} {box_lengths[0]:.6f}",
        f"{lo[1]:.6f} {box_lengths[1]:.6f}",
        f"{lo[2]:.6f} {box_lengths[2]:.6f}",
        f"ITEM: ATOMS id x y z {extra_col}",
    ]
    for i in range(n):
        x, y, z = positions[i]
        lines.append(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} 0.0")
    path.write_text("\n".join(lines) + "\n")


def _random_rotation(seed=0):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.3, 1.0)  # radians, avoid near-identity
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _perfect_fcc_positions(a0, n_cells, center_shift):
    """A perfect FCC cluster (conventional-cell sites), origin-centred."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n_cells) for j in range(n_cells)
                       for k in range(n_cells)])
    pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a0
    return pos - pos.mean(axis=0) + center_shift


@pytest.fixture
def rotated_fcc_dump(tmp_path):
    """A rotated, undistorted single-crystal FCC (Cu-like) LAMMPS dump."""
    a0 = 3.6356
    R_true = _random_rotation(seed=1)
    box_lengths = np.array([60.0, 60.0, 60.0])
    center = box_lengths / 2.0
    pos = _perfect_fcc_positions(a0, n_cells=14, center_shift=np.zeros(3))
    pos = pos @ R_true.T + center
    inside = np.all((pos > 2.0) & (pos < box_lengths - 2.0), axis=1)
    pos = pos[inside]
    dump_path = tmp_path / "fcc_rotated.dump"
    _write_lammps_dump(dump_path, pos, box_lengths)
    return dump_path, a0, R_true, box_lengths, center


def test_read_lammps_dump_roundtrip(rotated_fcc_dump):
    dump_path, a0, R_true, box_lengths, center = rotated_fcc_dump
    atoms = read_lammps_dump(dump_path)
    assert atoms.positions.shape[1] == 3
    assert atoms.positions.shape[0] > 1000
    np.testing.assert_allclose(atoms.box_lengths, box_lengths, atol=1e-6)
    assert "c_epot" in atoms.columns


def test_refine_fcc_lattice_constant_recovers_planted_a0(rotated_fcc_dump):
    dump_path, a0, R_true, box_lengths, center = rotated_fcc_dump
    atoms = read_lammps_dump(dump_path)
    a0_fit = refine_fcc_lattice_constant(atoms.positions, atoms.box_lengths, a0_guess=3.6)
    assert abs(a0_fit - a0) / a0 < 1e-3


def test_find_reference_orientation_is_a_proper_rotation(rotated_fcc_dump):
    dump_path, a0, R_true, box_lengths, center = rotated_fcc_dump
    atoms = read_lammps_dump(dump_path)
    template = fcc_neighbor_template(a0)
    cutoff = 1.2 * a0 / math.sqrt(2.0)
    R0 = find_reference_orientation(
        atoms.positions, atoms.box_lengths, template,
        seed_center_angstrom=center, cutoff=cutoff)
    np.testing.assert_allclose(R0 @ R0.T, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(R0), 1.0, atol=1e-8)


def test_atomic_deformation_gradients_recovers_identity_on_perfect_crystal(rotated_fcc_dump):
    dump_path, a0, R_true, box_lengths, center = rotated_fcc_dump
    atoms = read_lammps_dump(dump_path)
    template = fcc_neighbor_template(a0)
    cutoff = 1.2 * a0 / math.sqrt(2.0)
    R0 = find_reference_orientation(
        atoms.positions, atoms.box_lengths, template,
        seed_center_angstrom=center, cutoff=cutoff)
    fitted_idx, F = atomic_deformation_gradients(
        atoms.positions, atoms.box_lengths, R0, template, cutoff=cutoff)
    # A perfect, undistorted crystal: every fit that passes the residual gate
    # must recover F = I to high precision, and most atoms should pass.
    assert len(fitted_idx) > 0.5 * atoms.positions.shape[0]
    eye = np.eye(3)
    max_dev = np.abs(F - eye[None]).max()
    assert max_dev < 5e-3


def test_load_external_field_end_to_end_on_perfect_crystal(rotated_fcc_dump):
    dump_path, a0, R_true, box_lengths, center = rotated_fcc_dump
    crop = tuple((c - 15.0, c + 15.0) for c in center)
    field = load_external_field(str(dump_path), crop_angstrom=crop, voxel_angstrom=6.0)
    assert field.n_voxels > 0
    eps = 0.5 * (field.F + field.F.transpose(-1, -2)) - torch.eye(3, dtype=field.F.dtype)
    assert float(eps.abs().max()) < 1e-2
    # positions are in micrometers and inside the requested (padded) crop
    assert float(field.positions.abs().max()) < 1.0  # crop is tens of Angstrom, i.e. << 1 um


def test_load_external_field_requires_crop(rotated_fcc_dump):
    dump_path, *_ = rotated_fcc_dump
    with pytest.raises(ValueError, match="crop_angstrom"):
        load_external_field(str(dump_path))


def test_gzip_dump_reads_identically(tmp_path, rotated_fcc_dump):
    import gzip
    import shutil

    dump_path, *_ = rotated_fcc_dump
    gz_path = tmp_path / "fcc_rotated.dump.gz"
    with open(dump_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    plain = read_lammps_dump(dump_path)
    gz = read_lammps_dump(gz_path)
    np.testing.assert_allclose(plain.positions, gz.positions)

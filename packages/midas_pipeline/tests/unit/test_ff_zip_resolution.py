"""The FF zarr must be found without the caller naming it.

Regression: the retired midas-ff-pipeline resolved the peak-fit input as
``Path(det.zarr_path)``. On the resume path that field comes back empty, and
``Path("")`` renders as ``"."``, so ``peakfit_torch`` was handed the current
directory as its DataFile and died on ``IsADirectoryError``. The zip produced by
zip_convert was sitting in the layer directory the whole time; passing --zarr
explicitly was the only way through.

Resolution must therefore fall back to the layer directory, and must never hand
a directory or an empty path downstream.
"""

from __future__ import annotations

import types

import pytest

from midas_pipeline.stages.peakfit import _resolve_ff_zip


def _ctx(layer_dir, zarr_path=None):
    """Minimal stand-in: _resolve_ff_zip reads only these two fields."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(zarr_path=zarr_path),
        layer_dir=layer_dir,
    )


def test_finds_the_zip_in_the_layer_dir_without_being_told(tmp_path):
    zip_path = tmp_path / "scan_000999.MIDAS.zip"
    zip_path.write_bytes(b"")
    assert _resolve_ff_zip(_ctx(tmp_path)) == zip_path


def test_empty_zarr_path_does_not_become_dot(tmp_path):
    """The exact failure: "" must not resolve to the cwd."""
    zip_path = tmp_path / "scan_000999.MIDAS.zip"
    zip_path.write_bytes(b"")
    resolved = _resolve_ff_zip(_ctx(tmp_path, zarr_path=""))
    assert resolved == zip_path
    assert str(resolved) != "."


def test_explicit_zarr_path_still_wins(tmp_path):
    explicit = tmp_path / "elsewhere.MIDAS.zip"
    explicit.write_bytes(b"")
    (tmp_path / "insitu_000001.MIDAS.zip").write_bytes(b"")
    assert _resolve_ff_zip(_ctx(tmp_path, zarr_path=str(explicit))) == explicit


def test_missing_zip_reports_nothing_rather_than_a_directory(tmp_path):
    """No zip is a real condition; answering with the cwd is not."""
    resolved = _resolve_ff_zip(_ctx(tmp_path))
    assert resolved is None or not resolved.is_dir()

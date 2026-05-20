"""Smoke + pip-portability tests for midas_zipper.

These guard the package's core promise: it imports and runs with no MIDAS
source tree and no compiled C dependency. They deliberately avoid touching
real detector data (none is shipped) — they exercise import surface, the
public API, the vendored CBF reader, and the absence of source-tree imports.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

PKG_DIR = Path(__file__).resolve().parents[1] / "midas_zipper"


def test_package_imports_and_version():
    import midas_zipper

    assert isinstance(midas_zipper.__version__, str)
    assert midas_zipper.__version__


def test_public_api_present():
    import midas_zipper

    assert hasattr(midas_zipper, "generate_ff_zip")
    assert callable(midas_zipper.generate_ff_zip)


def test_ff_zip_module_imports():
    mod = importlib.import_module("midas_zipper.ff_zip")
    assert hasattr(mod, "main") and callable(mod.main)


def test_update_zarr_imports_without_executing():
    # update_zarr must NOT run argparse/work at import time (wrapped in main()).
    mod = importlib.import_module("midas_zipper.update_zarr")
    assert hasattr(mod, "main") and callable(mod.main)


def test_vendored_cbf_reader_api():
    mod = importlib.import_module("midas_zipper._read_cbf")
    assert callable(getattr(mod, "read_cbf", None))
    assert callable(getattr(mod, "read_cbf_metadata", None))
    assert isinstance(getattr(mod, "DATA_TYPES", None), dict)


@pytest.mark.parametrize("modname", ["ff_zip", "update_zarr", "_read_cbf", "__init__"])
def test_no_source_tree_or_c_imports(modname):
    """No module reaches into the MIDAS source tree or a C binary/lib."""
    fname = "__init__.py" if modname == "__init__" else f"{modname}.py"
    src = (PKG_DIR / fname).read_text()
    forbidden = [
        "import midas_config",
        "from read_cbf import",          # must use the vendored ._read_cbf
        "from version import",
        "/Users/",                        # no hardcoded personal paths
        "FF_HEDM/",
        "ctypes",
    ]
    offenders = [f for f in forbidden if f in src]
    assert not offenders, f"{fname} has source-tree/C references: {offenders}"


def test_generate_ff_zip_signature():
    from midas_zipper import generate_ff_zip

    sig = inspect.signature(generate_ff_zip)
    for kw in ("result_folder", "param_file", "layer_nr"):
        assert kw in sig.parameters, f"missing kwarg {kw}"

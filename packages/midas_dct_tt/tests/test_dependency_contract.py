"""The declared dependency floors must cover the API this package actually calls.

A floor that is too low does not fail here, in a dev tree where the newest
version is installed. It fails on a stranger's first `pip install`, with a
TypeError, on a call path our whole test suite exercises. That is the worst
failure mode we can ship, so it gets its own test.

This caught a real one: `recon.py` passes ``lr_schedule=`` to
``midas_invert.fit``, a keyword introduced in midas-invert 0.1.1, while
``pyproject.toml`` still floored the dependency at ``>=0.1``.
"""
import inspect
import pathlib
import re

import pytest

PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"


def declared_floor(distribution):
    """Lower bound this package declares for `distribution`, e.g. '0.1.1'."""
    text = PYPROJECT.read_text()
    m = re.search(rf'"{re.escape(distribution)}\s*>=\s*([0-9][^",\s]*)"', text)
    if m is None:
        pytest.fail(f"{distribution} has no >= floor in pyproject.toml")
    return tuple(int(p) for p in m.group(1).split(".") if p.isdigit())


def test_midas_invert_floor_covers_the_keywords_we_pass():
    """`fit(lr_schedule=..., return_best=...)` needs midas-invert >= 0.1.1."""
    from midas_invert import fit
    params = inspect.signature(fit).parameters
    for kw in ("lr_schedule", "return_best"):
        assert kw in params, (
            f"the installed midas_invert.fit has no {kw!r}; either the "
            f"installed version predates 0.1.1 or the API changed")
    assert declared_floor("midas-invert") >= (0, 1, 1)


def test_recon_actually_passes_those_keywords():
    """Guards the premise: if recon.py stops using them, the floor can relax."""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "midas_dct_tt" / "recon.py").read_text()
    assert "lr_schedule=" in src


def test_midas_hkls_floor_covers_the_lattice_api_we_call():
    """`goniometer.reciprocal_basis` needs Lattice.reciprocal_cartesian_vectors,
    added in midas-hkls 0.7.2 (commit 1ac5258d). Resolving 0.6.x is an
    AttributeError on the first call."""
    from midas_hkls.lattice import Lattice
    assert hasattr(Lattice(1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
                   "reciprocal_cartesian_vectors")
    assert declared_floor("midas-hkls") >= (0, 7, 2)


def test_every_midas_dependency_declares_a_floor():
    """An unpinned midas-* dependency resolves to whatever PyPI offers."""
    block = re.search(r"^dependencies = \[(.*?)^\]", PYPROJECT.read_text(),
                      re.S | re.M)
    assert block is not None
    for name in re.findall(r'"(midas-[a-z0-9-]+)', block.group(1)):
        assert declared_floor(name), name

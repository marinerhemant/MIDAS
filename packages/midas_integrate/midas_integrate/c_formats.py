"""The two output formats the C integrators produced, re-exported.

``midas_integrate`` historically *read* these (see :mod:`.exporters`, which
dumps a MIDAS ``.zarr.zip`` to CSV) but could never write one. The writers
live in :mod:`midas_integrate_v2.io` — one implementation, not two — and this
module is the import path for callers already working in the v1 package:

``write_gsas_zarr_zip``
    The ``.zarr.zip`` that ``IntegratorZarrOMP`` + ``integrator.py`` wrote,
    readable by GSAS-II's "MIDAS zarr" reader and by ``midas_zipper``.
``write_stacked_h5``
    The consolidated-array HDF5 that ``integrator_stream_process_h5_stacked.py``
    built from ``IntegratorFitPeaksGPUStream``'s binaries.

The import is deferred rather than declared: ``midas_integrate_v2`` depends on
this package, so a dependency the other way would be a cycle. Both writers
take a :class:`midas_integrate_v2.spec.IntegrationSpec` for the geometry.
"""
from __future__ import annotations

__all__ = ["write_gsas_zarr_zip", "write_stacked_h5",
           "GSASZarrWriter", "StackedH5Writer", "reta_map"]

_MISSING = (
    "{name} lives in midas_integrate_v2, which is not installed. "
    "pip install 'midas-integrate-v2' (add the [zarr] extra for the "
    "zarr writer). It is imported lazily because midas_integrate_v2 "
    "depends on this package; declaring it here would be a cycle."
)

#: Appended when midas_integrate_v2 IS installed but predates the writers.
_TOO_OLD = (" It is installed but does not provide this writer, so it predates "
            "midas-integrate-v2 0.6.0 -- upgrade it.")


def _v2_io(name: str):
    try:
        from midas_integrate_v2 import io
    except ImportError as e:                       # pragma: no cover
        raise ImportError(_MISSING.format(name=name)) from e
    try:
        return getattr(io, name)
    except AttributeError as e:
        # An INSTALLED-BUT-OLD midas_integrate_v2 imports fine and simply lacks
        # the writer, so the ImportError branch above never fires and the caller
        # gets a bare AttributeError instead of the message explaining what to
        # install. That is the likely case for anyone who upgraded this package
        # alone, which is exactly who needs the message.
        raise ImportError(_MISSING.format(name=name) + _TOO_OLD) from e


def write_gsas_zarr_zip(*args, **kwargs):
    """See :func:`midas_integrate_v2.io.write_gsas_zarr_zip`."""
    return _v2_io("write_gsas_zarr_zip")(*args, **kwargs)


def write_stacked_h5(*args, **kwargs):
    """See :func:`midas_integrate_v2.io.write_stacked_h5`."""
    return _v2_io("write_stacked_h5")(*args, **kwargs)


def reta_map(*args, **kwargs):
    """See :func:`midas_integrate_v2.io.reta_map`."""
    return _v2_io("reta_map")(*args, **kwargs)


def GSASZarrWriter(*args, **kwargs):
    """See :class:`midas_integrate_v2.io.GSASZarrWriter`."""
    return _v2_io("GSASZarrWriter")(*args, **kwargs)


def StackedH5Writer(*args, **kwargs):
    """See :class:`midas_integrate_v2.io.StackedH5Writer`."""
    return _v2_io("StackedH5Writer")(*args, **kwargs)

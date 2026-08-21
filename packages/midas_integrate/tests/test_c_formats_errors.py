"""c_formats must explain itself when midas_integrate_v2 is absent OR too old.

The writers live in midas_integrate_v2 and are imported lazily, because
midas_integrate_v2 depends on THIS package and declaring it here would be a
cycle.  The lazy import has two failure modes and only one of them used to be
handled:

  * not installed        -> ImportError from the import  (handled)
  * installed but OLD    -> the module imports and getattr raises
                            AttributeError, so the helpful message never fired

The second is the likely one: it is what anyone who upgraded midas-integrate
alone will hit.
"""
import sys
import types

import pytest


def _install_fake_v2(monkeypatch, *, with_writer: bool):
    fake = types.ModuleType("midas_integrate_v2")
    io = types.ModuleType("midas_integrate_v2.io")
    if with_writer:
        io.write_gsas_zarr_zip = lambda *a, **k: "wrote"
    fake.io = io
    monkeypatch.setitem(sys.modules, "midas_integrate_v2", fake)
    monkeypatch.setitem(sys.modules, "midas_integrate_v2.io", io)


def test_absent_v2_raises_importerror_naming_the_package(monkeypatch):
    from midas_integrate import c_formats
    monkeypatch.setitem(sys.modules, "midas_integrate_v2", None)
    with pytest.raises(ImportError) as e:
        c_formats.write_gsas_zarr_zip()
    assert "midas_integrate_v2" in str(e.value)


def test_old_v2_raises_importerror_not_attributeerror(monkeypatch):
    from midas_integrate import c_formats
    _install_fake_v2(monkeypatch, with_writer=False)
    with pytest.raises(ImportError) as e:            # NOT AttributeError
        c_formats.write_gsas_zarr_zip()
    msg = str(e.value)
    assert "0.6.0" in msg, "the message must say which version provides it"
    assert "upgrade" in msg.lower()


def test_current_v2_is_called_through(monkeypatch):
    from midas_integrate import c_formats
    _install_fake_v2(monkeypatch, with_writer=True)
    assert c_formats.write_gsas_zarr_zip() == "wrote"

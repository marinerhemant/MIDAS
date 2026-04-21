"""Smoke test: package imports and advertises the right version."""

from importlib.metadata import version


def test_import_version():
    import midas_integrate

    assert midas_integrate.__version__ == version("midas-integrate")


def test_paths_api_surface():
    from midas_integrate import MidasBinaryNotFoundError, midas_bin

    assert callable(midas_bin)
    assert issubclass(MidasBinaryNotFoundError, RuntimeError)


def test_interop_with_auto_calibrate():
    """midas-integrate's hard dep on midas-auto-calibrate must be live."""
    import midas_auto_calibrate

    assert hasattr(midas_auto_calibrate, "DetectorGeometry")
    assert hasattr(midas_auto_calibrate, "midas_bin")

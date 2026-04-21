"""Smoke test: package imports and advertises the right version."""

from importlib.metadata import version


def test_import_version():
    import midas_auto_calibrate

    assert midas_auto_calibrate.__version__ == version("midas-auto-calibrate")


def test_paths_api_surface():
    from midas_auto_calibrate import MidasBinaryNotFoundError, midas_bin

    assert callable(midas_bin)
    assert issubclass(MidasBinaryNotFoundError, RuntimeError)

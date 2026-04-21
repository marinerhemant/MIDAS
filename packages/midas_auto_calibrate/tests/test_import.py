"""Smoke test: package imports and advertises the right version."""


def test_import_version():
    import midas_auto_calibrate

    assert midas_auto_calibrate.__version__ == "0.1.0"


def test_paths_api_surface():
    from midas_auto_calibrate import MidasBinaryNotFoundError, midas_bin

    assert callable(midas_bin)
    assert issubclass(MidasBinaryNotFoundError, RuntimeError)

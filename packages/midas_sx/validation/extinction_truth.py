"""Shim: the extinction truth model lives in tests/support/extinction_truth.py (the unit tests import it). The validation scripts keep `import extinction_truth`."""
import importlib.util
import pathlib
import sys

_path = pathlib.Path(__file__).resolve().parents[1] / "tests" / "support" / "extinction_truth.py"
_spec = importlib.util.spec_from_file_location("_extinction_truth_impl", _path)
_impl = importlib.util.module_from_spec(_spec)
sys.modules["_extinction_truth_impl"] = _impl                                 # dataclasses look the module up by name
_spec.loader.exec_module(_impl)
globals().update({k: v for k, v in vars(_impl).items() if not (k.startswith("__") and k.endswith("__"))})

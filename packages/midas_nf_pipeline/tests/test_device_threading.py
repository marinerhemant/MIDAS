"""`--device cpu` must reach EVERY stage, not just the fit.

run_diffr_spots and run_image_processing used to build their Namespace with a
hardcoded ``device=None``, so the callee auto-detected and took CUDA even when
the user explicitly asked for CPU. On a shared GPU that surfaced as

    memory allocation failed with OOM on device 0 while trying to allocate
    3384803328 bytes (free: 1924792320, total: 101971591168)
    RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() ... ASSERT FAILED

from a stage the user had deliberately kept off the GPU. run_fitting always
took a device argument, which is why the bug looked like "only the fit respects
--device".
"""
from argparse import Namespace
from types import SimpleNamespace

import pytest

from midas_nf_pipeline import stages


@pytest.fixture
def captured(monkeypatch):
    """Capture the Namespace each stage hands to its callee."""
    seen = {}

    def fake_diffr(args):
        seen["diffr"] = args

    def fake_proc(args):
        seen["proc"] = args

    monkeypatch.setattr("midas_nf_preprocess.diffr_spots.cli.run", fake_diffr)
    monkeypatch.setattr("midas_nf_preprocess.process_images.cli.run", fake_proc)
    return seen


def test_diffr_spots_forwards_the_device(tmp_path, captured):
    (tmp_path / "hkls.csv").write_text("dummy\n")
    p = {"resultFolder": str(tmp_path), "SeedOrientations": ""}
    stages.run_diffr_spots(p, tmp_path / "params.txt", device="cpu", dtype="float64")
    assert captured["diffr"].device == "cpu", (
        "run_diffr_spots dropped the device and the callee will auto-detect CUDA")
    assert captured["diffr"].dtype == "float64"


def test_image_processing_forwards_the_device(tmp_path, captured):
    p = {"nDistances": 1, "nCPUs": 2}
    stages.run_image_processing(p, tmp_path / "params.txt", device="cpu")
    assert captured["proc"].device == "cpu", (
        "run_image_processing dropped the device")


def test_device_defaults_to_none_when_not_given(tmp_path, captured):
    """Omitting it keeps the historical auto-detect, so nothing else changes."""
    (tmp_path / "hkls.csv").write_text("dummy\n")
    stages.run_diffr_spots({"resultFolder": str(tmp_path), "SeedOrientations": ""},
                           tmp_path / "params.txt")
    assert captured["diffr"].device is None


def test_workflows_passes_device_to_every_preprocessing_call():
    """Guard the wiring itself: every call site must forward, not just the first.

    Two of the three run_preprocessing call sites were missed on the first fix
    because their indentation differed, and nothing would have caught it.
    """
    import inspect
    from midas_nf_pipeline import workflows

    src = inspect.getsource(workflows)

    def arglist(block):
        """Text of one call's arguments, up to its MATCHING close paren.

        Splitting on the first ')' is wrong -- a nested call such as
        ``bool(int(getattr(args, "ffSeedOrientations", 0)))`` closes first and
        truncates the argument list before ``device=`` is reached.
        """
        depth, out = 1, []
        for ch in block:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            out.append(ch)
        return "".join(out)

    for name in ("run_preprocessing", "run_image_processing"):
        blocks = src.split(f"stages.{name}(")[1:]
        with_device = sum(1 for b in blocks if "device=" in arglist(b))
        assert with_device == len(blocks), (
            f"{with_device}/{len(blocks)} {name} call sites forward a device; "
            f"the rest will silently auto-detect CUDA")

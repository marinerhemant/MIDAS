"""End-to-end test: run MIDASCalibrant on the wheel-bundled CeO2 calibrant.

Uses the Pilatus CeO2 frame shipped inside the wheel
(``midas_auto_calibrate._data/CeO2_Pilatus.tif``). Skipped when the C
binaries aren't reachable (dev workflow before ``pip install``); runs
unconditionally once the wheel is installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from midas_auto_calibrate import (
    MidasBinaryNotFoundError,
    midas_bin,
    data as mac_data,
)
from midas_auto_calibrate.calibrate import _parse_final_geometry
from midas_auto_calibrate import CalibrationConfig


def _binary_available() -> bool:
    try:
        midas_bin("MIDASCalibrant")
        midas_bin("GetHKLList")
        return True
    except MidasBinaryNotFoundError:
        return False


# Module-level: data must exist for any test here. Binary-skip is per-test
# (test_bundled_data_paths_resolve verifies data packaging, doesn't need the
# C binary).
pytestmark = pytest.mark.skipif(
    not mac_data.CEO2_PILATUS.exists(),
    reason="Bundled CeO2 calibrant data missing from wheel — "
           "rebuild the wheel (pip install .) so _data/*.tif gets shipped.",
)

needs_binary = pytest.mark.skipif(
    not _binary_available(),
    reason="MIDASCalibrant + GetHKLList not discoverable — set MIDAS_BIN "
           "to an installed _bin/, or run tests outside the package dir.",
)


def _stage_bundle(tmp_path: Path) -> Path:
    """Copy bundled Pilatus data into a writable workdir.

    The C binary writes .corr.csv / .convergence_history.csv / checkpoint
    files next to the input image, so we never run against the wheel's
    read-only installed tree.
    """
    workdir = tmp_path / "calib"
    workdir.mkdir()
    # Match the filenames the reference Parameters.txt expects.
    shutil.copy(mac_data.CEO2_PILATUS,
                workdir / "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif")
    shutil.copy(mac_data.CEO2_PILATUS_DARK,
                workdir / "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif")
    shutil.copy(mac_data.CEO2_PILATUS_MASK, workdir / "mask_upd.tif")
    # Reference parameter file — prepend runtime paths.
    params_text = mac_data.PARAMETERS_TXT.read_text()
    params_text = (
        f"Dark dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif\n"
        f"Folder .\n"
        f"FileStem CeO2_Pil_100x100_att000_650mm_71p676keV\n"
        f"MaskFile mask_upd.tif\n\n"
    ) + params_text
    (workdir / "parameters.txt").write_text(params_text)
    return workdir


@needs_binary
def test_ceo2_pilatus_end_to_end(tmp_path):
    """MIDASCalibrant converges on the bundled Pilatus CeO2 frame."""
    workdir = _stage_bundle(tmp_path)

    exe = midas_bin("MIDASCalibrant")
    result = subprocess.run(
        [str(exe), "parameters.txt", "4"],
        capture_output=True, text=True, cwd=workdir, timeout=300,
    )
    assert result.returncode == 0, (
        f"MIDASCalibrant exited {result.returncode}:\n"
        f"{result.stderr[-400:]}\n"
        f"---\n{result.stdout[-400:]}"
    )

    cfg = CalibrationConfig(
        wavelength=0.172973, pixel_size=172.0,
        lsd=657_436.9, ybc=685.5, zbc=921.0,
        nr_pixels_y=1475, nr_pixels_z=1679,
    )
    geom = _parse_final_geometry(result.stdout, cfg)

    # Loose bounds that still catch algorithmic regressions.
    assert 600_000 < geom.lsd < 700_000, f"Lsd drifted: {geom.lsd}"
    assert 600 < geom.ybc < 800, f"ybc drifted: {geom.ybc}"
    assert 800 < geom.zbc < 1000, f"zbc drifted: {geom.zbc}"
    assert 0 < geom.mean_strain < 100, f"MeanStrain drifted: {geom.mean_strain}"

    # Expected side-effects landed next to the input.
    assert any(workdir.glob("*.convergence_history.csv")), (
        "no convergence_history.csv produced")
    assert any(workdir.glob("*.corr.csv")), "no corr.csv produced"


def test_bundled_data_paths_resolve():
    """`mac.data.CEO2_PILATUS` etc. are real files with plausible sizes."""
    for path, min_bytes in [
        (mac_data.CEO2_PILATUS, 1_000_000),
        (mac_data.CEO2_PILATUS_DARK, 1_000_000),
        (mac_data.CEO2_PILATUS_MASK, 500_000),
        (mac_data.PARAMETERS_TXT, 100),
    ]:
        assert path.exists(), f"bundled file missing: {path}"
        assert path.stat().st_size > min_bytes, (
            f"bundled file suspiciously small: {path}")

"""End-to-end CPU integration against the bundled CeO2 Pilatus frame.

Exercises the full pipeline: Mapper → make_zarr_zip → Integrator → cake.
Skipped when the shipped binaries aren't discoverable (dev workflow
without ``MIDAS_BIN`` set) or when midas-auto-calibrate's bundled data
isn't available.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from midas_auto_calibrate import data as mac_data
from midas_integrate import (
    IntegrationConfig,
    Integrator,
    Mapper,
    MidasBinaryNotFoundError,
    make_zarr_zip,
    midas_bin,
)


def _binaries_available() -> bool:
    try:
        midas_bin("MIDASDetectorMapper")
        midas_bin("MIDASIntegrator")
        return True
    except MidasBinaryNotFoundError:
        return False


pytestmark = [
    pytest.mark.skipif(
        not mac_data.CEO2_PILATUS.exists(),
        reason="Bundled CeO2 calibrant missing — pip reinstall "
               "midas-auto-calibrate so _data/*.tif gets shipped.",
    ),
    pytest.mark.skipif(
        not _binaries_available(),
        reason="MIDASDetectorMapper / MIDASIntegrator not discoverable. "
               "Set MIDAS_BIN or run pytest from outside the package dir.",
    ),
]


def test_full_pipeline_on_bundled_ceo2(tmp_path):
    """Mapper → zarr.zip → Integrator → 2300×180 cake with plausible intensity."""
    workdir = tmp_path / "integ"
    workdir.mkdir()
    img = workdir / "CeO2_00001.tif"
    shutil.copy(mac_data.CEO2_PILATUS, img)

    cfg = IntegrationConfig(
        lsd=657_436.9, ybc=685.5, zbc=921.0,
        wavelength=0.172973, pixel_size=172.0,
        nr_pixels_y=1475, nr_pixels_z=1679,
        r_bin_size=0.5, eta_bin_size=2.0,
        r_min=50, r_max=1200,
    )

    # 1. Detector mapping.
    artefacts = Mapper(cfg).build(workdir, n_cpus=2)
    assert artefacts.map_bin.exists()
    assert artefacts.n_map_bin.exists()
    assert artefacts.map_bin.stat().st_size > 1_000_000

    # 2. Pack TIFF → zarr.zip.
    zarr_zip = workdir / "CeO2.zarr.zip"
    make_zarr_zip(img, cfg, zarr_zip)
    assert zarr_zip.exists()
    assert zarr_zip.stat().st_size > 100_000

    # 3. Run integrator.
    result = Integrator(cfg, artefacts).integrate(zarr_zip, n_cpus=2)
    assert result.cake_path.exists()
    assert result.backend == "cpu"

    # 4. Cake shape + non-trivial content.
    cake = result.load_cake()
    assert cake["I"].shape == (2300, 180), \
        f"unexpected cake shape: {cake['I'].shape}"
    assert cake["I"].max() > 100, \
        f"cake intensity suspiciously low: max={cake['I'].max():.2f}"
    assert len(cake["R"]) == 2300
    assert len(cake["Eta"]) == 180

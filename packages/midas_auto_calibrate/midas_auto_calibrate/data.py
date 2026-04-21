"""Paths to bundled calibrant datasets.

A single CeO₂ Pilatus pair (image + dark + mask, ~21 MB total) ships inside
the wheel so ``pip install midas-auto-calibrate && pytest`` works without
any external data. The four-detector set used for Paper A's Table 4 is
hosted on Zenodo — reach for that via :func:`zenodo_url` when you need it.

The actual bytes live in :mod:`midas_auto_calibrate._data`. This module is
the stable, documented entry point; the underscore-prefixed location
keeps it out of tab-completion for casual users.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = [
    "CEO2_PILATUS",
    "CEO2_PILATUS_DARK",
    "CEO2_PILATUS_MASK",
    "PARAMETERS_TXT",
    "zenodo_url",
]


def _data_file(name: str) -> Path:
    ref = resources.files("midas_auto_calibrate").joinpath("_data", name)
    return Path(str(ref))


CEO2_PILATUS: Path = _data_file("CeO2_Pilatus.tif")
"""Pilatus-detector CeO₂ calibrant frame (1475 × 1679 px, 71.676 keV, 650 mm)."""

CEO2_PILATUS_DARK: Path = _data_file("dark_Pilatus.tif")
"""Dark frame matched to :data:`CEO2_PILATUS`."""

CEO2_PILATUS_MASK: Path = _data_file("mask_Pilatus.tif")
"""Hot-pixel / gap mask for the Pilatus detector."""

PARAMETERS_TXT: Path = _data_file("parameters_pilatus.txt")
"""Reference Parameters.txt for the bundled Pilatus data."""


def zenodo_url() -> str:
    """Return the Zenodo URL for the full Paper A calibration dataset.

    The four-detector dataset (Pilatus 6M, Varex 4343CT, CeO₂ 10 s, Ceria)
    is ~60 MB — too large for a pip wheel. Deposited on Zenodo with a
    permanent DOI, fetched lazily by ``midas-calib-fig-stage`` when
    regenerating Paper A's multi-detector figures.

    Placeholder until the Zenodo record is minted on paper submission;
    the deposition plan matches the schedule in the release plan.
    """
    return "https://zenodo.org/record/0000000"  # TODO: replace on DOI mint

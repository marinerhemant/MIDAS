"""I/O helpers — CSV readers/writers with bit-stable formatting and binary
readers/writers for ``Spots.bin`` / ``ExtraInfo.bin`` / ``Data.bin`` / ``nData.bin``,
plus header-driven readers for the ProcessGrains ``Grains.csv`` /
``SpotMatrix.csv`` artefacts."""

from . import binary, csv, grains_csv, zarr_io  # noqa: F401
from .grains_csv import (  # noqa: F401
    GrainsFormatError,
    read_grains_columns,
    read_spot_matrix_columns,
)

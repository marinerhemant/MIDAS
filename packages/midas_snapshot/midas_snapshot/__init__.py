"""midas_snapshot: still-frame (no rotation) diffraction time series.

Pipeline: per-window pass (matrix scale, halo, calibrated sparse-spot detection)
-> windows from traces -> matrix classification -> features with a raw-photon
before/after test -> candidate-cell test with nulls and negative controls ->
image-level injection for the detection limit -> report.

The procedure and its measured limits: ``manuals/snapshot-xrd/``.
"""
__version__ = "0.1.0"

from .config import SnapshotConfig  # noqa: F401

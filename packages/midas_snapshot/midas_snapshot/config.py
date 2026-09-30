"""Run configuration for a still-frame series (JSON on disk).

Everything that decides a result is here, so a run can be repeated exactly and
reported with its settings.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Union, List, Optional


@dataclass
class SnapshotConfig:
    # data
    frames: str                          # directory of per-frame TIFFs, or a glob
    geometry: str                        # MIDAS parameter file (Lsd, BC, tilts, px, Wavelength, ...)
    out: str                             # output directory
    flip: Optional[str] = "ud"           # raw -> geometry frame: "ud", "lr", "udlr" or None
    mask: Optional[str] = None           # raw-orientation mask TIFF, nonzero = bad
    invalid_below: float = 0.0           # raw values below this are invalid (detector gap sentinels)

    # matrix (the dominant known phase): a CIF, used for its d-lines only
    matrix_cif: Optional[str] = None
    phase_cifs: List[str] = field(default_factory=list)   # further KNOWN phases (e.g. from `select`)
    matrix_fit_lines: int = 8            # strongest-lowest lines used for the per-frame scale fit
    scale_grid: List[float] = field(default_factory=lambda: [0.98, 1.05, 0.0002])
    spread_tol: float = 0.003            # max per-ring scale MAD for a window's fit to count
                                         # (hot blocks with split rings: <= 0.0027; wrong patterns: >= 0.0045)
    fit_window_deg: Optional[float] = None   # ring-centroid half-window; None = max(0.06, 6 pixels in 2theta)

    # profile / traces
    tth_step: float = 0.005
    halo_band: Optional[List[float]] = None   # [lo, hi] deg, diffuse band (None: no halo trace)
    base_band: Optional[List[float]] = None   # [lo, hi] deg, quiet band

    # detector (see midas_peakfit.snapshot_detect)
    sigma_px: Union[float, str] = "auto"   # matched-filter width (px); "auto" = measured on bright spots
    threshold_mode: str = "per_window"   # per_window: each window its own Poisson-null threshold;
                                         # table: interpolate by background level; single: one window
    n_null_window: int = 20              # null images per window (per_window mode)
    n_calib_windows: int = 9             # threshold table: this many windows spread over the series
    local_box: int = 31
    statistic: str = "gauss"
    fa_per_image: float = 0.05
    margin_px: int = 21
    tth_max: Optional[float] = None      # ignore spots beyond this angle (low-count corners)

    # windows (see midas_integrate_v2.streaming.windows_from_trace)
    window_sizes: List[int] = field(default_factory=lambda: [1, 25])
    rise: Optional[float] = None         # None: 6 x robust scatter of the baseline trace
    guard: int = 50
    after_len: int = 600
    baseline_until: int = 200            # frames used for the trace baseline
    min_before: int = 200                # shortest acceptable before window (frames)
    before: Optional[List[int]] = None   # explicit override [first, last]
    after: Optional[List[int]] = None

    # classification / features
    nsig_off: float = 6.0                # off-matrix if |rel| > nsig_off * sigma_matrix
    core_rel: float = 0.005              # matrix core used to measure sigma_matrix
    merge_px: float = 3.0
    min_det: int = 3

    nproc: int = 8
    seed: int = 20260923

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "SnapshotConfig":
        with open(path) as f:
            return cls(**json.load(f))

"""Per-worker process state + per-frame work function for the multi-process
producer pool.

The producer side of the pipeline (decompress + dark/flood/threshold +
connected-components + seed) is CPU-bound and benefits from real
parallelism via processes when the GIL or numpy-internal locks limit
threads. Each worker process holds:

  - the Zarr archive path (handles are opened lazily, per-worker)
  - the parsed ``ZarrParams`` (with dark / flood / mask / residual map)
  - the precomputed ``goodCoords`` and ``panels``

These are loaded once via ``init_worker`` (the ProcessPool initializer)
and then used by ``process_frame_in_worker`` for every frame the
worker is given.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import zarr

from midas_peakfit.connected import find_regions, filter_regions_by_size
from midas_peakfit.preprocess import preprocess_frame
from midas_peakfit.seeds import SeededRegion, seed_region
from midas_peakfit.zarr_io import frame_omega

# Workers each cache the Zarr handle once. Per-worker decompression
# overlaps with CC + seed and beats a bulk-read in main on this hardware
# (sequential bulk decompresses ~3.6 GB/s; 64 workers in parallel beat that).
_state: dict = {}


def init_worker(
    zarr_path: str,
    params_pickle: bytes,
    dark: np.ndarray,
    flood: np.ndarray,
    mask: np.ndarray,
    good_coords: np.ndarray,
    panels_pickle: bytes,
    compute_moments: bool = False,
    bg_bins: object = None,
) -> None:
    """ProcessPoolExecutor initializer. Runs once per worker process.

    Opens the Zarr archive ONCE per worker (vs. once per frame originally)
    and caches the data array handle. Per-frame reads decompress just one
    chunk — no zip-footer reparse cost.
    """
    import pickle

    p = pickle.loads(params_pickle)
    panels = pickle.loads(panels_pickle)

    p.dark = dark
    p.flood = flood
    p.mask = mask

    store = zarr.ZipStore(zarr_path, mode="r")
    root = zarr.open_group(store=store, mode="r")
    data = root["exchange/data"]

    _state.clear()
    _state.update(
        zarr_path=zarr_path,
        params=p,
        dark=dark,
        flood=flood,
        mask=mask,
        good_coords=good_coords,
        panels=panels,
        store=store,
        data=data,
        compute_moments=compute_moments,
        # Built once in the parent (full-detector distortion evaluation) and
        # shipped to each worker rather than recomputed per process.
        bg_bins=bg_bins,
    )


def process_frame_in_worker(local_idx: int) -> Tuple[int, float, int, List[SeededRegion]]:
    """Process one frame inside a worker process.

    ``local_idx`` is the position within the block's frame range (already
    adjusted for skipFrame in the main process). Returns
    (local_idx, _omega_placeholder, n_regions_total, seeded_list); the
    caller computes the real omega from the absolute frame index.
    """
    p = _state["params"]
    dark = _state["dark"]
    flood = _state["flood"]
    mask = _state["mask"]
    good_coords = _state["good_coords"]
    panels = _state["panels"]
    data = _state["data"]
    compute_moments = _state.get("compute_moments", False)
    bg_bins = _state.get("bg_bins")

    try:
        raw = np.asarray(data[local_idx], dtype=np.float64)
    except Exception:
        return local_idx, 0.0, 0, []

    img_corr = preprocess_frame(
        raw,
        NrPixels=p.NrPixels,
        NrPixelsY=p.NrPixelsY,
        NrPixelsZ=p.NrPixelsZ,
        transform_options=p.TransOpt,
        dark=dark,
        flood=flood,
        good_coords=good_coords,
        bc=p.bc,
        bad_px_intensity=p.BadPxIntensity,
        make_map=p.makeMap,
        bg_bins=bg_bins,
    )
    regions_all = find_regions(img_corr, good_coords)
    regions = filter_regions_by_size(regions_all, p.minNrPx, p.maxNrPx)
    seeded_list: List[SeededRegion] = []
    for reg in regions:
        sr = seed_region(
            reg, img_corr, mask,
            Ycen=p.Ycen, Zcen=p.Zcen,
            int_sat=p.IntSat, max_n_peaks=p.maxNPeaks,
            panels=panels,
            compute_moments=compute_moments,
        )
        if sr is not None:
            seeded_list.append(sr)
    return local_idx, 0.0, len(regions_all), seeded_list


__all__ = ["init_worker", "process_frame_in_worker"]

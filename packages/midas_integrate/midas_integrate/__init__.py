"""midas-integrate — pure-Python radial integration for area X-ray detectors.

Public API surface (re-exports from submodules so users can do
`from midas_integrate import build_csr` etc.).
"""
from __future__ import annotations

__version__ = "0.4.1"

from midas_integrate.bin_io import (
    PXLIST_DTYPE,
    MAP_HEADER_MAGIC,
    MAP_HEADER_SIZE,
    MapHeader,
    PixelMap,
    load_map,
    write_map,
    write_synthetic_map,
)
from midas_integrate.params import IntegrationParams, parse_params
from midas_integrate.geometry import (
    DEG2RAD,
    RAD2DEG,
    build_tilt_matrix,
    pixel_to_REta,
    REta_to_YZ,
    build_bin_edges,
    build_q_bin_edges_in_R,
    build_tth_bin_edges_in_R,
    polygon_area,
    circle_seg_intersect,
    ray_seg_intersect,
    point_in_quad,
    invert_REta_to_pixel,
)
from midas_integrate.detector_mapper import build_map, BuildMapResult
from midas_integrate.kernels import (
    AREA_THRESHOLD,
    CSRGeometry,
    build_csr,
    integrate,
    integrate_with_variance,
    profile_1d,
    profile_1d_with_variance,
)
from midas_integrate.image import (
    bytes_per_pixel,
    decode_payload,
    decode_hybrid_payload,
    NUMPY_DTYPE_FOR_CODE,
    DTYPE_CODE_UINT8,
    DTYPE_CODE_UINT16,
    DTYPE_CODE_UINT32,
    DTYPE_CODE_INT64,
    DTYPE_CODE_FLOAT32,
    DTYPE_CODE_FLOAT64,
    DTYPE_CODE_HYBRID,
)
from midas_integrate.peakfit import (
    fit_peaks,
    fit_peaks_autodetect,
    pseudo_voigt,
    snip_background,
    PF_PARAMS_PER_PEAK,
)
from midas_integrate.fused_csr import build_fused_geometry
from midas_integrate.postprocess import gauss_smooth_eta, median_filter_eta
from midas_integrate.exporters import export as zarr_to_csv
from midas_integrate.compat.from_v2 import (
    params_from_v2_unpacked,
    params_from_calibration_spec,
)

__all__ = [
    "__version__",
    # bin_io
    "PXLIST_DTYPE",
    "MAP_HEADER_MAGIC",
    "MAP_HEADER_SIZE",
    "MapHeader",
    "PixelMap",
    "load_map",
    "write_map",
    "write_synthetic_map",
    # params
    "IntegrationParams",
    "parse_params",
    # geometry
    "DEG2RAD",
    "RAD2DEG",
    "build_tilt_matrix",
    "pixel_to_REta",
    "REta_to_YZ",
    "build_bin_edges",
    "build_q_bin_edges_in_R",
    "build_tth_bin_edges_in_R",
    "polygon_area",
    "circle_seg_intersect",
    "ray_seg_intersect",
    "point_in_quad",
    "invert_REta_to_pixel",
    # detector_mapper
    "build_map",
    "BuildMapResult",
    # kernels
    "AREA_THRESHOLD",
    "CSRGeometry",
    "build_csr",
    "integrate",
    "integrate_with_variance",
    "profile_1d",
    "profile_1d_with_variance",
    # image
    "bytes_per_pixel",
    "decode_payload",
    "decode_hybrid_payload",
    "NUMPY_DTYPE_FOR_CODE",
    "DTYPE_CODE_UINT8",
    "DTYPE_CODE_UINT16",
    "DTYPE_CODE_UINT32",
    "DTYPE_CODE_INT64",
    "DTYPE_CODE_FLOAT32",
    "DTYPE_CODE_FLOAT64",
    "DTYPE_CODE_HYBRID",
    # peakfit
    "fit_peaks",
    "fit_peaks_autodetect",
    "pseudo_voigt",
    "snip_background",
    "PF_PARAMS_PER_PEAK",
    # alias-mitigation API (Family II + III)
    "build_fused_geometry",
    "gauss_smooth_eta",
    "median_filter_eta",
    # exporters
    "zarr_to_csv",
    # calibrate-v2 compat
    "params_from_v2_unpacked",
    "params_from_calibration_spec",
]

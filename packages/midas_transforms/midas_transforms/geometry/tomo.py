"""Tomographic reconstructions -> :class:`SampleShape`.

What this is for
----------------
:mod:`~midas_transforms.geometry.sample_shape` supplies the illuminated volume
that ``V_gauge = Hbeam * pi * Rsample^2`` has always been standing in for. Its
analytic constructors cover a rod or a prism; this module covers the case where
the specimen shape was actually *measured*.

Four input routes, because four exist in the wild:

===========================  =====================================================
:func:`from_midas_tomo_bin`  the engine's own ``.bin``; shape lives in the FILENAME
:func:`from_nxtomoproc`      ``midas_tomo.hdf5.write_recon_hdf5`` output
:func:`from_square_uint8`    the legacy ``TomoImage`` NF uses (side = isqrt(size))
:func:`from_array`           anything already in memory
===========================  =====================================================

What is refused, and why
------------------------
Every reader takes the registration as **required input**. This is not
defensiveness; each of these is a silent wrong answer:

``pixel_size_um``
    No reconstruction format in use here records it. It scales every path
    length, and volume as its cube. A wrong value gives a sharp, plausible
    reconstruction of an object of the wrong size.

``rot_axis_ix`` / ``rot_axis_iy``
    An *output* of the reconstruction (the shift sweep picked it), not a
    property of the detector. ``n // 2`` is a guess.

``in_plane``
    The wrong handedness mirrors the sample. A mirrored mask reconstructs
    perfectly and gives smooth, plausible, wrong path lengths — the failure
    mode ranked #3 in the plan's silent-failure list.

``shift_index``
    The ``n_shifts`` axis of a MIDAS-tomo cube is a **sweep over candidate
    rotation-axis shifts**, and all but one of them are wrong. Index 0 is the
    lowest shift in the sweep, not the best one. Omitting it is allowed only
    when the cube holds exactly one shift.

``threshold``
    It multiplies the illuminated volume directly. Use
    :func:`threshold_sensitivity` and report a band; if the volume is not
    stationary in the threshold, the mask is not usable as a volume estimate.

The padding trap
----------------
``recon_xdim = next_power_of_2(det_xdim)`` (``midas_tomo/config.py:198``), so a
1365-wide detector reconstructs onto a 2048 grid and **a third of every slice
is padding that no ray ever sampled**. Parallel-beam FBP can only reconstruct
the disc of radius ``det_xdim / 2`` about the rotation axis; outside it the
values are reconstruction artefacts, and thresholding picks them up.

The readers therefore **clip** the mask to that disc and record how much was
clipped. They do not reject a reconstruction for a handful of corner voxels —
the corners of a square grid are outside every projection's field of view by
construction, so a reject-on-any rule refuses essentially every real tomogram
(measured: it refused all 12 thresholds on bt_1id_jun25b NMC811 s5, including ones
where only 0.5 % of the mask was outside). A *large* overflow still raises,
because that means the sample is wider than the field of view — truncated, and
then no threshold gives a usable mask — or the rotation axis is wrong.
``max_pad_fraction`` sets the boundary, 1 % by default.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .sample_shape import SampleShape

__all__ = [
    "from_array",
    "from_midas_tomo_bin",
    "from_nxtomoproc",
    "from_square_uint8",
    "load_square_tomo",
    "otsu_threshold",
    "parse_recon_filename",
    "threshold_sensitivity",
]

log = logging.getLogger(__name__)

# midas_tomo/config.py:207 output_path()
_RECON_NAME = re.compile(
    r"(?:_NrCleanup_(?P<cleanup>\d+))?"
    r"_NrShifts_(?P<shifts>\d+)"
    r"_NrSlices_(?P<slices>\d+)"
    r"_XDim_(?P<xdim>\d+)"
    r"_YDim_(?P<ydim>\d+)"
    r"_float32\.bin$"
)


def parse_recon_filename(path: Union[str, Path]) -> Dict[str, int]:
    """Recover the cube shape a MIDAS-tomo ``.bin`` encodes in its name.

    The engine writes a bare float32 blob and puts the only copy of its shape
    in the filename (``midas_tomo/api.py:48`` reads it back from the config
    instead, which works exactly as long as the config is still around). A
    rename or a copy loses it, so :func:`from_midas_tomo_bin` cross-checks the
    parsed shape against the file size rather than trusting either alone.
    """
    name = Path(path).name
    m = _RECON_NAME.search(name)
    if m is None:
        raise ValueError(
            f"{name!r} is not a MIDAS-tomo reconstruction filename. Expected "
            "'..._NrShifts_NNN_NrSlices_NNNNN_XDim_NNNNNN_YDim_NNNNNN_float32.bin'. "
            "The shape exists nowhere else in this format, so a renamed file "
            "cannot be read — re-export it with midas_tomo.hdf5.write_recon_hdf5, "
            "which stores the shape with the data."
        )
    out = {
        "n_shifts": int(m.group("shifts")),
        "n_slices": int(m.group("slices")),
        "xdim": int(m.group("xdim")),
        "ydim": int(m.group("ydim")),
    }
    if m.group("cleanup") is not None:
        out["n_cleanup"] = int(m.group("cleanup"))
    return out


# --------------------------------------------------------------------- core

def from_array(
    volume,
    *,
    pixel_size_um: float,
    rot_axis_ix: float,
    rot_axis_iy: float,
    in_plane: str,
    threshold: float,
    slice_pitch_um: Optional[float] = None,
    slice0_z_um: float = 0.0,
    det_xdim: Optional[int] = None,
    max_pad_fraction: float = 0.01,
    provenance: Optional[Dict[str, Any]] = None,
) -> SampleShape:
    """Threshold a reconstruction array into a :class:`SampleShape`.

    ``volume`` is ``(n_slices, ny, nx)``, or ``(ny, nx)`` for a single slice.
    Occupancy is ``volume >= threshold`` — hard, not feathered, because a
    reconstruction's greyscale is attenuation and interpolating it as a volume
    fraction would double-count the partial-volume effect the recon already
    has.

    ``det_xdim`` enables the padding check described in the module docstring.
    Omitting it falls back to the grid's own inscribed circle, which is a
    weaker check; the difference is recorded in ``provenance``.
    """
    vol = np.asarray(volume)
    if vol.ndim == 2:
        vol = vol[None, :, :]
    if vol.ndim != 3:
        raise ValueError(
            f"volume must be 2-D or 3-D (slice, iy, ix); got shape {vol.shape}"
        )
    if not np.isfinite(vol).all():
        n_bad = int((~np.isfinite(vol)).sum())
        raise ValueError(
            f"reconstruction holds {n_bad} non-finite values. Thresholding "
            "them would put NaN geometry into the volume estimate; clean the "
            "reconstruction first."
        )

    occ = (vol >= float(threshold)).astype(np.float64)
    prov: Dict[str, Any] = {
        "threshold": float(threshold),
        "n_voxels_occupied": int(occ.sum()),
        "recon_value_range": (float(vol.min()), float(vol.max())),
        # Nothing here has been registered against a diffraction dataset. Say
        # so explicitly: a SampleShape carries no other record of it.
        "registration": "NOT verified - run the V1/V2 checks before trusting "
                        "slice0_z_um, rot_axis_* or in_plane",
    }
    if provenance:
        prov.update(provenance)

    prov.update(
        _check_pad_occupancy(
            occ, rot_axis_ix=rot_axis_ix, rot_axis_iy=rot_axis_iy,
            det_xdim=det_xdim, max_pad_fraction=max_pad_fraction,
        )
    )

    return SampleShape(
        occupancy=occ,
        pixel_size_um=float(pixel_size_um),
        slice_pitch_um=float(
            slice_pitch_um if slice_pitch_um is not None else pixel_size_um
        ),
        rot_axis_ix=float(rot_axis_ix),
        rot_axis_iy=float(rot_axis_iy),
        slice0_z_um=float(slice0_z_um),
        in_plane=in_plane,
        provenance=prov,
    )


def _check_pad_occupancy(
    occ: np.ndarray, *, rot_axis_ix: float, rot_axis_iy: float,
    det_xdim: Optional[int], max_pad_fraction: float = 0.01,
) -> Dict[str, Any]:
    """Clip the mask to the reconstructible disc, refusing a big overflow.

    Parallel-beam FBP samples only the disc of radius ``det_xdim / 2`` about
    the rotation axis; outside it there is no data, only ringing. So the
    correct treatment is to **zero that region**, not to reject a whole
    reconstruction because a handful of corner voxels rang above the
    threshold — the corners of a square reconstruction grid are outside the
    field of view of every projection by construction, so a strict
    reject-on-any rule refuses essentially every real tomogram.

    A *large* overflow is a different matter and still raises: it means either
    the sample is wider than the field of view (truncated, and then no
    threshold gives a usable mask) or the rotation axis is wrong. The default
    boundary is 1 % of the mask.
    """
    _, ny, nx = occ.shape
    if det_xdim is not None:
        r_px = 0.5 * float(det_xdim)
        basis = f"det_xdim={int(det_xdim)}"
    else:
        r_px = 0.5 * float(min(nx, ny))
        basis = "grid inscribed circle (det_xdim not supplied - weaker check)"

    iy, ix = np.mgrid[0:ny, 0:nx].astype(np.float64)
    outside = np.hypot(ix - rot_axis_ix, iy - rot_axis_iy) > r_px
    n_out = int(occ[:, outside].sum())
    total = float(occ.sum())
    frac = n_out / total if total > 0 else 0.0

    if frac > max_pad_fraction:
        raise ValueError(
            f"{n_out} occupied voxels ({100 * frac:.2f} % of the mask, above "
            f"the {100 * max_pad_fraction:.2f} % limit) lie outside the "
            f"reconstructible disc of radius {r_px:.1f} px about the rotation "
            f"axis ({basis}). That is too much to be corner ringing. Either "
            "the sample is wider than the field of view - in which case the "
            "reconstruction cups and no threshold gives a usable mask - or "
            "rot_axis_ix/iy is wrong. Raising the threshold treats the "
            "symptom; check the projections for truncation first."
        )

    occ[:, outside] = 0.0
    return {"pad_occupancy_clipped": n_out,
            "pad_occupancy_clipped_fraction": frac,
            "reconstructible_radius_px": r_px,
            "pad_check_basis": basis,
            "max_pad_fraction": float(max_pad_fraction)}


# ------------------------------------------------------------------ readers

def from_midas_tomo_bin(
    path: Union[str, Path],
    *,
    pixel_size_um: float,
    rot_axis_ix: float,
    rot_axis_iy: float,
    in_plane: str,
    threshold: float,
    shift_index: Optional[int] = None,
    cleanup_index: Optional[int] = None,
    slice_range: Optional[Tuple[int, int]] = None,
    slice_pitch_um: Optional[float] = None,
    slice0_z_um: float = 0.0,
    det_xdim: Optional[int] = None,
    max_pad_fraction: float = 0.01,
) -> SampleShape:
    """Read the MIDAS-tomo engine's float32 cube.

    Layout ``(n_shifts, n_slices, X, X)`` — or ``(n_cleanup, n_shifts,
    n_slices, X, X)`` in stripe-sweep mode — with the shape carried only in the
    filename (``midas_tomo/config.py:207``). Memory-mapped, so ``slice_range``
    reads just the slices an FF layer needs instead of a multi-GB cube.

    ``shift_index`` selects one entry of the rotation-axis sweep. It is
    required whenever the cube holds more than one, because the sweep is a
    search and every entry but one is a mis-registered reconstruction.
    """
    path = Path(path)
    meta = parse_recon_filename(path)
    n_shifts, n_slices = meta["n_shifts"], meta["n_slices"]
    xdim, ydim = meta["xdim"], meta["ydim"]
    n_cleanup = meta.get("n_cleanup")

    shape = (n_shifts, n_slices, ydim, xdim)
    if n_cleanup is not None:
        shape = (n_cleanup, *shape)

    want = int(np.prod(shape)) * 4
    have = path.stat().st_size
    if have != want:
        raise ValueError(
            f"{path.name}: the filename says {'x'.join(str(s) for s in shape)} "
            f"float32 = {want} bytes, but the file is {have} bytes. The shape "
            "is only in the name, so either the file was renamed or it is "
            "truncated. Do not guess which."
        )

    cube = np.memmap(path, dtype=np.float32, mode="r", shape=shape)

    if n_cleanup is not None:
        if cleanup_index is None:
            raise ValueError(
                f"{path.name} is a stripe-removal sweep with {n_cleanup} "
                "cleanup settings; pass cleanup_index to choose one."
            )
        cube = cube[int(cleanup_index)]

    shift_index = _resolve_shift_index(shift_index, n_shifts, path.name)
    cube = cube[shift_index]

    s0, s1 = _resolve_slice_range(slice_range, n_slices)
    vol = np.asarray(cube[s0:s1], dtype=np.float32)

    return from_array(
        vol,
        pixel_size_um=pixel_size_um,
        slice_pitch_um=slice_pitch_um,
        rot_axis_ix=rot_axis_ix, rot_axis_iy=rot_axis_iy,
        in_plane=in_plane, threshold=threshold,
        slice0_z_um=slice0_z_um
        + s0 * float(slice_pitch_um if slice_pitch_um is not None else pixel_size_um),
        det_xdim=det_xdim, max_pad_fraction=max_pad_fraction,
        provenance={
            "source": str(path), "format": "midas_tomo.bin",
            "shift_index": shift_index, "n_shifts": n_shifts,
            "cleanup_index": cleanup_index,
            "slice_range": (s0, s1), "n_slices_in_file": n_slices,
            "recon_xdim": xdim,
        },
    )


def from_nxtomoproc(
    path: Union[str, Path],
    *,
    pixel_size_um: float,
    rot_axis_ix: float,
    rot_axis_iy: float,
    in_plane: str,
    threshold: float,
    shift_index: Optional[int] = None,
    slice_range: Optional[Tuple[int, int]] = None,
    slice_pitch_um: Optional[float] = None,
    slice0_z_um: float = 0.0,
    det_xdim: Optional[int] = None,
    max_pad_fraction: float = 0.01,
    dataset: str = "entry/reconstruction/data",
) -> SampleShape:
    """Read a reconstruction written by ``midas_tomo.hdf5.write_recon_hdf5``.

    ``/entry/reconstruction/data`` is ``(shift, slice, y, x)``; the axis order
    is asserted from the dataset's own ``axes`` attribute rather than assumed,
    since a transposed cube would reconstruct a sample rotated into the beam.
    ``axis_shift``, when present, is recorded in provenance so the chosen
    ``shift_index`` can be reported as a shift in pixels.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "reading NXtomoproc needs h5py: pip install h5py"
        ) from exc

    path = Path(path)
    with h5py.File(path, "r") as hf:
        if dataset not in hf:
            raise KeyError(
                f"{path} has no {dataset!r}. Available top-level groups: "
                f"{sorted(hf.keys())}"
            )
        ds = hf[dataset]
        axes = ds.attrs.get("axes")
        if axes is not None:
            axes = axes.decode() if isinstance(axes, bytes) else str(axes)
            if axes != "shift:slice:y:x":
                raise ValueError(
                    f"{path}: {dataset} declares axes {axes!r}, not "
                    "'shift:slice:y:x'. A transposed cube reconstructs a "
                    "sample rotated into the beam and is silent about it."
                )
        if ds.ndim != 4:
            raise ValueError(
                f"{path}: {dataset} is {ds.ndim}-D; expected 4-D "
                "(shift, slice, y, x)"
            )
        n_shifts, n_slices = int(ds.shape[0]), int(ds.shape[1])
        shift_index = _resolve_shift_index(shift_index, n_shifts, path.name)
        s0, s1 = _resolve_slice_range(slice_range, n_slices)
        vol = np.asarray(ds[shift_index, s0:s1])

        prov: Dict[str, Any] = {
            "source": str(path), "format": "NXtomoproc",
            "shift_index": shift_index, "n_shifts": n_shifts,
            "slice_range": (s0, s1), "n_slices_in_file": n_slices,
        }
        grp = ds.parent
        if "axis_shift" in grp:
            shifts = np.asarray(grp["axis_shift"])
            if shifts.size == n_shifts:
                prov["axis_shift_px"] = float(shifts[shift_index])
        if "entry" in hf:
            for k, v in dict(hf["entry"].attrs).items():
                prov[f"h5_{k}"] = v

    return from_array(
        vol,
        pixel_size_um=pixel_size_um,
        slice_pitch_um=slice_pitch_um,
        rot_axis_ix=rot_axis_ix, rot_axis_iy=rot_axis_iy,
        in_plane=in_plane, threshold=threshold,
        slice0_z_um=slice0_z_um
        + s0 * float(slice_pitch_um if slice_pitch_um is not None else pixel_size_um),
        det_xdim=det_xdim, max_pad_fraction=max_pad_fraction,
        provenance=prov,
    )


def from_square_uint8(
    path: Union[str, Path],
    *,
    pixel_size_um: float,
    in_plane: str,
    rot_axis_ix: Optional[float] = None,
    rot_axis_iy: Optional[float] = None,
    threshold: float = 1.0,
    slice_pitch_um: Optional[float] = None,
    slice0_z_um: float = 0.0,
    det_xdim: Optional[int] = None,
    max_pad_fraction: float = 0.01,
) -> SampleShape:
    """The legacy square ``TomoImage`` — one uint8 slice, side = ``isqrt(size)``.

    This is what NF's ``GridMask`` / ``TomoImage`` path consumes
    (``midas_nf_preprocess/tomo_filter/filter.py:33``). It is a *mask*, not a
    reconstruction, so the default threshold of 1.0 means "any non-zero".

    Unlike the other readers the rotation axis defaults to ``(n - 1) / 2``,
    because that is not a guess here: the C consumer hard-codes
    ``x_pos = int(x / px) + n // 2`` (``filter.py:106``), so the format's own
    convention *is* the centre. The half-pixel difference between ``n // 2``
    and ``(n - 1) / 2`` is the C off-by-one recorded in ``manuals/tomo/``;
    pass the values explicitly to reproduce the C exactly.
    """
    path = Path(path)
    img = load_square_tomo(path)
    n = int(img.shape[0])
    if rot_axis_ix is None:
        rot_axis_ix = (n - 1) / 2.0
    if rot_axis_iy is None:
        rot_axis_iy = (n - 1) / 2.0

    return from_array(
        img.astype(np.float64)[None, :, :],
        pixel_size_um=pixel_size_um,
        slice_pitch_um=slice_pitch_um,
        rot_axis_ix=rot_axis_ix, rot_axis_iy=rot_axis_iy,
        in_plane=in_plane, threshold=threshold, slice0_z_um=slice0_z_um,
        det_xdim=det_xdim, max_pad_fraction=max_pad_fraction,
        provenance={
            "source": str(path), "format": "legacy square uint8 TomoImage",
            "side_px": n,
            "note": "a single slice; illuminated volume needs slice_pitch_um "
                    "to be the beam height, not the pixel size",
        },
    )


def load_square_tomo(path: Union[str, Path], dtype=np.uint8) -> np.ndarray:
    """Legacy ``TomoImage``: a raw square image whose side is ``isqrt(size)``.

    Transcribed from ``filterGridfromTomo.c:13-21``. Deliberately a second copy
    of the six lines in ``midas_nf_preprocess/tomo_filter/filter.py:33`` rather
    than an import: ``midas_transforms`` sits *below* ``midas_nf_preprocess``
    in the dependency graph, and inverting that to share a file-size ``isqrt``
    would be the wrong trade. The thing that must not be duplicated is the
    coordinate convention, and that lives once in ``midas_stress.frames``.
    """
    path = Path(path)
    sz = path.stat().st_size
    itemsize = np.dtype(dtype).itemsize
    n = int(math.isqrt(sz // itemsize))
    if n * n * itemsize != sz:
        raise ValueError(
            f"{path}: size {sz} bytes is not a perfect square for dtype "
            f"{np.dtype(dtype)}. This format stores no shape, so a non-square "
            "file cannot be interpreted."
        )
    return np.fromfile(path, dtype=dtype, count=n * n).reshape(n, n)


def otsu_threshold(volume) -> float:
    """Between the air mode and the specimen mode, from the histogram.

    The threshold range for :func:`threshold_sensitivity` has to come from
    somewhere physical; the two obvious shortcuts are both wrong (see that
    function's docstring). Otsu's split maximises the between-class variance of
    the greyscale, which for a reconstruction is exactly the air/specimen
    boundary, and — unlike a percentile — it moves when the reconstruction
    changes, which is what makes a sweep around it meaningful.
    """
    v = np.asarray(volume, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to threshold")
    hist, edges = np.histogram(v, bins=256)
    centres = 0.5 * (edges[1:] + edges[:-1])
    w = hist.astype(np.float64) / max(hist.sum(), 1)
    cw = np.cumsum(w)
    cm = np.cumsum(w * centres)
    gm = cm[-1]
    denom = np.maximum(cw * (1.0 - cw), 1e-12)
    between = (gm * cw - cm) ** 2 / denom
    return float(centres[int(np.nanargmax(between))])


# ----------------------------------------------------------- threshold sweep

def threshold_sensitivity(
    volume,
    thresholds: Sequence[float],
    *,
    voxel_volume_um3: float,
    stationary_tol: float = 0.05,
) -> Dict[str, Any]:
    """Sweep the threshold and report whether the volume is stationary in it.

    The threshold multiplies ``V_illum`` directly and there is no principled
    value, so a single number is a free parameter dressed as a measurement.
    What makes a mask *usable* is a plateau: a range over which the volume
    barely moves, because the reconstruction has real contrast between sample
    and air.

    Returns ``volumes_um3``, the fractional spread over the sweep, and
    ``stationary`` — True when that spread is within ``stationary_tol``. A
    False verdict means the volume estimate is threshold-driven; report the
    band, do not pick the middle.

    **The thresholds must be chosen on physical grounds, and two obvious
    choices are both wrong.** Measured while gating Paganin on bt_1id_jun25b
    NMC811 s5:

    * *Percentiles of the reconstruction* (``linspace(p50, p99.5)``) make
      ``radius_spread`` a **constant**. The volume then always runs from ~50 %
      of the voxels to ~0.5 %, a ratio of 100, so it reports
      ``100**(1/3) = 4.642`` whatever the data looks like — identical across
      six reconstructions that were visibly different. ``fractional_spread``
      still carries information, because it depends on the shape of the curve
      in between; ``radius_spread`` does not.
    * *A fixed absolute range* is unfair to any processing that changes the
      value scale. Paganin lowers peak attenuation as it smooths, so on a
      fixed scale its volumes collapse (72.9 um extent to 5.7 um across a
      delta/beta sweep) for a reason that has nothing to do with mask quality.

    What the check is really asking is whether the histogram is bimodal with a
    plateau between the modes. Choose the range from the air and specimen
    levels, and say which choice was made alongside the number.
    """
    vol = np.asarray(volume)
    th = np.asarray(thresholds, dtype=np.float64)
    if th.size < 2:
        raise ValueError("need at least two thresholds to measure sensitivity")

    vols = np.array([float((vol >= t).sum()) * float(voxel_volume_um3) for t in th])
    lo, hi = float(vols.min()), float(vols.max())
    mid = float(np.median(vols))
    spread = (hi - lo) / mid if mid > 0 else math.inf

    return {
        "thresholds": th,
        "volumes_um3": vols,
        "median_volume_um3": mid,
        "fractional_spread": spread,
        "stationary": bool(spread <= stationary_tol),
        "radius_spread": (hi / lo) ** (1 / 3) if lo > 0 else math.inf,
    }


# ----------------------------------------------------------------- helpers

def _resolve_shift_index(shift_index: Optional[int], n_shifts: int, name: str) -> int:
    if shift_index is None:
        if n_shifts == 1:
            return 0
        raise ValueError(
            f"{name} holds {n_shifts} rotation-axis shifts. That axis is a "
            "SWEEP over candidate shifts and all but one are mis-registered "
            "reconstructions, so there is no safe default — index 0 is the "
            "lowest shift tried, not the best. Pass shift_index explicitly "
            "(midas_tomo.center picks it)."
        )
    idx = int(shift_index)
    if not (0 <= idx < n_shifts):
        raise IndexError(f"shift_index {idx} out of range for {n_shifts} shifts")
    return idx


def _resolve_slice_range(
    slice_range: Optional[Tuple[int, int]], n_slices: int
) -> Tuple[int, int]:
    if slice_range is None:
        return 0, n_slices
    s0, s1 = (int(v) for v in slice_range)
    if not (0 <= s0 < s1 <= n_slices):
        raise ValueError(
            f"slice_range {(s0, s1)} is not a valid half-open range within "
            f"{n_slices} slices"
        )
    return s0, s1

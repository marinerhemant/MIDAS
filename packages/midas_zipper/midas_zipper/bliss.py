"""ESRF Bliss (ID11) raw data -> one ``.MIDAS.zip`` per scan, with per-frame omega.

A Bliss master file holds one entry per scan (``1.1``, ``2.1``, ...). Each entry
has ``measurement/<detector>`` -- usually a virtual dataset over the Lima files
``scanNNNN/eiger_NNNN.h5`` next to the master -- ``measurement/rot_center``, the
encoder angle at the centre of every frame, and ``instrument/positioners/dty``.
Frames are written by :func:`midas_zipper.generate_ff_zip`, unchanged; this
module only finds them and adds what the zipper does not know about:

``measurement/process/scan_parameters/omegaCenter``
    per-frame omega. midas_peakfit uses it in place of start + i*step, which is
    what makes a SNAKE scan (omega running up on one scan and down on the next)
    come out right. It is ``sign * rot_center + offset``.
``measurement/process/scan_parameters/dty``
    the scan's translation, for the record (positions.csv is built separately).
``exchange/mask``
    optional, copied from a reference zip of the same detector.

**The omega sign is not guessed.** ``rot`` is the diffractometer's angle; whether
MIDAS's omega is ``+rot`` or ``-rot`` (and with what offset) is a convention a
wrong choice of which mirrors the map without any error (pf-hedm halt condition).
:func:`omega_map_from_reference` takes it from data: zips of another layer of the
same experiment that are known to be right. A repeated fly scan re-reads the same
encoder positions, so exactly one of ``+rot`` / ``-rot`` matches the reference's
omegaCenter frame by frame (to the encoder's repeatability), and the other does
not. If neither matches, it refuses.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Eiger/Lima frames are bitshuffle/LZ4 compressed
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__all__ = ["BlissScan", "survey_master", "omega_map_from_reference", "convert_master"]

OMEGA_CENTER_KEY = "measurement/process/scan_parameters/omegaCenter"
DTY_KEY = "measurement/process/scan_parameters/dty"
MASK_KEY = "exchange/mask"


@dataclass
class BlissScan:
    entry: str                 # e.g. "12.1"
    title: str
    n_frames: int
    frames_file: str           # file the zipper reads (a Lima file if the VDS has one source, else the master)
    frames_path: str           # dataset path inside frames_file
    rot_center: np.ndarray = field(repr=False)
    dty: Optional[float]
    frame_start: int = 0       # frames [frame_start, frame_start + n_frames) of frames_path (fscan2d lines)
    total_frames: int = -1     # length of frames_path; -1 = n_frames (the whole dataset is this scan)


def _entry_key(name: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in name.split("."))
    except ValueError:
        return (10**9,)


def survey_master(master: os.PathLike | str, *, detector: str = "eiger", rot: str = "rot_center",
                  dty: str = "dty", dty_tol: float = 1e-3, min_frames_per_line: int = 10) -> List[BlissScan]:
    """Every scan of a Bliss master, in order: each entry with a 3-D ``measurement/<detector>`` and a
    per-frame ``measurement/<rot>`` (entries without them - alignment counts, ct - are skipped). An entry
    whose ``dty`` is recorded per frame and steps (an fscan2d: every line in one entry) is split into one
    scan per run of constant ``dty`` (changes > ``dty_tol``); a ``dty`` that changes within fewer than
    ``min_frames_per_line`` frames is refused as a continuous motion."""
    import h5py

    master = Path(master).resolve()
    out: List[BlissScan] = []
    with h5py.File(master, "r") as f:
        for name in sorted(f.keys(), key=_entry_key):
            g = f[name]
            if f"measurement/{detector}" not in g or f"measurement/{rot}" not in g:
                continue
            d = g[f"measurement/{detector}"]
            if not isinstance(d, h5py.Dataset) or d.ndim != 3:
                continue
            frames_file, frames_path = str(master), f"/{name}/measurement/{detector}"
            if d.is_virtual:
                # Read the Lima file directly only when the VDS maps ALL of one source dataset onto ALL of
                # itself, same shape; a partial mapping (a frame range, several scans in one file) must be
                # read through the VDS, or the wrong frames come out.
                vs = d.virtual_sources()
                if len(vs) == 1 and vs[0].file_name != ".":
                    s = vs[0]; npts = int(np.prod(d.shape))
                    src_all = (s.src_space.get_select_type() == h5py.h5s.SEL_ALL
                               or (s.src_space.shape == d.shape and s.src_space.get_select_npoints() == npts))
                    whole = src_all and s.vspace.get_select_npoints() == npts
                    p = Path(s.file_name)
                    p = p if p.is_absolute() else (master.parent / p).resolve()
                    if whole:                    # confirm against the source itself (SEL_ALL carries no shape)
                        try:
                            with h5py.File(p, "r") as src:
                                whole = src[s.dset_name].shape == d.shape
                        except (OSError, KeyError):
                            whole = False
                    if whole:
                        frames_file, frames_path = str(p), s.dset_name
            r = np.asarray(g[f"measurement/{rot}"][()], dtype=np.float64).reshape(-1)
            if r.size != d.shape[0]:
                raise ValueError(f"{master.name} entry {name}: {r.size} {rot} values for {d.shape[0]} frames")
            title = g["title"][()] if "title" in g else b""
            title = title.decode() if isinstance(title, bytes) else str(title)
            yf = None                                   # per-frame dty (fscan2d: all lines in one entry)
            for cand in (f"measurement/{dty}", f"measurement/{dty}_center"):
                if cand in g and g[cand].shape == (d.shape[0],):
                    yf = np.asarray(g[cand][()], dtype=np.float64); break
            pos = g.get("instrument/positioners")
            if yf is None and pos is not None and dty in pos:
                v = np.asarray(pos[dty][()], dtype=np.float64).reshape(-1)
                if v.size == d.shape[0] and v.size > 1:
                    yf = v
                elif v.size and np.ptp(v) > 1e-6:
                    raise ValueError(f"{master.name} entry {name}: {dty} has {v.size} values that differ, "
                                     f"for {d.shape[0]} frames; cannot tell which frames belong to which line")
                elif v.size:
                    yf = np.full(d.shape[0], v[0])
            if yf is None or np.ptp(yf) <= 1e-6:
                y = None if yf is None else float(yf[0])
                out.append(BlissScan(name, title, int(d.shape[0]), frames_file, frames_path, r, y))
                continue
            # dty moves inside the entry: one scan per run of constant dty, read through the master's VDS
            cut = np.flatnonzero(np.abs(np.diff(yf)) > dty_tol) + 1
            starts = np.r_[0, cut]; stops = np.r_[cut, yf.size]
            if (stops - starts).min() < min_frames_per_line:
                raise ValueError(f"{master.name} entry {name}: {dty} changes every {int((stops - starts).min())} "
                                 f"frame(s) - a continuously moving {dty}, not a line scan")
            for a, b in zip(starts, stops):
                out.append(BlissScan(f"{name}[{a}:{b}]", title, int(b - a), str(master), f"/{name}/measurement/{detector}",
                                     r[a:b], float(np.median(yf[a:b])), int(a), int(d.shape[0])))
    return out


def omega_map_from_reference(rots: Sequence[np.ndarray], ref_omegas: Sequence[np.ndarray], *,
                             tol_deg: float = 0.05, max_abs_offset: Optional[float] = 0.05) -> Dict[str, float]:
    """Pin ``omega = sign * rot + offset`` against reference per-frame omegas of the same scans.

    ``rots[k]`` and ``ref_omegas[k]`` are the k-th scan of the new data and of the reference
    (same scan pattern, same frame count). For each sign the offset is the median difference
    over all frames, and the residual is the 99th percentile of |difference - offset|. Exactly
    one sign must have a residual below ``tol_deg``; otherwise this raises.

    The offset must also be ~0 (``|offset| < max_abs_offset``; pass None to allow any). A snake scan that
    STARTS in the other direction from the reference matches it frame by frame as ``-rot + (first + last)``:
    the wrong sign with a residual at encoder noise, i.e. a silently mirrored map. A true convention is
    ``+-rot``; an offset is accepted only when the caller knows it is real.
    Returns {"sign", "offset", "residual_p99", "other_residual_p99", "n_scans", "n_frames"}."""
    pairs = [(np.asarray(r, float).ravel(), np.asarray(o, float).ravel()) for r, o in zip(rots, ref_omegas)]
    if not pairs:
        raise ValueError("no scans to compare")
    for k, (r, o) in enumerate(pairs):
        if r.size != o.size:
            raise ValueError(f"scan {k}: {r.size} frames vs {o.size} in the reference")
    R = np.concatenate([r for r, _ in pairs]); O = np.concatenate([o for _, o in pairs])
    res = {}
    for s in (+1.0, -1.0):
        off = float(np.median(O - s * R))
        res[s] = (off, float(np.percentile(np.abs(O - s * R - off), 99)))
    good = [s for s in res if res[s][1] < tol_deg]
    if len(good) != 1:
        raise ValueError("omega mapping not determined: residual p99 "
                         f"+rot {res[1.0][1]:.4f} deg, -rot {res[-1.0][1]:.4f} deg (tolerance {tol_deg}); "
                         "the reference is not the same scan pattern, or the encoder differs")
    s = good[0]
    if max_abs_offset is not None and abs(res[s][0]) >= max_abs_offset:
        raise ValueError(f"omega mapping needs an offset of {res[s][0]:+.4f} deg (sign {s:+.0f}); a snake that starts in "
                         "the other direction from the reference matches this way with the WRONG sign. Check the first "
                         "scan's direction; pass max_abs_offset=None only if the offset is known to be real")
    return {"sign": s, "offset": res[s][0], "residual_p99": res[s][1], "other_residual_p99": res[-s][1],
            "n_scans": len(pairs), "n_frames": int(R.size)}


def convert_master(master: os.PathLike | str, out_raw: os.PathLike | str, stem: str,
                   param_file: os.PathLike | str, *, omega_sign: float, omega_offset: float = 0.0,
                   mask_from: Optional[os.PathLike | str] = None, first_scan_nr: int = 1,
                   scans: Optional[List[BlissScan]] = None, only: Optional[Sequence[int]] = None,
                   detector: str = "eiger") -> List[Path]:
    """Write ``out_raw/<n>/<stem>_<n:06d>.MIDAS.zip`` for every scan of ``master`` (n from ``first_scan_nr``,
    in scan order; ``only`` restricts to those n). ``omega_sign`` must be given (+1 or -1): take it from
    :func:`omega_map_from_reference`. ``param_file`` supplies the analysis parameters as for any zip."""
    import zarr
    from . import generate_ff_zip
    from .zipwrite import write_arrays

    if omega_sign not in (1, -1, 1.0, -1.0):
        raise ValueError("omega_sign must be +1 or -1 (pin it with omega_map_from_reference)")
    scans = survey_master(master, detector=detector) if scans is None else scans
    mask = None
    if mask_from is not None:
        mask = np.asarray(zarr.open(str(mask_from), "r")[MASK_KEY][...])
    written = []
    for i, sc in enumerate(scans):
        n = first_scan_nr + i
        if only is not None and n not in set(only):
            continue
        d = Path(out_raw) / str(n); d.mkdir(parents=True, exist_ok=True)
        data_fn, data_loc = sc.frames_file, sc.frames_path
        if sc.total_frames not in (-1, sc.n_frames):     # one line of an fscan2d: a VDS over its frame range
            import h5py
            data_fn = str(d / f"{stem}_{n:06d}_frames.h5"); data_loc = "/data"
            with h5py.File(sc.frames_file, "r") as f:
                full = f[sc.frames_path]; shp, dt = full.shape, full.dtype
            with h5py.File(data_fn, "w") as h:
                lay = h5py.VirtualLayout(shape=(sc.n_frames,) + shp[1:], dtype=dt)
                lay[:] = h5py.VirtualSource(sc.frames_file, sc.frames_path, shape=shp)[sc.frame_start:sc.frame_start + sc.n_frames]
                h.create_virtual_dataset("data", lay)
        generate_ff_zip(result_folder=str(d), param_file=str(param_file), data_fn=data_fn,
                        extra_args=["-dataLoc", data_loc])
        made = d / f"{Path(data_fn).name}.analysis.MIDAS.zip"
        final = d / f"{stem}_{n:06d}.MIDAS.zip"
        if not made.is_file():
            raise RuntimeError(f"zipper wrote no {made}")
        shutil.move(str(made), str(final))
        z = zarr.open(str(final), "r")
        nf = z["exchange/data"].shape[0]
        if nf != sc.n_frames:
            raise RuntimeError(f"{final.name}: {nf} frames written, entry {sc.entry} has {sc.n_frames}")
        arrays = {OMEGA_CENTER_KEY: float(omega_sign) * sc.rot_center + float(omega_offset)}
        if sc.dty is not None:
            arrays[DTY_KEY] = np.array([sc.dty])
        chunks = {}
        if mask is not None:
            if mask.shape[-2:] != z["exchange/data"].shape[1:]:
                raise ValueError(f"mask {mask.shape} does not fit frames {z['exchange/data'].shape}")
            arrays[MASK_KEY] = mask; chunks[MASK_KEY] = (1,) + mask.shape[-2:]
        write_arrays(final, arrays, chunks=chunks)
        written.append(final)
    return written


def main(argv=None) -> int:
    import argparse, json
    ap = argparse.ArgumentParser(prog="midas-bliss-zip", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey", help="list the scans of a Bliss master")
    s.add_argument("master"); s.add_argument("--detector", default="eiger")
    m = sub.add_parser("omega-map", help="pin omega = sign*rot + offset against reference zips of the same scan pattern")
    m.add_argument("master"); m.add_argument("reference_zips", nargs="+", help="reference zips in scan order")
    m.add_argument("--detector", default="eiger"); m.add_argument("--tol-deg", type=float, default=0.05)
    m.add_argument("--allow-offset", action="store_true", help="accept a nonzero offset (see omega_map_from_reference)")
    c = sub.add_parser("convert", help="write one MIDAS zip per scan")
    c.add_argument("master"); c.add_argument("out_raw"); c.add_argument("stem"); c.add_argument("param_file")
    c.add_argument("--omega-sign", type=float, required=True); c.add_argument("--omega-offset", type=float, default=0.0)
    c.add_argument("--mask-from"); c.add_argument("--first-scan-nr", type=int, default=1)
    c.add_argument("--only", type=int, nargs="*"); c.add_argument("--detector", default="eiger")
    a = ap.parse_args(argv)
    if a.cmd == "survey":
        sc = survey_master(a.master, detector=a.detector)
        for k, x in enumerate(sc, 1):
            print(f"{k:4d} {x.entry:>7s} frames {x.n_frames:5d} rot {x.rot_center[0]:9.4f} -> {x.rot_center[-1]:9.4f} "
                  f"dty {x.dty if x.dty is None else round(x.dty, 4)}  [{x.title}]  <- {Path(x.frames_file).name}:{x.frames_path}")
        print(f"{len(sc)} scans")
    elif a.cmd == "omega-map":
        import zarr
        sc = survey_master(a.master, detector=a.detector)
        refs = [np.asarray(zarr.open(z, "r")[OMEGA_CENTER_KEY][...]) for z in a.reference_zips]
        n = min(len(sc), len(refs))
        print(json.dumps(omega_map_from_reference([x.rot_center for x in sc[:n]], refs[:n], tol_deg=a.tol_deg,
                                                  max_abs_offset=None if a.allow_offset else 0.05)))
    else:
        out = convert_master(a.master, a.out_raw, a.stem, a.param_file, omega_sign=a.omega_sign,
                             omega_offset=a.omega_offset, mask_from=a.mask_from,
                             first_scan_nr=a.first_scan_nr, only=a.only, detector=a.detector)
        print(f"wrote {len(out)} zips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

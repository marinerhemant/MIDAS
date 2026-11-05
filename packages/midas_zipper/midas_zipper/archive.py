"""Lossless archive encoding of a MIDAS Zarr.zip: store ``zigzag(frame - background)``.

A stored, subtracted background shrinks detector frames more than any change of
codec does (compression study 2026-10-26, ``dev/compression_study``; provisional).
The mapping here is a bijection on n-bit words, so the raw frames come back bit
for bit, negative residuals and gap-pixel sentinels included. This is NOT the
zipper's ``-preProcThresh`` path, which zeroes pixels and cannot be undone.

Layout of an archive zip
------------------------
``exchange/data`` is absent. In its place:

* ``exchange/data_resid``  unsigned words, same width as the raw dtype, one frame
  per chunk, attrs ``encoding`` / ``background`` / ``shift`` / ``orig_dtype``;
* ``exchange/background``  ``(1, Z, Y)`` in the raw dtype;
* everything else (``exchange/dark``, ``exchange/bright``, ``measurement``,
  ``analysis``) is copied unchanged.

A reader that does not know the encoding fails with a ``KeyError`` on
``exchange/data`` instead of silently reading residuals. Use :func:`read_frames`
or :func:`restore_zip` (back to an ordinary MIDAS zip).

If no candidate beats the raw frames by ``MIN_GAIN`` the zip is left untouched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ENCODING = "zigzag_bg_v1"
MIN_GAIN = 0.02          # a candidate must be this much smaller than raw
TRIAL_FRAMES = 16        # frames used to pick the encoding
MEDIAN_FRAMES = 64       # frames used for the scan median, memory permitting
MEMORY_BUDGET = 1.5e9    # bytes of frames held at once (Mac limit is 3 GB/process)
MODES = ("dark", "median", "auto")


def _make_compressor():
    from numcodecs import Blosc
    return Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)


def _uns(dtype):
    return np.dtype(f"u{np.dtype(dtype).itemsize}")


def _sig(dtype):
    return np.dtype(f"i{np.dtype(dtype).itemsize}")


def zigzag_encode(a, ref):
    """``zigzag(a - ref)`` modulo 2^n, returned as unsigned words. ``ref`` broadcasts."""
    a = np.ascontiguousarray(a)
    U, S, n = _uns(a.dtype), _sig(a.dtype), 8 * a.dtype.itemsize
    ref = np.ascontiguousarray(np.asarray(ref).astype(a.dtype, copy=False))
    d = (a.view(U) - ref.view(U)).view(S)
    return ((d << 1) ^ (d >> (n - 1))).view(U)


def zigzag_decode(z, ref, dtype):
    """Inverse of :func:`zigzag_encode`; returns an array of ``dtype``."""
    dtype = np.dtype(dtype)
    U = _uns(dtype)
    z = np.ascontiguousarray(z).view(U)
    ref = np.ascontiguousarray(np.asarray(ref).astype(dtype, copy=False))
    d = (z >> U.type(1)) ^ (np.zeros_like(z) - (z & U.type(1)))
    return (ref.view(U) + d).view(dtype)


def common_shift(frames):
    """Number of low bits that are zero in every value (12-bit data in 16 bits: 4)."""
    U = _uns(frames.dtype)
    v = int(np.bitwise_or.reduce(np.ascontiguousarray(frames).view(U).ravel()))
    return 0 if v == 0 else (v & -v).bit_length() - 1


def quantize_background(bg, shift):
    """Round a background down onto the data's bit grid (needed for 12-in-16 cameras)."""
    if shift == 0:
        return bg
    U = _uns(bg.dtype)
    return ((np.ascontiguousarray(bg).view(U) >> U.type(shift)) << U.type(shift)).view(bg.dtype)


def _clip_round(x, dtype):
    info = np.iinfo(dtype)
    return np.clip(np.rint(x), info.min, info.max).astype(dtype)


def _evenly(n, k):
    return np.unique(np.linspace(0, n - 1, min(n, k)).astype(int))


def scan_median(data, nsub=MEDIAN_FRAMES, budget=MEMORY_BUDGET):
    """Per-pixel median of evenly spaced frames of ``data`` (an ``(N, Z, Y)`` zarr array).

    Holds at most ``budget`` bytes of frames (8 frames minimum, so very large detectors use few) and
    takes the median in row bands, so the temporary arrays stay small.
    """
    frame_bytes = int(np.prod(data.shape[1:])) * data.dtype.itemsize
    k = int(max(8, min(nsub, budget // (2 * max(frame_bytes, 1)))))
    idx = _evenly(data.shape[0], k)
    stack = np.empty((len(idx),) + tuple(data.shape[1:]), dtype=data.dtype)
    for j, i in enumerate(idx):
        stack[j] = data[int(i)]
    med = np.empty(data.shape[1:], dtype=data.dtype)
    rows = max(1, int(64e6 // (len(idx) * int(np.prod(data.shape[2:])) * data.dtype.itemsize * 2)))
    for r in range(0, data.shape[1], rows):
        med[r:r + rows] = _clip_round(np.median(stack[:, r:r + rows], axis=0), data.dtype)
    return med, len(idx)


def dark_background(dark, dtype):
    """Mean of the stored dark frames, rounded onto the raw dtype."""
    d = np.asarray(dark)
    if d.ndim == 3:
        d = d.mean(axis=0)
    return _clip_round(d, dtype)


def _candidates(data, dark, mode):
    """Yield ``(name, background, shift)``; ``raw`` is handled by the caller."""
    dtype = data.dtype
    frame_bytes = int(np.prod(data.shape[1:])) * dtype.itemsize
    k = int(max(2, min(TRIAL_FRAMES, MEMORY_BUDGET // (3 * max(frame_bytes, 1)))))
    sample = np.stack([data[int(i)] for i in _evenly(data.shape[0], k)])
    shift = common_shift(sample)
    bgs = []
    if mode in ("dark", "auto") and dark is not None and np.any(dark):
        bgs.append(("dark", dark_background(dark, dtype)))
    if mode in ("median", "auto"):
        med, _ = scan_median(data)
        bgs.append(("median", med))
    out = []
    for name, bg in bgs:
        out.append((name, bg, 0))
        if shift:
            out.append((name + "_q", quantize_background(bg, shift), shift))
    return sample, out


def choose_encoding(data, dark, mode="dark"):
    """Trial-encode a few frames and keep the smallest. Returns ``(name, background, shift, sizes)``.

    ``name`` is ``"raw"`` when nothing beats the raw frames by ``MIN_GAIN``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    comp = _make_compressor()
    sample, cands = _candidates(data, dark, mode)
    sizes = {"raw": sum(len(comp.encode(np.ascontiguousarray(f))) for f in sample)}
    best = ("raw", None, 0)
    for name, bg, shift in cands:
        sizes[name] = sum(len(comp.encode(zigzag_encode(f, bg))) for f in sample)
        if sizes[name] < sizes[best[0]] * (1.0 if best[0] != "raw" else 1.0 - MIN_GAIN):
            best = (name, bg, shift)
    return best[0], best[1], best[2], sizes


def _copy_tree(src, dst, skip, prefix=""):
    import zarr
    for name, arr in src.arrays():
        if prefix + name in skip:
            continue
        zarr.copy(arr, dst, name=name)
    for name, grp in src.groups():
        sub = dst.require_group(name)
        sub.attrs.update(dict(grp.attrs))
        _copy_tree(grp, sub, skip, prefix + name + "/")


def convert_zip(src, dst=None, mode="dark", chunk_frames=8, verify=True, log=print):
    """Re-encode ``src`` (an ordinary MIDAS zip) as an archive zip at ``dst``.

    ``dst=None`` replaces ``src``. Returns ``(name, sizes)``; with ``name == "raw"`` the frames
    are kept raw: ``src`` is untouched and, if ``dst`` is given, it receives a plain copy of
    ``src``. A failed conversion leaves no partial file. Every chunk is decoded again and compared with the source
    while writing (``verify``), so a bad encode cannot be saved.
    """
    import zarr
    src, dst = Path(src), Path(dst) if dst else None
    zin = zarr.open(zarr.ZipStore(str(src), mode="r"), mode="r")
    data = zin["exchange/data"]
    dark = zin["exchange/dark"][()] if "exchange/dark" in zin else None
    name, bg, shift, sizes = choose_encoding(data, dark, mode)
    log(f"archive: trial sizes {sizes} -> {name}")
    if name == "raw":
        if dst is not None:
            shutil.copyfile(src, dst)
        return name, sizes

    out = dst or src.with_name(src.name + ".archive.tmp")
    if out.exists():
        out.unlink()
    store = zarr.ZipStore(str(out), mode="w")
    try:
        zout = zarr.group(store=store, overwrite=True)
        N, Z, Y = data.shape
        _copy_tree(zin, zout, skip={"exchange/data"})
        exc = zout["exchange"]
        exc.create_dataset("background", data=bg.reshape(1, Z, Y), chunks=(1, Z, Y),
                           compressor=_make_compressor())
        res = exc.create_dataset("data_resid", shape=data.shape, dtype=_uns(data.dtype),
                                 chunks=(1, Z, Y), compressor=_make_compressor())
        res.attrs.update(encoding=ENCODING, background=name, shift=int(shift),
                         orig_dtype=str(data.dtype))
        for s0 in range(0, N, chunk_frames):
            raw = data[s0:s0 + chunk_frames]
            z = zigzag_encode(raw, bg)
            if verify and not np.array_equal(zigzag_decode(z, bg, data.dtype), raw):
                raise RuntimeError(f"archive: round trip failed in frames {s0}..{s0 + len(raw) - 1}")
            res[s0:s0 + len(raw)] = z
        store.close()
    except BaseException:
        try:
            store.close()
        finally:
            if out.exists():
                out.unlink()
        raise
    if dst is None:
        os.replace(out, src)
    return name, sizes


def read_frames(path_or_group, start=0, stop=None):
    """Raw frames ``[start:stop]`` from an archive zip, or from an ordinary zip."""
    import zarr
    z = path_or_group
    if isinstance(z, (str, os.PathLike)):
        z = zarr.open(zarr.ZipStore(str(z), mode="r"), mode="r")
    exc = z["exchange"]
    if "data" in exc:
        return exc["data"][start:stop]
    res = exc["data_resid"]
    if res.attrs.get("encoding") != ENCODING:
        raise ValueError(f"unknown archive encoding {res.attrs.get('encoding')!r}")
    return zigzag_decode(res[start:stop], exc["background"][0], res.attrs["orig_dtype"])


def restore_zip(src, dst, chunk_frames=8):
    """Write an ordinary MIDAS zip (with ``exchange/data``) from an archive zip."""
    import zarr
    src, dst = Path(src), Path(dst)
    zin = zarr.open(zarr.ZipStore(str(src), mode="r"), mode="r")
    res = zin["exchange/data_resid"]
    if dst.exists():
        dst.unlink()
    store = zarr.ZipStore(str(dst), mode="w")
    zout = zarr.group(store=store, overwrite=True)
    _copy_tree(zin, zout, skip={"exchange/data_resid", "exchange/background"})
    N, Z, Y = res.shape
    dt = np.dtype(res.attrs["orig_dtype"])
    out = zout["exchange"].create_dataset("data", shape=res.shape, dtype=dt,
                                          chunks=(1, Z, Y), compressor=_make_compressor())
    for s in range(0, N, chunk_frames):
        out[s:s + chunk_frames] = read_frames(zin, s, s + chunk_frames)
    store.close()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Lossless background-subtracted ('archive') encoding of a MIDAS zip.")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("convert", help="ordinary zip -> archive zip")
    a.add_argument("src")
    a.add_argument("dst", nargs="?", help="default: replace src")
    a.add_argument("--mode", choices=MODES, default="dark",
                   help="background candidates: stored dark, scan median, or both (auto)")
    b = sub.add_parser("restore", help="archive zip -> ordinary zip")
    b.add_argument("src")
    b.add_argument("dst")
    args = p.parse_args(argv)
    if args.cmd == "convert":
        convert_zip(args.src, args.dst, mode=args.mode)
    else:
        restore_zip(args.src, args.dst)


if __name__ == "__main__":
    sys.exit(main())

"""Read APS 6-ID-C DFXM rocking scans into a :class:`midas_dfxm.rocking.RockingScan`.

Two acquisition layouts are known, and each has already produced a silently wrong answer.

**Named frames + ``S###_motorInfo.csv``** (2021-2023, e.g. NaMnO2 and CsV3Sb5). One TIFF per
scan point in ``S###/``, angles also written into the filename, one motor-table row per
frame. Traps met on real data:

* The NaMnO2 Dryad deposit ships two sets of logs with identical file names. Its ``motors/``
  folder is the **July 2021** beamtime (50 rows per frame, 10 mdeg steps, 2theta 22.874);
  the December 2021 frames need ``Plumb_DFXM_Dec2021_motors`` (41 rows, 5 mdeg, 2theta
  30.250). The July log doubles every tilt. Rows must equal frames, so it is refused here.
* A filename-pattern sort once matched a constant field (``_10_``) and left the frames in
  directory-listing order: no error, a scrambled map. Here frames are ordered by natural
  filename sort and then **cross-checked** against the motor log wherever a filename field
  carries the scanned angle.

**Indexed frames + ``<prefix>_motor_information_S<n>_<exp>s.csv``** (Dec 2025 onward).
``scans_raw/DFXM_S<n>/data_NNNNN.tif`` with R repeats per point, and a motor table with one
row per POINT (``Num, tth, th, chi, phi, ...``). Traps met on real data:

* R is not constant (20 at 0.07-0.1 s, 2 at 0.5 s): it is ``n_frames / n_points``.
* A partial transfer can leave ``n_frames // n_points`` "working" while reading the wrong
  frames; here the count must divide exactly and the indices must be contiguous.
* The first repeat of each point read +2.5-4.3 % hot at 20 repeats (not at 2); the reader
  measures it and says what it did.
* A zero-byte TIFF exists; it is refused by name.
* Frames are point-major (``index = point * R + repeat``); ``order="repeat"`` exists so
  :func:`midas_dfxm.check_frame_order` can compare the two.

Every reader refuses to guess, and writes what it did into ``scan.notes``.
"""
from __future__ import annotations

import csv
import glob
import os
import re
from typing import Optional

import numpy as np

from .rocking import RockingScan, classify_motors

__all__ = ["read_motor_table", "find_motor_tables", "load_6idc_scan"]

_INDEXED = re.compile(r"^data_(\d+)\.tiff?$", re.IGNORECASE)
_TIFF = re.compile(r"\.tiff?$", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
HC_KEV_A = 12.398419843320026
D_SI111_A = 3.135601


def read_motor_table(path: str) -> dict:
    """A motor CSV as ``{column: float array}``. Non-numeric cells become NaN.

    Blank rows are dropped; columns with no finite value are dropped.
    """
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError(f"{path}: no data rows")
    header = [h.strip() for h in rows[0]]
    out = {}
    for j, name in enumerate(header):
        vals = []
        for r in rows[1:]:
            try:
                vals.append(float(r[j]))
            except (IndexError, ValueError):
                vals.append(np.nan)
        v = np.asarray(vals, dtype=float)
        if np.isfinite(v).any() and name:
            out[name] = v
    return out


def _scan_number(path: str) -> Optional[int]:
    m = re.search(r"S(\d+)(?!.*S\d)", os.path.basename(os.path.normpath(path)))
    return int(m.group(1)) if m else None


def find_motor_tables(scan_dir: str, root: Optional[str] = None) -> list:
    """Motor tables that name this scan's number, near the scan folder.

    Looks under ``root`` (default: two levels above ``scan_dir``) for ``S###_motorInfo.csv``
    and ``*_motor_information_S<n>_*.csv``, at shallow depth and inside folders whose name
    contains ``motor``. More than one hit is common and is the trap: pick the table from the
    same beamtime as the frames.
    """
    n = _scan_number(scan_dir)
    if n is None:
        raise ValueError(f"no scan number (S<digits>) in {scan_dir!r}")
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(scan_dir)))
    pat = re.compile(rf"^(S0*{n}_motorInfo|.+_motor_information_S0*{n}_[\d.]+s)\.csv$")
    hits = set()
    for g in ("*.csv", "*/*.csv", "*motor*/**/*.csv", "*/*motor*/**/*.csv"):
        for f in glob.glob(os.path.join(root, g), recursive=True):
            if pat.match(os.path.basename(f)):
                hits.add(os.path.abspath(f))
    return sorted(hits)


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _tiff_reader():
    try:
        import tifffile
    except ImportError as e:                                    # pragma: no cover
        raise ImportError("reading TIFF frames needs `tifffile` (pip install tifffile)") from e
    return tifffile.imread


def _refuse_empty(paths):
    empty = [p for p in paths if os.path.getsize(p) == 0]
    if empty:
        raise ValueError(f"{len(empty)} zero-byte frame file(s), e.g. {empty[:3]}. The scan is "
                         "incomplete; re-transfer it rather than reading around the gap.")


def _load_dark(dark, roi, reader):
    if dark is None:
        return 0.0
    if isinstance(dark, (int, float, np.floating, np.integer)):
        return float(dark)
    if isinstance(dark, str):
        if os.path.isdir(dark):
            files = sorted((f for f in os.listdir(dark) if _TIFF.search(f)), key=_natural_key)
            if not files:
                raise ValueError(f"no TIFF frames in dark folder {dark}")
            acc = None
            for f in files:
                a = np.asarray(reader(os.path.join(dark, f)), dtype=np.float64)
                acc = a if acc is None else acc + a
            arr = acc / len(files)
        else:
            arr = np.asarray(reader(dark), dtype=np.float64)
    else:
        arr = np.asarray(dark, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"dark must be 2-D; got {arr.shape}")
    return _crop(arr, roi).astype(np.float32)


def _crop(a, roi):
    if roi is None:
        return a
    r0, r1, c0, c1 = roi
    return a[..., r0:r1, c0:c1]


def _scanned_axis(table: dict) -> Optional[str]:
    try:
        return classify_motors(table)["axes"][0]
    except ValueError:
        return None


def _crosscheck_filenames(files, table, notes):
    """Order named frames by natural sort, then confirm (or fix) it against the motor log."""
    axis = _scanned_axis(table)
    if axis is None:
        notes.append("no scanned angle found to cross-check the file order against")
        return files
    m = table[axis]
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    tokens = [_NUMBER.findall(s) for s in stems]
    if len({len(t) for t in tokens}) != 1 or not tokens[0]:
        notes.append(f"WARNING: filenames do not share a numeric pattern; file order is the "
                     f"natural filename order, NOT confirmed against {axis}. Run "
                     "check_frame_order.")
        return files
    # Only an EXACT match to the logged angle confirms anything. A field that is merely linear
    # in the motor confirms nothing: after a natural sort, whatever field drives the sort is
    # monotone, and a uniformly stepped motor is monotone too (a first version accepted that
    # as corroboration and never looked at the field that did carry the angle).
    exact = permuted = None
    linear = []
    for j in range(len(tokens[0])):
        col = [t[j] for t in tokens]
        vals = np.asarray([float(c) for c in col])
        dec = max((len(c.split(".")[1]) if "." in c else 0) for c in col)
        tol = 0.5 * 10.0 ** (-dec) + 1e-9
        if np.ptp(vals) == 0:
            continue
        if np.max(np.abs(vals - m)) <= tol:
            exact = (j, float(np.max(np.abs(vals - m))))
            break
        if (permuted is None and np.allclose(np.sort(vals), np.sort(m), atol=tol, rtol=0)
                and len(np.unique(np.round(vals / tol))) == len(vals)):
            permuted = (j, vals)
        elif np.std(m) > 0 and abs(np.corrcoef(vals, m)[0, 1]) > 0.99999:
            linear.append(j)
    if exact is not None:
        notes.append(f"file order confirmed: filename field {exact[0]} equals {axis} row by "
                     f"row (max |difference| {exact[1]:.1e} deg)")
        return files
    if permuted is not None:
        j, vals = permuted
        order = np.argsort(vals, kind="stable")[np.argsort(np.argsort(m, kind="stable"))]
        notes.append(f"files REORDERED so that filename field {j} equals {axis} row by row "
                     "(the natural filename order did not)")
        return [files[i] for i in order]
    msg = (f"WARNING: no filename field equals {axis}; file order is the natural filename "
           "order, NOT confirmed against the motor log.")
    if linear:
        msg += (f" Field(s) {linear} are linear in {axis}, but natural sorting makes any "
                "monotone field look that way, so that confirms nothing.")
    notes.append(msg + " Run check_frame_order.")
    return files


def _load_named(scan_dir, table_path, table, roi, dark, reader, notes):
    files = sorted((os.path.join(scan_dir, f) for f in os.listdir(scan_dir) if _TIFF.search(f)),
                   key=lambda p: _natural_key(os.path.basename(p)))
    if not files:
        raise FileNotFoundError(f"no TIFF frames in {scan_dir}")
    _refuse_empty(files)
    n_rows = len(next(iter(table.values())))
    if n_rows != len(files):
        per = f" ({n_rows / len(files):g} per frame)" if n_rows % len(files) == 0 else ""
        raise ValueError(
            f"{os.path.basename(table_path)} has {n_rows} rows for {len(files)} frames{per}. "
            "One row per frame is required. A table with another row count belongs to another "
            "scan or another beamtime (the NaMnO2 Dryad 'motors/' folder is the July 2021 "
            "beamtime; the December 2021 frames need 'Plumb_DFXM_Dec2021_motors'). Refusing "
            "to guess the frame/angle pairing.")
    files = _crosscheck_filenames(files, table, notes)
    first = np.asarray(reader(files[0]))
    shape = _crop(first, roi).shape
    frames = np.empty((len(files),) + shape, dtype=np.float32)
    for i, f in enumerate(files):
        frames[i] = _crop(np.asarray(reader(f)), roi)
    frames -= dark
    return frames, None, 1, {"files": [os.path.basename(f) for f in files]}


def _load_indexed(scan_dir, table_path, table, roi, dark, reader, notes, *, order,
                  drop_first_repeat, hot_threshold, frames_per_point, halves):
    names = [f for f in os.listdir(scan_dir) if _INDEXED.match(f)]
    if not names:
        raise FileNotFoundError(f"no data_NNNNN.tif frames in {scan_dir}")
    index = {int(_INDEXED.match(f).group(1)): os.path.join(scan_dir, f) for f in names}
    idx = sorted(index)
    first = idx[0]
    missing = sorted(set(range(first, first + len(idx))) - set(idx))
    if missing or idx[-1] != first + len(idx) - 1:
        gaps = sorted(set(range(first, idx[-1] + 1)) - set(idx))
        raise ValueError(f"{scan_dir}: frame indices are not contiguous ({len(gaps)} missing, "
                         f"e.g. {gaps[:10]}). Files were lost in transfer; re-transfer the scan.")
    paths = [index[i] for i in idx]
    _refuse_empty(paths)
    n_points = len(next(iter(table.values())))
    n = len(paths)
    if frames_per_point is None:
        if n % n_points:
            raise ValueError(
                f"{n} frames for {n_points} motor-table points is not a whole number of "
                "repeats: frames are missing, or the table belongs to another scan. "
                "n_frames // n_points would read the wrong frames without an error.")
        R = n // n_points
    else:
        R = int(frames_per_point)
        if R * n_points != n:
            raise ValueError(f"frames_per_point={R} x {n_points} points != {n} frames")
    if "Num" in table and not np.array_equal(table["Num"], np.arange(n_points)):
        notes.append("WARNING: the table's Num column is not 0..n_points-1")
    if order not in ("point", "repeat"):
        raise ValueError("order must be 'point' (point-major) or 'repeat' (repeat-major)")

    def frame_path(pt, r):
        return paths[pt * R + r] if order == "point" else paths[r * n_points + pt]

    shape = _crop(np.asarray(reader(paths[0])), roi).shape
    rep0 = np.zeros((n_points,) + shape, np.float32)
    odd = np.zeros_like(rep0)            # repeats 1, 3, 5, ...
    even = np.zeros_like(rep0)           # repeats 2, 4, 6, ...
    level = np.zeros(R)
    for pt in range(n_points):
        for r in range(R):
            a = _crop(np.asarray(reader(frame_path(pt, r)), dtype=np.float32), roi) - dark
            level[r] += float(a.mean())
            if r == 0:
                rep0[pt] = a
            elif r % 2:
                odd[pt] += a
            else:
                even[pt] += a
    n_odd, n_even = R // 2, (R - 1) // 2
    hot = (level[0] / np.median(level[1:]) - 1.0) if R >= 2 else 0.0
    offset = None
    if R >= 2:
        # Is the first repeat's excess additive (a detector offset on every pixel, dark ones
        # included) or multiplicative (more signal)? A whole-frame ratio cannot tell.
        rep0_mean = rep0.mean(0)
        others = (odd + even).mean(0) / (R - 1)
        dim = others <= np.percentile(others, 20)
        bright = others >= np.percentile(others, 80)
        offset = float(np.median((rep0_mean - others)[dim]))
        with np.errstate(invalid="ignore", divide="ignore"):
            gain_bright = float(np.nanmedian(rep0_mean[bright] / others[bright])) - 1.0
        notes.append(f"first repeat vs the others, per pixel: {offset:+.2f} counts on the dimmest "
                     f"fifth of the ROI (additive offset), {100 * gain_bright:+.2f} % on the "
                     "brightest fifth")
    if drop_first_repeat == "auto":
        # With 2 repeats, dropping one leaves no repeat halves and so no repeat check: a
        # misread scan then passed on the adjacent-frame test alone.
        drop = R >= 3 and hot > hot_threshold
    else:
        drop = bool(drop_first_repeat)
    if R >= 2:
        why = ""
        if drop_first_repeat == "auto":
            why = (f", threshold {100 * hot_threshold:.1f} %" if R >= 3
                   else "; with 2 repeats it is never dropped automatically")
        notes.append(f"first repeat reads {100 * hot:+.2f} % against the other repeats "
                     f"({'DROPPED' if drop else 'kept'}{why})")
    kept = R - 1 if drop else R
    if kept < 1:
        raise ValueError("no repeats left after dropping the first")
    frames = ((odd + even + (0 if drop else rep0)) / kept).astype(np.float32)
    split = None
    if halves and kept >= 2:
        if drop:
            A, nA, B, nB = odd, n_odd, even, n_even
        else:
            A, nA, B, nB = rep0 + even, 1 + n_even, odd, n_odd
        if nA >= 1 and nB >= 1:
            split = (A / nA, B / nB)
            if nA != nB:
                notes.append(f"repeat-parity halves hold {nA} and {nB} repeats")
    notes.append(f"{n} frames = {n_points} points x {R} repeats, read {order}-major; "
                 f"{kept} repeat(s) averaged per point")
    return frames, split, kept, {"repeats_per_point": R, "first_repeat_excess": hot,
                                 "first_repeat_offset": offset,
                                 "first_repeat_dropped": drop, "order": order}


def load_6idc_scan(scan_dir: str, motor_table: Optional[str] = None, *, roi=None, dark=None,
                   layout: str = "auto", order: str = "point",
                   drop_first_repeat="auto", hot_threshold: float = 0.01,
                   frames_per_point: Optional[int] = None, halves: bool = True) -> RockingScan:
    """Load one 6-ID-C rocking scan, refusing every ambiguity that has bitten before.

    Parameters
    ----------
    scan_dir : folder holding the scan's TIFF frames (``.../S006`` or ``.../DFXM_S190``).
    motor_table : the scan's motor CSV. If omitted, :func:`find_motor_tables` looks for it
        and the load stops unless exactly one table is found.
    roi : ``(row0, row1, col0, col1)`` crop applied as frames are read. Use one: a full
        Zyla frame is 2560 x 2160 px, so 61 points need ~1.4 GB as float32.
    dark : ``None``, a number, a 2-D array, a TIFF path or a folder of dark frames; subtracted
        from every frame and kept in ``scan.dark_level`` for the error model. Match its
        exposure to the scan's.
    layout : ``"auto"``, ``"named"`` (2021-2023) or ``"indexed"`` (2025).
    order, drop_first_repeat, hot_threshold, frames_per_point, halves
        Indexed layout only. ``drop_first_repeat="auto"`` drops repeat 0 when it reads more
        than ``hot_threshold`` above the median of the others. ``halves`` keeps the
        even/odd-repeat averages for a photon-noise split-half error bar.

    Returns
    -------
    RockingScan
        Frames in acquisition order with their motor readings; ``scan.notes`` lists every
        check and decision. Read ``scan.summary()`` before reducing.
    """
    reader = _tiff_reader()
    scan_dir = os.path.abspath(scan_dir)
    if not os.path.isdir(scan_dir):
        raise FileNotFoundError(scan_dir)
    notes = []
    if motor_table is None:
        found = find_motor_tables(scan_dir)
        if len(found) != 1:
            raise ValueError(
                f"found {len(found)} motor tables for {os.path.basename(scan_dir)}: {found}. "
                + ("Pass motor_table= explicitly -- and pick the one from the SAME beamtime as "
                   "the frames." if found else "Pass motor_table= explicitly."))
        motor_table = found[0]
        notes.append(f"motor table found automatically: {motor_table}")
    table = read_motor_table(motor_table)
    if layout == "auto":
        layout = "indexed" if any(_INDEXED.match(f) for f in os.listdir(scan_dir)) else "named"
    dark_arr = _load_dark(dark, roi, reader)
    if layout == "named":
        frames, split, R, meta = _load_named(scan_dir, motor_table, table, roi, dark_arr,
                                             reader, notes)
    elif layout == "indexed":
        frames, split, R, meta = _load_indexed(
            scan_dir, motor_table, table, roi, dark_arr, reader, notes, order=order,
            drop_first_repeat=drop_first_repeat, hot_threshold=hot_threshold,
            frames_per_point=frames_per_point, halves=halves)
    else:
        raise ValueError("layout must be 'auto', 'named' or 'indexed'")
    if "mono" in table:
        mono = np.nanmedian(table["mono"])
        e = HC_KEV_A / (2.0 * D_SI111_A * np.sin(np.radians(mono)))
        meta["energy_keV_si111"] = float(e)
        notes.append(f"energy {e:.3f} keV from mono = {mono:.6f} deg, ASSUMING a Si(111) "
                     "monochromator: confirm against the beamline record")
    ct = os.path.join(scan_dir, "creation_times.npy")
    if os.path.isfile(ct):
        t = np.load(ct)
        meta["creation_times_ns"] = t
        notes.append(f"creation_times.npy: {t.size} entries (one per INTENDED frame; index by "
                     "file number, not position)")
    meta.update(scan_dir=scan_dir, motor_table=os.path.abspath(motor_table), layout=layout,
                roi=roi)
    dark_level = dark_arr if np.ndim(dark_arr) else float(dark_arr)
    return RockingScan.from_arrays(frames, table, halves=split, n_repeats=R,
                                   dark_level=dark_level,
                                   source=f"{scan_dir} + {os.path.basename(motor_table)}",
                                   notes=notes, meta=meta)

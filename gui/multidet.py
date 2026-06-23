#!/usr/bin/env python3
"""Multi-detector compositing helpers for the FF viewer.

Pure functions (no Qt) for parsing per-detector MIDAS-style param files,
remapping each detector image into a common BigDet composite frame, and
combining them via max/sum. Designed so the per-detector inverse coordinate
maps can be cached — geometry only changes when BC, tx, px, or BigDetSize
changes; per-frame work is just an HDF5 read + scipy.ndimage.map_coordinates
+ a pixel-wise reduction.

Coordinate convention matches MIDAS lab frame (matches ff_asym_qt cursor
handler and the lab-axes overlay):
    +Y → display LEFT, +Z → display UP, +X → into page.
The composite has its origin (Y=Z=0) at (BigDetSize/2, BigDetSize/2).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Register bundled HDF5 compression filters (Blosc, LZ4, Bitshuffle, Zstd)
# before h5py is used; detector files often need these and libhdf5's
# default plugin path is rarely populated.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.ndimage import map_coordinates
except ImportError:
    map_coordinates = None


DEFAULT_DATA_LOC = '/exchange/data'
DEFAULT_DARK_LOC = '/exchange/data_dark'


_seen_warnings: set[str] = set()


def _warn_once(msg: str) -> None:
    """Print msg to stdout the first time it's seen this session.

    Used by the dark-subtraction path to surface why a particular detector's
    dark didn't apply (missing dataset, shape mismatch, etc.) without
    flooding the log on every composite. Call ``reset_warn_once()`` to
    re-arm — useful after the user changes paths.
    """
    if msg not in _seen_warnings:
        _seen_warnings.add(msg)
        print(msg)


def reset_warn_once() -> None:
    _seen_warnings.clear()


# ── Param file parsing ──────────────────────────────────────────────────────

def parse_detector_param_file(fn: str) -> dict:
    """Parse a MIDAS-style per-detector param file.

    Recognized keys (all optional, sensible defaults):
        BC               two floats (Y, Z) — beam center pixels
        Lsd              float (μm)
        tx, ty, tz       floats (degrees), tilts; tx is rotation about beam
        px / PixelSize   float (μm)
        NrPixels         int (square)
        NrPixelsY        int
        NrPixelsZ        int
        ImTransOpt       repeatable; list of int codes
                         1=HFlip, 2=VFlip, 3=Transpose
        dataLoc          str — HDF5 dataset path for frames
        darkLoc          str — HDF5 dataset path for dark
        BadPxIntensity   float — pixel ≥ this → masked (NaN)
        GapIntensity     float — pixel ≥ this → masked (NaN)
        Wavelength, SpaceGroup, LatticeConstant, MaxRingRad — for shared use
    """
    raw: dict[str, list[list[str]]] = {}
    with open(fn, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip().rstrip(';').strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            raw.setdefault(tokens[0], []).append(tokens[1:])

    def first(*keys):
        for k in keys:
            if k in raw:
                return raw[k][0]
        return None

    def get_float(*keys):
        v = first(*keys)
        try:
            return float(v[0]) if v else None
        except (ValueError, IndexError):
            return None

    def get_int(*keys):
        v = first(*keys)
        try:
            return int(float(v[0])) if v else None
        except (ValueError, IndexError):
            return None

    def get_str(*keys):
        v = first(*keys)
        return v[0] if v else None

    out: dict = {
        'bc_y': None, 'bc_z': None,
        'lsd': None,
        'tx': 0.0, 'ty': 0.0, 'tz': 0.0,
        'px': None,
        'ny': None, 'nz': None,
        'im_trans_opts': [],
        'data_loc': DEFAULT_DATA_LOC,
        'dark_loc': DEFAULT_DARK_LOC,
        'dark_file': None,             # external dark HDF5 path, if specified
        'bad_px': None,
        'gap_px': None,
        # Raw GE binary parameters (only used when data_file is .geX/.dat/etc.).
        'header_size': None,
        'bytes_per_pixel': None,
        'wavelength': None, 'space_group': None,
        'lattice_constant': None, 'max_ring_rad': None,
    }

    bc = first('BC')
    if bc and len(bc) >= 2:
        try:
            out['bc_y'] = float(bc[0]); out['bc_z'] = float(bc[1])
        except ValueError:
            pass
    if out['bc_y'] is None:
        out['bc_y'] = get_float('YCen')
        out['bc_z'] = get_float('ZCen')

    out['lsd'] = get_float('Lsd')
    out['tx'] = get_float('tx') or 0.0
    out['ty'] = get_float('ty') or 0.0
    out['tz'] = get_float('tz') or 0.0
    out['px'] = get_float('px', 'PixelSize')

    npx = get_int('NrPixels')
    if npx is not None:
        out['ny'] = out['nz'] = npx
    npy = get_int('NrPixelsY')
    npz = get_int('NrPixelsZ')
    if npy is not None: out['ny'] = npy
    if npz is not None: out['nz'] = npz

    if 'ImTransOpt' in raw:
        for line_vals in raw['ImTransOpt']:
            for v in line_vals:
                try:
                    out['im_trans_opts'].append(int(v))
                except ValueError:
                    pass

    out['data_loc'] = _normalize_h5_path(get_str('dataLoc')) or DEFAULT_DATA_LOC
    out['dark_loc'] = _normalize_h5_path(get_str('darkLoc')) or DEFAULT_DARK_LOC
    # h5py is forgiving with trailing slashes but our `path in file` lookups
    # are stricter — strip them.
    if out['data_loc'].endswith('/') and out['data_loc'] != '/':
        out['data_loc'] = out['data_loc'].rstrip('/')
    if out['dark_loc'].endswith('/') and out['dark_loc'] != '/':
        out['dark_loc'] = out['dark_loc'].rstrip('/')

    # External dark file path (from `Dark <path>` line in MIDAS param files).
    dk = get_str('Dark', 'DarkStem')
    if dk and (os.sep in dk or '/' in dk):
        out['dark_file'] = dk

    out['bad_px'] = get_float('BadPxIntensity')
    out['gap_px'] = get_float('GapIntensity')

    out['header_size']     = get_int('HeadSize', 'HeaderSize')
    out['bytes_per_pixel'] = get_int('BytesPerPixel')

    out['wavelength']  = get_float('Wavelength')
    out['space_group'] = get_int('SpaceGroup', 'SpaceGroupNumber')
    lc = first('LatticeConstant', 'LatticeParameter')
    if lc and len(lc) >= 6:
        try:
            out['lattice_constant'] = [float(x) for x in lc[:6]]
        except ValueError:
            pass
    out['max_ring_rad'] = get_float('MaxRingRad', 'RhoD')

    return out


def _normalize_h5_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return p if p.startswith('/') else '/' + p


# ── HDF5 / raw-binary IO ────────────────────────────────────────────────────

_HDF5_EXTS = ('.h5', '.hdf', '.hdf5', '.nxs')


def _is_hdf5(fn: str) -> bool:
    return os.path.splitext(fn)[1].lower() in _HDF5_EXTS


def _raw_n_frames(fn: str, header: int, bpp: int, ny: int, nz: int) -> int:
    """Frame count for a raw binary stack (file size minus header / frame
    bytes). Returns 0 on any error so the caller treats it as empty."""
    try:
        size = os.path.getsize(fn)
    except OSError:
        return 0
    frame_bytes = bpp * ny * nz
    if frame_bytes <= 0 or size <= header:
        return 0
    return max(1, (size - header) // frame_bytes)


def _raw_read_frame(fn: str, header: int, bpp: int, ny: int, nz: int,
                    frame_idx: int) -> Optional[np.ndarray]:
    try:
        dtype = np.uint16 if bpp == 2 else np.int32
        offset = header + frame_idx * (bpp * ny * nz)
        with open(fn, 'rb') as f:
            f.seek(offset, os.SEEK_SET)
            data = np.fromfile(f, dtype=dtype, count=ny * nz)
        if data.size != ny * nz:
            return None
        return data.reshape((ny, nz)).astype(np.float32)
    except OSError:
        return None


def n_frames_in_h5(fn: str, loc: str,
                   header: int = 8192, bpp: int = 2,
                   ny: int = 2048, nz: int = 2048) -> int:
    """Return frame count for `fn`. HDF5 if extension matches, otherwise
    raw GE binary (header + bpp×ny×nz per frame). The raw kwargs are
    ignored for HDF5 files."""
    if not fn:
        return 0
    if _is_hdf5(fn):
        if h5py is None:
            return 0
        try:
            with h5py.File(fn, 'r') as f:
                ds = f.get(_normalize_h5_path(loc))
                if ds is None:
                    return 0
                return int(ds.shape[0]) if ds.ndim >= 3 else 1
        except OSError:
            return 0
    return _raw_n_frames(fn, header, bpp, ny, nz)


def read_h5_frame(fn: str, loc: str, frame_idx: int,
                  header: int = 8192, bpp: int = 2,
                  ny: int = 2048, nz: int = 2048) -> Optional[np.ndarray]:
    """Read one frame from `fn` as float32 2-D. HDF5 by extension, otherwise
    raw GE binary."""
    if not fn:
        return None
    if _is_hdf5(fn):
        if h5py is None:
            return None
        try:
            with h5py.File(fn, 'r') as f:
                ds = f[_normalize_h5_path(loc)]
                if ds.ndim >= 3:
                    return ds[frame_idx].astype(np.float32)
                return ds[...].astype(np.float32)
        except (OSError, KeyError, IndexError):
            return None
    return _raw_read_frame(fn, header, bpp, ny, nz, frame_idx)


def read_h5_dark(fn: str, loc: str,
                 header: int = 8192, bpp: int = 2,
                 ny: int = 2048, nz: int = 2048) -> Optional[np.ndarray]:
    """Read dark frames and average them; returns float32 2-D or None.
    Averages all frames for HDF5 stacks, or all frames in the raw binary
    stack."""
    if not fn:
        return None
    if _is_hdf5(fn):
        if h5py is None:
            return None
        try:
            with h5py.File(fn, 'r') as f:
                ds = f.get(_normalize_h5_path(loc))
                if ds is None:
                    return None
                data = ds[...].astype(np.float32)
                return np.mean(data, axis=0) if data.ndim >= 3 else data
        except (OSError, KeyError):
            return None
    n = _raw_n_frames(fn, header, bpp, ny, nz)
    if n <= 0:
        return None
    acc = None
    for i in range(n):
        f = _raw_read_frame(fn, header, bpp, ny, nz, i)
        if f is None:
            return None
        acc = f if acc is None else acc + f
    return acc / float(n)


# ── Image preprocessing (matches ff_asym_qt convention) ─────────────────────

def apply_imtransopt(img: np.ndarray, opts: list[int]) -> np.ndarray:
    """Apply MIDAS DoImageTransformations codes in order.

    1 = HFlip (cols), 2 = VFlip (rows), 3 = Transpose. 0 = no-op.
    """
    for opt in opts:
        if opt == 1:
            img = img[:, ::-1]
        elif opt == 2:
            img = img[::-1, :]
        elif opt == 3:
            img = img.T
    return np.ascontiguousarray(img)


def mask_sentinels(img: np.ndarray,
                   bad_px: Optional[float],
                   gap_px: Optional[float]) -> np.ndarray:
    """Replace BadPx / Gap-Intensity sentinel pixels with NaN."""
    if bad_px is None and gap_px is None:
        return img
    out = img.astype(np.float32, copy=True)
    if bad_px is not None and np.isfinite(bad_px):
        out[out >= bad_px] = np.nan
    if gap_px is not None and np.isfinite(gap_px):
        out[out >= gap_px] = np.nan
    return out


# ── Geometry: inverse coordinate maps ───────────────────────────────────────

def compute_inv_coords(bc_y: float, bc_z: float, tx_deg: float,
                       big_det_size: int, px: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute inverse-mapping coordinate grids for one detector.

    For composite output pixel (Yo, Zo) — where Yo is column index (Y axis)
    and Zo is row index (Z axis) in MIDAS convention — return the detector
    pixel (Y_pix, Z_pix) that should be sampled.

    Math (matches `_apply_tx_rotation` sign convention in ff_asym_qt):
        Y_lab = (Yo − BigDetSize/2) · px
        Z_lab = (Zo − BigDetSize/2) · px
        Y_det =  Y_lab·cos(tx) + Z_lab·sin(tx)
        Z_det = −Y_lab·sin(tx) + Z_lab·cos(tx)
        Y_pix = BCy − Y_det / px
        Z_pix = BCz + Z_det / px

    Returns
    -------
    y_pix, z_pix : ndarray, shape (BigDetSize, BigDetSize)
        Per-output-pixel detector-pixel coordinates suitable for
        ``scipy.ndimage.map_coordinates`` (with the (row, col) order).

    Notes
    -----
    Detector image is shape (ny, nz). For map_coordinates we need
    [row_indices, col_indices] where row → first axis (ny). We treat
    Z (vertical) as rows and Y (horizontal) as cols, matching the
    rest of the FF viewer.
    """
    bds = int(big_det_size)
    half = bds * 0.5
    tx_rad = math.radians(tx_deg)
    c, s = math.cos(tx_rad), math.sin(tx_rad)

    # Output pixel indices: yo (cols), zo (rows). Build with meshgrid.
    yo = np.arange(bds, dtype=np.float32)
    zo = np.arange(bds, dtype=np.float32)
    Yo, Zo = np.meshgrid(yo, zo)        # shape (bds, bds): Yo[r,c] = c, Zo[r,c] = r

    Y_lab = (Yo - half) * px
    Z_lab = (Zo - half) * px

    Y_det =  Y_lab * c + Z_lab * s
    Z_det = -Y_lab * s + Z_lab * c

    y_pix = bc_y - Y_det / px           # detector column index (Y axis)
    z_pix = bc_z + Z_det / px           # detector row index    (Z axis)

    # map_coordinates wants axes in the order data array indices: (row, col)
    # data[row, col] = data[z_pix, y_pix] in MIDAS convention.
    return z_pix.astype(np.float32), y_pix.astype(np.float32)


def remap_to_composite(image: np.ndarray,
                       row_coords: np.ndarray,
                       col_coords: np.ndarray,
                       cval: float = np.nan) -> np.ndarray:
    """Inverse-warp `image` into the composite frame using cached coords.

    Out-of-bounds composite pixels are filled with `cval` (NaN by default
    so they are excluded from max/sum reductions via nanmax / nansum).
    Bilinear interpolation (order=1).
    """
    if map_coordinates is None:
        raise ImportError("scipy is required for multi-detector compositing")
    return map_coordinates(image,
                           [row_coords, col_coords],
                           order=1, mode='constant', cval=cval,
                           prefilter=False)


# ── Compositing ─────────────────────────────────────────────────────────────

def composite(images: list[np.ndarray], op: str = 'max') -> np.ndarray:
    """Reduce a stack of remapped detector images into one composite array.

    NaN pixels are treated as "not observed" by this detector so they do not
    drag the max down or invalidate the sum. If every detector is NaN at a
    given pixel, the result is 0 (so the displayed image has no spurious
    NaN holes; intensity overlay levels stay sensible).
    """
    if not images:
        raise ValueError("composite() needs at least one image")
    stack = np.stack(images, axis=0)        # shape (N, bds, bds)
    if op == 'sum':
        out = np.nansum(stack, axis=0)
    else:                                    # default: max
        # nanmax warns "All-NaN slice" for pixels no detector covers.
        # Suppress — those are legitimately empty composite pixels.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            out = np.nanmax(stack, axis=0)
        out = np.where(np.isnan(out), 0.0, out)
    return out.astype(np.float32)


# ── Per-detector state object ───────────────────────────────────────────────

@dataclass
class DetectorState:
    """In-memory state for one detector slot in multi-mode."""
    enabled: bool = True
    data_file: str = ''
    dark_file: str = ''
    param_file: str = ''

    # From param file (None until loaded):
    bc_y: float = 0.0
    bc_z: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    lsd: float = 0.0
    px: float = 200.0
    ny: int = 2048
    nz: int = 2048
    im_trans_opts: list = field(default_factory=list)
    data_loc: str = DEFAULT_DATA_LOC
    dark_loc: str = DEFAULT_DARK_LOC
    bad_px: Optional[float] = None
    gap_px: Optional[float] = None
    # Raw GE binary defaults (overridden by HeadSize / BytesPerPixel in
    # param file). Mirrors FFViewer.header_size / bytes_per_pixel for the
    # single-detector path.
    header_size: int = 8192
    bytes_per_pixel: int = 2

    # Caches:
    _dark_image: Optional[np.ndarray] = None
    _dark_cache_key: tuple = ()
    _inv_coords: Optional[tuple] = None
    _inv_cache_key: tuple = ()

    def has_files(self) -> bool:
        return bool(self.data_file) and bool(self.param_file)

    def n_frames(self) -> int:
        return n_frames_in_h5(self.data_file, self.data_loc,
                              self.header_size, self.bytes_per_pixel,
                              self.ny, self.nz)

    def load_param_file(self, fn: str) -> None:
        """Read fn, populate per-detector geometry fields.

        Returns the parsed dict so the caller can extract shared params
        (Wavelength, SpaceGroup, etc.) for the GUI.
        """
        params = parse_detector_param_file(fn)
        self.param_file = fn
        if params['bc_y'] is not None: self.bc_y = params['bc_y']
        if params['bc_z'] is not None: self.bc_z = params['bc_z']
        self.tx = params['tx']
        self.ty = params['ty']
        self.tz = params['tz']
        if params['lsd'] is not None: self.lsd = params['lsd']
        if params['px']  is not None: self.px  = params['px']
        if params['ny']  is not None: self.ny  = params['ny']
        if params['nz']  is not None: self.nz  = params['nz']
        self.im_trans_opts = list(params['im_trans_opts'])
        self.data_loc = params['data_loc']
        self.dark_loc = params['dark_loc']
        self.bad_px = params['bad_px']
        self.gap_px = params['gap_px']
        if params['header_size']     is not None: self.header_size     = params['header_size']
        if params['bytes_per_pixel'] is not None: self.bytes_per_pixel = params['bytes_per_pixel']
        # External dark from param file's `Dark <path>` line — only apply if
        # the file actually exists, and don't clobber a manual-pick made via
        # the GUI dark-picker.
        dk = params.get('dark_file')
        if dk and not self.dark_file and os.path.exists(dk):
            self.dark_file = dk
        # Geometry changed → invalidate cached coord map
        self._inv_coords = None
        self._inv_cache_key = ()
        self._dark_image = None
        self._dark_cache_key = ()
        return params

    def get_dark(self) -> Optional[np.ndarray]:
        """Cached dark frame. Reads from `dark_file` if set, else from
        `data_file` at `dark_loc`."""
        src = self.dark_file or self.data_file
        if not src:
            return None
        key = (src, self.dark_loc, tuple(self.im_trans_opts))
        # Cache HIT only when both key matches AND a real image is stored.
        # Several call sites invalidate by clearing `_dark_image = None`
        # without touching `_dark_cache_key`; without this guard, a matching
        # key would return the cleared None forever.
        if self._dark_cache_key == key and self._dark_image is not None:
            return self._dark_image
        dark = read_h5_dark(src, self.dark_loc,
                            self.header_size, self.bytes_per_pixel,
                            self.ny, self.nz)
        if dark is not None:
            dark = apply_imtransopt(dark, self.im_trans_opts)
            # Only cache successful reads — caching None freezes transient
            # I/O failures into permanent ones.
            self._dark_image = dark
            self._dark_cache_key = key
        return dark

    def get_inv_coords(self, big_det_size: int, px: float) -> tuple:
        """Cached inverse-mapping coordinates. Recomputes when geometry
        (BC, tx, BigDetSize, px) changes."""
        key = (self.bc_y, self.bc_z, self.tx, int(big_det_size), float(px))
        if self._inv_cache_key == key and self._inv_coords is not None:
            return self._inv_coords
        rows, cols = compute_inv_coords(
            self.bc_y, self.bc_z, self.tx, big_det_size, px)
        self._inv_coords = (rows, cols)
        self._inv_cache_key = key
        return self._inv_coords

    def get_remapped_frame(self, frame_idx: int, big_det_size: int,
                            px: float, subtract_dark: bool = True) -> Optional[np.ndarray]:
        """Read frame, transform/mask/dark-subtract, remap into composite."""
        if not self.enabled or not self.data_file:
            return None
        img = read_h5_frame(self.data_file, self.data_loc, frame_idx,
                            self.header_size, self.bytes_per_pixel,
                            self.ny, self.nz)
        if img is None:
            return None
        img = apply_imtransopt(img, self.im_trans_opts)
        img = mask_sentinels(img, self.bad_px, self.gap_px)
        if subtract_dark:
            dark = self.get_dark()
            tag = os.path.basename(self.data_file)
            src = self.dark_file or self.data_file
            if dark is None:
                _warn_once(
                    f'dark-subtract[{tag}]: get_dark() returned None — '
                    f'src={os.path.basename(src)}  loc={self.dark_loc!r} '
                    f'(dataset missing or unreadable; subtraction skipped)')
            elif dark.shape != img.shape:
                _warn_once(
                    f'dark-subtract[{tag}]: shape mismatch — '
                    f'dark={dark.shape} vs img={img.shape}; subtraction skipped')
            else:
                d_mean = float(np.nanmean(dark))
                i_mean = float(np.nanmean(img))
                _warn_once(
                    f'dark-subtract[{tag}]: applied — '
                    f'<dark>={d_mean:.1f}  <img>={i_mean:.1f}  '
                    f'<img-dark>={i_mean - d_mean:.1f}  '
                    f'src={os.path.basename(src)}  loc={self.dark_loc!r}')
                img = img - dark
                np.maximum(img, 0.0, out=img,
                           where=~np.isnan(img))
        rows, cols = self.get_inv_coords(big_det_size, px)
        return remap_to_composite(img, rows, cols)


# ── Multi-detector composite driver ─────────────────────────────────────────

def composite_frame(states: list[DetectorState],
                    frame_idx: int,
                    big_det_size: int,
                    px: float,
                    op: str = 'max',
                    subtract_dark: bool = True,
                    parallel: bool = True) -> np.ndarray:
    """Composite `frame_idx` from all enabled detectors.

    Returns a `BigDetSize × BigDetSize` float32 array. Uses a small
    ThreadPoolExecutor for the per-detector remap so the 4 scipy calls
    happen concurrently (map_coordinates releases the GIL).
    """
    enabled = [s for s in states if s.enabled and s.data_file]
    if not enabled:
        return np.zeros((big_det_size, big_det_size), dtype=np.float32)

    if parallel and len(enabled) > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(enabled)) as pool:
            futures = [pool.submit(s.get_remapped_frame, frame_idx,
                                   big_det_size, px, subtract_dark)
                       for s in enabled]
            results = [f.result() for f in futures]
    else:
        results = [s.get_remapped_frame(frame_idx, big_det_size, px,
                                         subtract_dark)
                   for s in enabled]
    valid = [r for r in results if r is not None]
    if not valid:
        return np.zeros((big_det_size, big_det_size), dtype=np.float32)
    return composite(valid, op=op)


def autopick_big_det_size(states: list[DetectorState]) -> int:
    """Choose a BigDetSize that comfortably fits every detector's content.

    For each enabled detector, find the maximum BC-to-corner distance in
    pixels. The composite must be at least ``2 × max_extent`` so that the
    farthest detector corner lands inside the canvas; otherwise the rotated
    panels get clipped (visible as "windmill cut off at the corners").
    Adds a small margin and rounds up to the next 256 pixels.

    BC may sit OUTSIDE the detector — typical for hydra setups where the
    direct beam doesn't hit the active area — so the max corner distance
    can far exceed ``max(NrPixels)``. The previous heuristic of
    ``2 × max(NrPixels)`` was correct only when BC ≈ image center.

    Falls back to ``2 × max(NrPixels)`` when no params are loaded yet, or
    to 4096 if even that's unavailable.
    """
    max_extent = 0.0
    for s in states:
        if not s.enabled or not s.param_file:
            continue
        # 4 corners in (row=Z, col=Y) detector pixel coords
        corners = ((0, 0),
                   (0, s.nz - 1),
                   (s.ny - 1, 0),
                   (s.ny - 1, s.nz - 1))
        for (i, j) in corners:
            d = math.hypot(j - s.bc_y, i - s.bc_z)
            if d > max_extent:
                max_extent = d
    if max_extent == 0.0:
        sizes = [max(s.ny, s.nz) for s in states
                 if s.enabled and s.param_file]
        return 2 * max(sizes) if sizes else 4096
    # 2× max corner distance + small margin, rounded up to a tidy 256.
    raw = 2.0 * max_extent + 64.0
    return int(math.ceil(raw / 256.0)) * 256

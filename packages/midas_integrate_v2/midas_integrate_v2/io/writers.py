"""Profile writers: CSV / XYE / FXYE / DAT / 2D-CSV.

Each writer emits a header block carrying provenance metadata
(`midas_integrate_v2` version, ISO timestamp, param hash, geometry
summary) so the output file is self-describing and reproducible.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .. import __version__


@dataclass
class ProfileMetadata:
    """Provenance + integration metadata embedded in profile outputs."""
    package: str = "midas_integrate_v2"
    version: str = __version__
    timestamp_iso: str = field(default_factory=lambda:
                                datetime.datetime.now(datetime.timezone.utc)
                                .isoformat())
    integrate_mode: str = "polygon"          # hard / subpixel / polygon / soft
    integrate_K: Optional[int] = None        # only for subpixel
    n_r_bins: int = 0
    n_eta_bins: int = 0
    spec_summary: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_header_lines(self, prefix: str = "# ") -> list[str]:
        """Render as a list of comment-prefixed lines for file headers."""
        d = {
            "package":        self.package,
            "version":        self.version,
            "timestamp_iso":  self.timestamp_iso,
            "integrate_mode": self.integrate_mode,
        }
        if self.integrate_K is not None:
            d["integrate_K"] = self.integrate_K
        d["n_r_bins"]   = self.n_r_bins
        d["n_eta_bins"] = self.n_eta_bins
        d["spec_summary"] = self.spec_summary
        if self.extra:
            d["extra"] = self.extra
        return [prefix + line for line in
                json.dumps(d, indent=2).splitlines()]


def build_provenance(
    spec,
    *,
    integrate_mode: str = "polygon",
    integrate_K: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    eta_coverage_per_ring: Optional[Any] = None,
) -> ProfileMetadata:
    """Build :class:`ProfileMetadata` from an :class:`IntegrationSpec`.

    Captures the geometry-relevant fields (Lsd, BC, tilts, distortion
    summary, wavelength, binning) so a later reader can sanity-check
    that the profile was produced with the expected configuration.
    """
    summary = {
        "Lsd_um":      float(spec.Lsd),
        "BC_y_px":     float(spec.BC_y),
        "BC_z_px":     float(spec.BC_z),
        "tx_deg":      float(spec.tx),
        "ty_deg":      float(spec.ty),
        "tz_deg":      float(spec.tz),
        "Parallax_um": float(spec.Parallax),
        "Wavelength_A": float(spec.Wavelength),
        "pxY_um":      spec.pxY,
        "pxZ_um":      spec.pxZ,
        "NrPixelsY":   spec.NrPixelsY,
        "NrPixelsZ":   spec.NrPixelsZ,
        "RhoD_px":     spec.RhoD,
        "RBinSize_px": spec.RBinSize,
        "EtaBinSize_deg": spec.EtaBinSize,
        "TransOpt":    list(spec.TransOpt),
    }
    # Hash of the geometry summary for reproducibility checks
    summary["spec_hash"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode()
    ).hexdigest()[:16]
    extra_dict = dict(extra or {})
    if eta_coverage_per_ring is not None:
        coverage_list = (
            eta_coverage_per_ring.detach().cpu().tolist()
            if hasattr(eta_coverage_per_ring, "detach")
            else list(eta_coverage_per_ring)
        )
        extra_dict["eta_coverage_per_ring"] = [float(c) for c in coverage_list]
    return ProfileMetadata(
        integrate_mode=integrate_mode,
        integrate_K=integrate_K,
        n_r_bins=spec.n_r_bins,
        n_eta_bins=spec.n_eta_bins,
        spec_summary=summary,
        extra=extra_dict,
    )


def _write_with_header(
    path: Path,
    header_lines: list[str],
    array: np.ndarray,
    *,
    fmt: str = "%.6e",
    delimiter: str = " ",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        np.savetxt(f, array, fmt=fmt, delimiter=delimiter)


def write_csv(
    path: str | Path,
    *,
    r_axis: np.ndarray,
    intensity: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    metadata: Optional[ProfileMetadata] = None,
) -> Path:
    """Write a 1D profile as comma-separated R, I, [σ].

    Header is JSON inside ``# `` comment block. Easily reloaded by
    ``np.loadtxt`` or ``pandas.read_csv``.
    """
    path = Path(path)
    cols = [r_axis, intensity]
    header_cols = "R_px,intensity"
    if sigma is not None:
        cols.append(sigma)
        header_cols += ",sigma"
    arr = np.column_stack(cols)
    header_lines = [] if metadata is None else metadata.to_header_lines()
    header_lines.append("# " + header_cols)
    _write_with_header(path, header_lines, arr, fmt="%.6e", delimiter=",")
    return path


def write_xye(
    path: str | Path,
    *,
    r_axis: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    metadata: Optional[ProfileMetadata] = None,
) -> Path:
    """Write the canonical Rietveld XYE format (3 columns: x, y, error).

    ``r_axis`` may be in any monotonic-increasing units (R px, 2θ deg,
    Q Å⁻¹) — the metadata header documents which.
    """
    if sigma is None or sigma.shape != intensity.shape:
        raise ValueError("XYE requires sigma matching intensity shape")
    arr = np.column_stack([r_axis, intensity, sigma])
    header_lines = [] if metadata is None else metadata.to_header_lines()
    header_lines.append("# x  y  error  (XYE format for Rietveld)")
    _write_with_header(path, header_lines, arr, fmt="%.6e", delimiter=" ")
    return path


def write_fxye(
    path: str | Path,
    *,
    r_axis: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    metadata: Optional[ProfileMetadata] = None,
    title: str = "midas-integrate-v2 export",
    x_unit: str = "centidegrees_2theta",
) -> Path:
    """Write GSAS FXYE format (fixed-format header + 3-column data).

    GSAS / GSAS-II / TOPAS expect ``2θ`` in **centidegrees** with the
    ESD column (column 3) carrying ``σ`` (one standard deviation) — *not*
    variance, *not* σ². We emit exactly that.

    Parameters
    ----------
    r_axis :
        Independent axis. Per the FXYE convention this should be 2θ in
        **centidegrees**. If you pass 2θ in degrees, set
        ``x_unit="degrees_2theta"`` and we'll multiply by 100 for you.
    intensity, sigma :
        Per-bin intensity and 1-σ uncertainty. ``sigma`` must be the
        standard deviation, never the variance.
    title :
        80-char header line written verbatim before the BANK statement.
    x_unit :
        ``"centidegrees_2theta"`` (default) or ``"degrees_2theta"``. Other
        values raise ``ValueError`` to keep the contract with downstream
        Rietveld tools explicit.
    """
    if x_unit not in ("centidegrees_2theta", "degrees_2theta"):
        raise ValueError(
            f"x_unit must be 'centidegrees_2theta' or 'degrees_2theta'; "
            f"got {x_unit!r}"
        )
    if sigma.shape != intensity.shape:
        raise ValueError("sigma must have the same shape as intensity")
    if x_unit == "degrees_2theta":
        x_centideg = np.asarray(r_axis, dtype=np.float64) * 100.0
    else:
        x_centideg = np.asarray(r_axis, dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        # GSAS / GSAS-II layout (order matters!):
        #   line 0 : 80-char title comment (always consumed as the title)
        #   then   : optional '#' metadata comment lines
        #   then   : the BANK statement
        #   then   : the x/f/s data rows
        # The metadata MUST precede the BANK line. GSAS-II's FXYE reader seeks
        # to the byte just after BANK and reads data with a loop that STOPS at
        # the first line beginning with '#':
        #     while S and S[:4] != 'BANK' and S[0] != '#':
        # so any '#' comment placed between BANK and the data truncates the
        # read to zero points (the file still "validates" — ContentsValidator
        # only counts BANK lines — but reads empty). Comments before BANK are
        # collected into GSAS-II's per-bank comment list, so metadata survives.
        f.write(f"{title:<80s}\n")
        if metadata is not None:
            for line in metadata.to_header_lines(prefix="# "):
                f.write(line + "\n")
        nlines = len(x_centideg)
        step = (x_centideg[1] - x_centideg[0]) if nlines >= 2 else 0.0
        # BANK 1, n points, n records, CONST step, start, step, 0 0 FXYE
        f.write(
            f"BANK 1 {nlines} {nlines} CONST "
            f"{x_centideg[0]:.5f} {step:.5f} "
            f"0 0 FXYE\n"
        )
        for x, y, e in zip(x_centideg, intensity, sigma):
            f.write(f"{x:14.5f} {y:14.5e} {e:14.5e}\n")
    return path


def write_esg(
    path: str | Path,
    *,
    two_theta_deg: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    wavelength_A: float,
    metadata: Optional[ProfileMetadata] = None,
    block_id: str = "midas|#0|0",
    bank_id: int = 1,
) -> Path:
    """Write MAUD/MILK ESG (CIF-like) dataset format.

    ESG is the canonical input format for MAUD's ``maudbatch`` / MILK
    orchestration. It's a CIF-like text file with an optional header
    block, then a ``loop_`` over ``2θ, I, σ`` triples. ``σ`` is the
    *standard deviation*, written under ``_pd_proc_intensity_total_su``
    (CIF "su" = standard uncertainty = 1σ).

    Parameters
    ----------
    two_theta_deg :
        2θ axis in **degrees** (ESG canonical unit; MAUD's reader
        converts internally). Must be monotonically increasing and
        evenly spaced for MAUD's CONST-step ingest.
    intensity, sigma :
        Per-bin intensity and 1-σ uncertainty. Same length as
        ``two_theta_deg``.
    wavelength_A :
        Source wavelength in Ångströms; written to
        ``_diffrn_radiation_wavelength`` for MAUD.
    block_id :
        ``_pd_block_id`` value. The ``#0`` index lets MILK round-trip
        bank assignments when emitting multi-bank datasets.
    bank_id :
        Bank index for multi-bank Hydra-style datasets (item 46).
    """
    if not (intensity.shape == sigma.shape == two_theta_deg.shape):
        raise ValueError(
            "two_theta_deg, intensity, and sigma must share shape; got "
            f"{two_theta_deg.shape}, {intensity.shape}, {sigma.shape}"
        )
    if two_theta_deg.ndim != 1:
        raise ValueError("two_theta_deg must be 1-D")
    n = int(two_theta_deg.shape[0])
    if n < 2:
        raise ValueError("ESG requires at least 2 points")
    tth = np.asarray(two_theta_deg, dtype=np.float64)
    step = float(tth[1] - tth[0])
    # Sanity: warn (not raise) on non-uniform spacing
    spacings = np.diff(tth)
    uniform = bool(np.allclose(spacings, step, rtol=1e-4, atol=1e-6))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# ESG dataset (MAUD/MILK canonical) "
                "written by midas-integrate-v2\n")
        if metadata is not None:
            for line in metadata.to_header_lines(prefix="# "):
                f.write(line + "\n")
        if not uniform:
            f.write("# WARNING: 2theta axis is not uniformly spaced; "
                    "MAUD's CONST ingest assumes uniform step.\n")
        f.write(f"_pd_block_id {block_id}\n")
        f.write(f"_diffrn_radiation_wavelength {wavelength_A:.6f}\n")
        f.write(f"_pd_meas_bank_id {bank_id}\n")
        f.write(f"_pd_meas_2theta_range_min {tth[0]:.6f}\n")
        f.write(f"_pd_meas_2theta_range_max {tth[-1]:.6f}\n")
        f.write(f"_pd_meas_2theta_range_inc {step:.6f}\n")
        f.write(f"_pd_meas_number_of_points {n}\n")
        f.write("\nloop_\n")
        f.write("_pd_meas_2theta_scan\n")
        f.write("_pd_proc_intensity_total\n")
        f.write("_pd_proc_intensity_total_su\n")
        for x, y, e in zip(tth, intensity, sigma):
            f.write(f"{x:14.6f} {y:14.6e} {e:14.6e}\n")
    return path


def write_dat(
    path: str | Path,
    *,
    q_axis_invA: np.ndarray,
    intensity: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    metadata: Optional[ProfileMetadata] = None,
) -> Path:
    """Write a `.dat` file for PDF analysis (`PDFGetN`, `PDFGetX2`, …).

    Q axis must be in inverse Ångströms. Two or three columns
    (`Q  I  [σ]`).
    """
    cols = [q_axis_invA, intensity]
    header_cols = "Q_invA  intensity"
    if sigma is not None:
        cols.append(sigma)
        header_cols += "  sigma"
    arr = np.column_stack(cols)
    header_lines = [] if metadata is None else metadata.to_header_lines()
    header_lines.append("# " + header_cols)
    _write_with_header(path, header_lines, arr, fmt="%.6e", delimiter="  ")
    return path


def write_h5(
    path: str | Path,
    *,
    profiles: np.ndarray,                    # (N, n_r)
    r_axis: np.ndarray,                      # (n_r,)
    frame_ids: Optional[list] = None,
    sigmas: Optional[np.ndarray] = None,     # (N, n_r)
    metadata: Optional[ProfileMetadata] = None,
    extra_datasets: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """Write a stack of integrated profiles to a single HDF5 file.

    Layout (NeXus-friendly subset):
      /profiles      (N, n_r)  — intensity per frame, per R bin
      /sigmas        (N, n_r)  — uncertainty per bin (if supplied)
      /r_axis_px     (n_r,)    — radial axis
      /frame_ids     (N,)      — frame identifiers (utf-8 strings)
      /metadata      attribute — JSON of :class:`ProfileMetadata`
      /<extra_key>   any extra dataset the caller supplies

    Requires h5py.
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError("write_h5 requires h5py; pip install h5py") from e
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if profiles.ndim != 2:
        raise ValueError(
            f"profiles must be 2-D (N, n_r); got shape {profiles.shape}"
        )
    if r_axis.shape[0] != profiles.shape[1]:
        raise ValueError(
            f"r_axis length {r_axis.shape[0]} != n_r {profiles.shape[1]}"
        )
    if sigmas is not None and sigmas.shape != profiles.shape:
        raise ValueError(
            f"sigmas shape {sigmas.shape} != profiles shape {profiles.shape}"
        )
    if frame_ids is None:
        frame_ids = [f"frame_{i:05d}" for i in range(profiles.shape[0])]
    if len(frame_ids) != profiles.shape[0]:
        raise ValueError("frame_ids length must match profiles axis 0")

    with h5py.File(path, "w") as f:
        # NeXus-strict layout (Item 33):
        #   /                NXroot
        #   /entry/          NXentry
        #   /entry/data/     NXdata     (signal=profiles, axes=...)
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        data_grp = entry.create_group("data")
        data_grp.attrs["NX_class"] = "NXdata"
        data_grp.attrs["signal"] = "profiles"
        data_grp.attrs["axes"] = ["frame_index", "r_px"]
        data_grp.attrs["r_px_indices"] = 1
        prof_ds = data_grp.create_dataset(
            "profiles", data=profiles, compression="gzip",
        )
        prof_ds.attrs["units"] = "counts"
        rax_ds = data_grp.create_dataset("r_px", data=r_axis)
        rax_ds.attrs["units"] = "pixel"
        frames_ds = data_grp.create_dataset(
            "frame_ids", data=np.array(frame_ids, dtype="S"),
        )
        frames_ds.attrs["long_name"] = "frame identifier"
        if sigmas is not None:
            sig_ds = data_grp.create_dataset(
                "sigmas", data=sigmas, compression="gzip",
            )
            sig_ds.attrs["units"] = "counts"
            sig_ds.attrs["long_name"] = "1-sigma uncertainty per bin"
        if metadata is not None:
            entry.attrs["metadata_json"] = json.dumps(asdict(metadata))
        if extra_datasets:
            extra_grp = entry.create_group("extra")
            extra_grp.attrs["NX_class"] = "NXcollection"
            for k, v in extra_datasets.items():
                extra_grp.create_dataset(k, data=v, compression="gzip")
                # Backward-compat: top-level access mirror via soft link
                f[k] = h5py.SoftLink(f"/entry/extra/{k}")
        # Top-level NeXus signal pointer
        f.attrs["NX_class"] = "NXroot"
        f.attrs["default"] = "entry"
        # Backward-compat: keep flat top-level too via soft links so
        # callers that read the old layout (profiles/, r_axis_px/, …)
        # continue to work.
        f["profiles"] = h5py.SoftLink("/entry/data/profiles")
        f["r_axis_px"] = h5py.SoftLink("/entry/data/r_px")
        f["frame_ids"] = h5py.SoftLink("/entry/data/frame_ids")
        if sigmas is not None:
            f["sigmas"] = h5py.SoftLink("/entry/data/sigmas")
        if metadata is not None:
            f.attrs["metadata_json"] = json.dumps(asdict(metadata))
    return path


def write_2d_csv(
    path: str | Path,
    *,
    int2d: np.ndarray,                       # (n_eta, n_r)
    r_axis_px: np.ndarray,                   # (n_r,)
    eta_axis_deg: np.ndarray,                # (n_eta,)
    metadata: Optional[ProfileMetadata] = None,
) -> Path:
    """Write the 2D ``(η, R)`` integrated array.

    Layout: header block, then a CSV with the R axis in row 0, η axis
    as the first column, and intensity in the rest. NaN-filled
    upper-left corner.
    """
    if int2d.shape != (eta_axis_deg.shape[0], r_axis_px.shape[0]):
        raise ValueError(
            f"int2d shape {int2d.shape} does not match "
            f"(n_eta={eta_axis_deg.shape[0]}, n_r={r_axis_px.shape[0]})"
        )
    n_eta, n_r = int2d.shape
    out = np.full((n_eta + 1, n_r + 1), np.nan, dtype=np.float64)
    out[0, 1:]  = r_axis_px
    out[1:, 0]  = eta_axis_deg
    out[1:, 1:] = int2d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if metadata is not None:
            for line in metadata.to_header_lines():
                f.write(line + "\n")
        f.write("# 2D integrated array; row 0 = R_px axis, "
                "col 0 = eta_deg axis\n")
        np.savetxt(f, out, fmt="%.6e", delimiter=",")
    return path

"""One command from a scan record to a registered reconstruction.

Before this, reconstructing a 1-ID tomography scan meant: hand-write a
``prepare_data_<sample>.py`` with four eye-counted frame indices, hand-write a
driver with the angles and a shift range, reconstruct 501 candidate shifts,
render a 501-panel contact sheet, pick one by eye, and remember the number
because it is not stored anywhere. One pair of scripts per sample.

:func:`reconstruct_scan` does the same job from the scan's own record:

1. **read** ``<prefix>_TomoFastScan.dat`` — pixel size, propagation distance,
   energy, handedness, angles and the exact frame layout
   (:mod:`midas_tomo.scanrecord`);
2. **ingest** the TIFFs into the engine's binary layout, frame boundaries
   derived and cross-checked (:mod:`midas_tomo.ingest`);
3. optionally **phase-retrieve** (:mod:`midas_tomo.phase_retrieval`) — off by
   default, because it is a strong smoother whose parameter sets how big the
   specimen comes out;
4. **find the rotation-axis shift** automatically, coarse then fine, scored by
   two criteria that must agree (:mod:`midas_tomo.center`);
5. **reconstruct** at that shift;
6. optionally **measure the detector roll**
   (:mod:`midas_tomo.detector_tilt`);
7. **write** NXtomoproc with the full provenance attached, and print the
   :class:`~midas_transforms.geometry.SampleShape` arguments to use.

What it will not do
-------------------
It does not invent a registration. The pixel size comes from the scan record
and nothing else; the rotation-axis shift is measured and reported with a
``trustworthy`` flag; the in-plane handedness is *recorded* in the metastr but
mapping it onto a signed axis permutation is unresolved, so
:func:`reconstruct_scan` passes the string through to provenance and does not
choose for you.

And it does not quietly proceed on an uncertified shift. ``strict=True`` (the
default) stops when automatic centring cannot certify its answer, because the
alternative is a reconstruction that looks fine and is mis-registered.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .center import find_center_consensus, slices_with_signal
from .ingest import IngestResult, stage_scan_to_binary
from .scanrecord import TomoScan, read_scan_record

__all__ = ["ReconstructionResult", "reconstruct_scan"]

log = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    """Everything the run produced, and everything needed to cite it."""

    scan: TomoScan
    ingest: IngestResult
    shift: float
    shift_trustworthy: bool
    shift_reason: str
    recon_path: Optional[Path]
    recon_shape: Tuple[int, ...]
    tilt_deg: Optional[float] = None
    tilt_trustworthy: Optional[bool] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def sample_shape_hint(self) -> str:
        """The `SampleShape` call this reconstruction supports, with the gaps."""
        s = self.scan
        nx = self.recon_shape[-1] if self.recon_shape else "<X>"
        return (
            "from midas_transforms.geometry import from_nxtomoproc\n"
            f"shape = from_nxtomoproc(\n"
            f"    {str(self.recon_path)!r},\n"
            f"    pixel_size_um={s.pixel_size_um},        # from tomo_metastr\n"
            f"    rot_axis_ix=<X/2 + shift>, rot_axis_iy=<same>,\n"
            f"    in_plane=<UNRESOLVED: metastr says {s.handedness!r} handed,\n"
            f"             which names a convention, not an axis assignment>,\n"
            f"    threshold=<sweep it: threshold_sensitivity()>,\n"
            f"    slice0_z_um=<stage vertical for this scan>,\n"
            f"    det_xdim={nx},\n"
            ")"
        )

    def summary(self) -> str:
        s = self.scan
        lines = [
            f"scan        {s.image_prefix}",
            f"  detector  {s.detector} {s.magnification}, "
            f"{s.pixel_size_um} um/px, D~{s.propagation_mm} mm",
            f"  energy    {s.energy_kev} keV",
            f"  angles    {s.omega_start_deg} .. {s.omega_end_deg} step "
            f"{s.omega_step_deg} ({s.n_projections} projections), "
            f"{'aero sign applied' if s.is_aero else 'as recorded'}",
            f"  frames    " + ", ".join(f"{b.role}@{b.start}x{b.count}"
                                        for b in s.blocks),
            f"shift       {self.shift:+.3f} px  "
            f"(trustworthy={self.shift_trustworthy})",
        ]
        if not self.shift_trustworthy:
            lines.append(f"            {self.shift_reason}")
        if self.tilt_deg is not None:
            lines.append(f"roll        {self.tilt_deg:+.4f} deg  "
                         f"(trustworthy={self.tilt_trustworthy})")
        if self.recon_path:
            lines.append(f"wrote       {self.recon_path}  {self.recon_shape}")
        return "\n".join(lines)


def _even(lo: float, hi: float, step: float) -> Tuple[float, float, float]:
    """The engine reconstructs shifts in pairs and rejects an odd count."""
    n = int(round(abs(hi - lo) / step)) + 1
    if n % 2:
        hi = hi + step
    return (lo, hi, step)


def reconstruct_scan(
    scan_record: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    root: Union[str, Path],
    crop: Optional[Tuple[int, int, int, int]] = None,
    ext: str = ".tif",
    coarse: Tuple[float, float, float] = (-25.0, 25.0, 1.0),
    fine_half_width: float = 2.0,
    fine_step: float = 0.1,
    delta_beta: float = 0.0,
    centre_slab: int = 16,
    measure_tilt: bool = False,
    n_cpus: int = 8,
    filter_nr: int = 2,
    strict: bool = True,
    stripe: Optional[Dict[str, Any]] = None,
    write_hdf5: bool = True,
    progress=None,
) -> ReconstructionResult:
    """Scan record -> reconstruction, with the shift found automatically.

    ``root`` is the local directory holding the scan's image folder; the
    ``Path:`` in the record is the acquisition machine's view.

    ``delta_beta`` enables Paganin phase retrieval. It defaults to 0, which is
    bit-exactly no filtering — the parameter is a strong low-pass and choosing
    it sets how large the specimen reconstructs, so it is never on by default.

    ``centre_slab`` limits the shift sweeps to that many detector rows around
    the signal-bearing centre. Necessary, not an optimisation: the sweep cube
    is ``n_shifts x n_slices x X x X``, so a 52-shift coarse sweep over a full
    2048-row, 2320-wide scan is **111 GB**. Sixteen rows makes it 1.7 GB, and
    centring does not improve with more rows once several carry specimen. The
    final reconstruction uses every row.

    ``strict`` stops with a :class:`RuntimeError` when automatic centring
    cannot certify the shift. Pass ``strict=False`` to reconstruct anyway; the
    result then carries ``shift_trustworthy=False`` and everything downstream
    should treat the geometry as unverified.
    """
    say = progress or (lambda m: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 ---------------------------------------------------------- the record
    scan = read_scan_record(scan_record)
    say(f"scan record: {scan.image_prefix}, {scan.pixel_size_um} um/px, "
        f"{scan.n_projections} projections")

    # 2 ---------------------------------------------------------- ingest
    bin_path = out_dir / "input.bin"
    ing = stage_scan_to_binary(scan, bin_path, root=root, crop=crop, ext=ext,
                               progress=say)
    say(ing.summary())

    ny, nx = ing.ny, ing.nx
    cal = ny * nx
    buf = np.fromfile(bin_path, dtype=np.float32, count=3 * cal)
    dark = buf[:cal].reshape(ny, nx).astype(np.float64)
    whites = np.stack([buf[cal:2 * cal].reshape(ny, nx),
                       buf[2 * cal:3 * cal].reshape(ny, nx)]).astype(np.float64)
    data = np.fromfile(bin_path, dtype=np.uint16, offset=3 * cal * 4,
                       count=scan.n_projections * cal
                       ).reshape(scan.n_projections, ny, nx)
    angles = scan.thetas()

    # 3 ---------------------------------------------------- phase retrieval
    if delta_beta > 0:
        from .phase_retrieval import paganin_filter

        say(f"Paganin filter, delta/beta={delta_beta}")
        denom = whites.mean(axis=0) - dark
        denom = np.where(denom <= 0, np.nan, denom)
        trans = (data.astype(np.float64) - dark) / denom
        trans = np.nan_to_num(trans, nan=1.0, posinf=1.0, neginf=1.0)
        trans = paganin_filter(
            trans, pixel_size_um=scan.pixel_size_um,
            distance_mm=scan.propagation_mm, energy_kev=scan.energy_kev,
            delta_beta=delta_beta,
        )
        # Re-express as counts so the engine's own flat-field step is a no-op
        # on top of what we just did, rather than being applied twice.
        data = np.clip(trans * denom + dark, 0, 65535).astype(np.uint16)
        data = np.nan_to_num(data)

    # 4 ------------------------------------------------------- find the shift
    from .api import run_tomo

    stripe_kw = dict(stripe or {})
    try:
        signal_rows = slices_with_signal(data, dark, whites, k=64)
    except ValueError as exc:
        if strict:
            raise RuntimeError(
                f"cannot choose slices to centre on: {exc}"
            ) from exc
        signal_rows = None
        log.warning("no rows with signal (%s); centring on evenly spaced "
                    "slices, which is unreliable", exc)

    # Centre on a SLAB, not the whole stack -- see centre_slab in the docstring.
    if signal_rows:
        mid = int(np.median(signal_rows))
    else:
        mid = ny // 2
    half = max(2, int(centre_slab) // 2)
    r0, r1 = max(0, mid - half), min(ny, mid + half)
    if r1 - r0 < 4:
        r0, r1 = 0, min(ny, 8)
    say(f"centring on rows {r0}:{r1} of {ny} "
        f"(full sweep over {ny} rows would need "
        f"{52 * ny * 2 * (2 ** int(np.ceil(np.log2(nx)))) ** 2 / 1e9:.0f} GB)")
    d_s, k_s, w_s = data[:, r0:r1, :], dark[r0:r1], whites[:, r0:r1]
    probe = None if signal_rows is None else \
        sorted({min(max(s - r0, 0), r1 - r0 - 1) for s in signal_rows
                if r0 <= s < r1}) or None

    say(f"coarse shift sweep {coarse}")
    cube = run_tomo(d_s, k_s, w_s, out_dir / "sweep_coarse", angles,
                    shifts=_even(*coarse), filter_nr=filter_nr,
                    do_log=True, n_cpus=n_cpus, **stripe_kw)
    c = find_center_consensus(cube, _even(*coarse), slices=probe)
    say(f"  coarse -> {c['best_shift']:+.3f} (trustworthy={c['trustworthy']})")

    fine = _even(c["best_shift"] - fine_half_width,
                 c["best_shift"] + fine_half_width, fine_step)
    say(f"fine shift sweep {fine}")
    cube = run_tomo(d_s, k_s, w_s, out_dir / "sweep_fine", angles,
                    shifts=fine, filter_nr=filter_nr, do_log=True,
                    n_cpus=n_cpus, **stripe_kw)
    f = find_center_consensus(cube, fine, slices=probe)
    say(f"  fine   -> {f['best_shift']:+.3f} (trustworthy={f['trustworthy']})")

    if strict and not f["trustworthy"]:
        raise RuntimeError(
            "automatic centring could not certify a rotation-axis shift: "
            f"{f['reason']}. Reconstructing anyway would give a plausible, "
            "mis-registered volume. Inspect the sweep in "
            f"{out_dir / 'sweep_fine'}, or pass strict=False to proceed with "
            "the geometry marked unverified."
        )

    # 5 ------------------------------------------------------- reconstruct
    shift = float(f["best_shift"])
    say(f"reconstructing at shift {shift:+.3f}")
    recon = run_tomo(data, dark, whites, out_dir / "recon", angles,
                     shifts=(shift, shift, 1.0), filter_nr=filter_nr,
                     do_log=True, n_cpus=n_cpus, **stripe_kw)

    # 6 --------------------------------------------------------- the roll
    tilt = tilt_ok = None
    if measure_tilt:
        from .detector_tilt import tilt_from_beam_box

        try:
            t = tilt_from_beam_box(whites[0], dark)
            tilt, tilt_ok = t.angle_deg, t.trustworthy
            say(f"detector roll {tilt:+.4f} deg (trustworthy={tilt_ok})")
        except ValueError as exc:
            say(f"detector roll not measurable from the flat: {exc}")

    # 7 ------------------------------------------------------------- write
    prov = dict(ing.provenance)
    prov.update({
        "shift_px": shift,
        "shift_trustworthy": bool(f["trustworthy"]),
        "shift_reason": f["reason"],
        "shift_coarse_px": c["best_shift"],
        "centring_slices": list(probe) if probe else None,
        "centring_rows": [r0, r1],
        "delta_beta": float(delta_beta),
        "phase_retrieval": "Paganin" if delta_beta > 0 else "none",
        "filter_nr": filter_nr,
        "stripe_removal": stripe_kw or None,
        "detector_roll_deg": tilt,
        "detector_roll_trustworthy": tilt_ok,
    })

    recon_path = None
    if write_hdf5:
        from .hdf5 import write_recon_hdf5

        recon_path = write_recon_hdf5(
            out_dir / f"{Path(scan.image_prefix).name}_recon.h5", recon,
            angles=angles, shifts=np.array([shift]),
            metadata={k: (v if isinstance(v, (str, int, float)) else str(v))
                      for k, v in prov.items()},
        )
    (out_dir / "provenance.json").write_text(
        json.dumps(prov, indent=2, default=str)
    )

    return ReconstructionResult(
        scan=scan, ingest=ing, shift=shift,
        shift_trustworthy=bool(f["trustworthy"]), shift_reason=f["reason"],
        recon_path=recon_path, recon_shape=tuple(recon.shape),
        tilt_deg=tilt, tilt_trustworthy=tilt_ok, provenance=prov,
    )

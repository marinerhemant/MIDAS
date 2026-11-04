"""Calibrate a TILTED detector from a powder-calibrant image, and check the answer.

One entry point, :func:`calibrate_tilted`, for the case the default ``calibrate()`` does not
handle (it returned ``tz`` = -0.014 deg on a 14 deg detector, and 2500 microstrain at 5 deg
with distortion railed; see ``manuals/calibrate-integrate/RUNBOOK.md``). It chains pieces that
already exist:

1. clean the frame (negative pixels, dead columns), ``make_seed`` for beam centre and distance
   (the seed has NO tilt);
2. :func:`~midas_calibrate_v2.pipelines.frozen_point.iterate_frozen_point_until_stable`
   (re-centred bounds, geometry only), blind to tilt, with ``min_ring_gap_deg = 0.08`` so that
   closely spaced rings are not all dropped at high energy;
3. one :func:`~midas_calibrate_v2.pipelines.frozen_point.autocalibrate_frozen_point` pass from
   the converged geometry with the radial distortion block refined (kept only if it does not
   make the gate worse);
4. :func:`~midas_calibrate_v2.ring_position_gate.ring_position_gate`, the INDEPENDENT check in
   pixels. ``converged=True`` from step 2 is not a quality check (on a 5 deg frame the same
   route "converged" onto a geometry 6 deg off in tilt, 10,000 microstrain); only the gate
   decides the verdict.

Evidence for this recipe is two real LaB6 frames (5 deg at 0.728 A: 150 microstrain, gate
0.05-0.09 px; 45 deg at 0.108 A: 558 microstrain, gate 0.08-0.17 px) and synthetic images with
exact truth. The basin is narrow: a start with the distance 3 % too long, or a tilt-0 start with
the beam centre far off, did not converge on either frame. See ``manuals/calibrate-integrate/
phase-4c-tilted.md`` for the status of every statement. This is PROVISIONAL: two frames, one
calibrant, no multi-panel detectors.

Command line::

    python -m midas_calibrate_v2.pipelines.tilted frame.tif --wavelength 0.10798 \
        --px-um 200 --calibrant LaB6 [--nominal-tilt 45 | --nominal-tilt 0.0,-44.8] \
        [--out geometry.json]
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..ring_position_gate import RingPositionGate, ring_position_gate

__all__ = ["TiltedCalibration", "calibrate_tilted", "clean_frame"]

_GEOM_KEYS = ("Lsd", "BC_y", "BC_z", "tx", "ty", "tz")
_DIST_KEYS = ("iso_R2", "iso_R4", "iso_R6") + tuple(f"a{k}" for k in range(1, 7)) + \
    tuple(f"phi{k}" for k in range(1, 7))
_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass
class TiltedCalibration:
    """Outcome of :func:`calibrate_tilted`. ``verdict`` is the gate's, not the optimiser's."""

    verdict: str                          # "PASS" | "WARN" | "FAIL"
    geometry: Dict[str, float]            # Lsd (um), BC_y/BC_z (px), tx/ty/tz (deg), distortion terms
    tilt_deg: float                       # |tilt| of the detector plane
    perp_distance_mm: float
    gate: RingPositionGate
    route: str                            # which candidate was kept
    converged: bool                       # iterate loop only; NOT a quality check
    n_iter: int
    seed: Dict[str, float]
    seconds: float
    notes: List[str] = field(default_factory=list)
    result: object = None                 # route result (has .unpacked) for to_integrate_params
    v1_params: object = None              # the CalibrationParams template the route started from
    bad_mask: object = None               # bool (NZ, NY), True = bad pixel, as cleaned by clean_frame

    def write_paramstest(self, path) -> str:
        """Write a v1-format ``paramstest.txt`` for ``midas-integrate-v2`` (single-panel).

        Refuses to write a FAIL geometry: integrating with it applies a wrong calibration to
        every frame afterwards.
        """
        from ..compat.to_v1 import write_v1_paramstest

        if self.verdict == "FAIL":
            raise RuntimeError("the ring-position gate FAILED for this geometry; refusing to write it")
        write_v1_paramstest(self.result.unpacked, self.v1_params, path)
        return str(path)

    def write_mask(self, path) -> str:
        """Write the bad-pixel mask used here as a uint8 TIFF (1 = masked) for ``midas-integrate-v2 --mask``.

        ``read_image(..., return_mask=True)`` only flags integer sentinels; negative pixels of a float frame
        are masked here by :func:`clean_frame`, and the integration must mask the same pixels.
        """
        import tifffile

        tifffile.imwrite(str(path), np.asarray(self.bad_mask, dtype=np.uint8))
        return str(path)

    def to_dict(self) -> dict:
        g = self.gate
        return dict(verdict=self.verdict, geometry=self.geometry, tilt_deg=self.tilt_deg,
                    perp_distance_mm=self.perp_distance_mm, route=self.route,
                    converged=self.converged, n_iter=self.n_iter, seed=self.seed,
                    seconds=self.seconds, notes=self.notes,
                    gate=dict(verdict=g.verdict, median_px=g.median_px, p90_px=g.p90_px,
                              p95_px=g.p95_px, max_px=g.max_px, n_fits=g.n_fits,
                              n_cells=g.n_cells_expected, n_rings=g.n_rings_used,
                              scale=g.scale, reasons=g.reasons))

    def __str__(self) -> str:
        return (f"tilted calibration {self.verdict}: tilt {self.tilt_deg:.2f} deg, "
                f"distance {self.perp_distance_mm:.2f} mm (perpendicular), route {self.route}, "
                f"{self.seconds:.0f} s\n  {self.gate}"
                + ("".join(f"\n  note: {n}" for n in self.notes)))


def clean_frame(raw: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(image, mask)`` with bad pixels zeroed, as the recipe expects.

    Bad = given mask, plus any negative pixel (detector sentinel / over-subtraction), plus
    columns whose median is below 20 % of the median column (dead strips), dilated by 2 px.
    """
    from scipy.ndimage import binary_dilation

    img = np.asarray(raw, dtype=np.float64)
    bad = np.zeros(img.shape, bool) if mask is None else np.asarray(mask, bool).copy()
    bad |= img < 0
    colmed = np.median(img, axis=0)
    good = np.median(colmed)
    if good > 0:
        bad[:, colmed < 0.2 * good] = True
    bad = binary_dilation(bad, iterations=2)
    out = img.copy()
    out[bad] = 0.0
    return out, bad


def _v1_params(img, wavelength_A, px_um, cal, seed_lsd, bc_y, bc_z, ty, tz, first_ring_px):
    from midas_calibrate.params import CalibrationParams
    from ..seed.calibrant import phases_from_calibrants

    nz, ny = img.shape
    corner = math.hypot(max(bc_y, ny - 1 - bc_y), max(bc_z, nz - 1 - bc_z))
    v1 = CalibrationParams(
        NrPixelsY=ny, NrPixelsZ=nz, pxY=px_um, pxZ=px_um, Lsd=seed_lsd, BC_y=bc_y, BC_z=bc_z,
        tx=0.0, ty=ty, tz=tz, Wavelength=wavelength_A, SpaceGroup=cal["sg"],
        LatticeConstant=(cal["a"], cal["b"], cal["c"], cal["alpha"], cal["beta"], cal["gamma"]),
        MaxRingRad=float(corner * 0.97 - 10.0), MinRingRad=float(max(20.0, 0.6 * first_ring_px)),
        RhoD=corner, nIterations=4,
        Refine={"Lsd": True, "BC": True, "ty": True, "tz": True,
                "Wavelength": False, "Parallax": False})
    v1.Phases = phases_from_calibrants(cal["name"] if cal.get("name") in ("CeO2", "LaB6", "Si") else cal)
    return v1


def _unpack(res) -> Dict[str, float]:
    out = {}
    for k, v in res.unpacked.items():
        if k in _GEOM_KEYS or k in _DIST_KEYS:
            out[k] = float(v.detach().reshape(-1)[0])
    out.setdefault("tx", 0.0)
    return out


def _two_theta_map(geom: Dict[str, float], shape, wavelength_A: float, px_um: float) -> np.ndarray:
    from midas_integrate_v2 import IntegrationSpec
    from midas_integrate_v2.forward import pixel_to_REta_from_spec

    f64 = torch.float64
    nz, ny = shape
    s = IntegrationSpec()
    s.NrPixelsY, s.NrPixelsZ, s.pxY, s.pxZ = ny, nz, px_um, px_um
    s.Lsd = torch.tensor(geom["Lsd"], dtype=f64)
    s.BC_y = torch.tensor(geom["BC_y"], dtype=f64)
    s.BC_z = torch.tensor(geom["BC_z"], dtype=f64)
    for k in ("tx", "ty", "tz"):
        setattr(s, k, torch.tensor(geom.get(k, 0.0), dtype=f64))
    s.Wavelength = torch.tensor(wavelength_A, dtype=f64)
    s.Parallax = torch.tensor(0.0, dtype=f64)
    s.RhoD = math.hypot(max(geom["BC_y"], ny - 1 - geom["BC_y"]),
                        max(geom["BC_z"], nz - 1 - geom["BC_z"])) * px_um
    for k in _DIST_KEYS:
        if k in geom:
            setattr(s, k, torch.tensor(geom[k], dtype=f64))
    zs, ys = torch.meshgrid(torch.arange(nz, dtype=f64), torch.arange(ny, dtype=f64), indexing="ij")
    with torch.no_grad():
        r = pixel_to_REta_from_spec(ys, zs, s).R_px
    return torch.rad2deg(torch.atan(r * px_um / s.Lsd)).numpy()


def _gate(img, mask, geom, rings, wavelength_A, px_um) -> RingPositionGate:
    tth = _two_theta_map(geom, img.shape, wavelength_A, px_um)
    return ring_position_gate(img, tth, rings, lsd_um=geom["Lsd"], px_um=px_um, mask=mask)


def calibrate_tilted(
    image: np.ndarray,
    *,
    wavelength_A: float,
    px_um: float,
    calibrant: Union[str, dict] = "CeO2",
    mask: Optional[np.ndarray] = None,
    nominal_tilt_deg: Union[None, float, Sequence[float]] = None,
    lsd_um: Optional[float] = None,
    polish: bool = True,
    fallback_pv: bool = True,
    point_pick_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> TiltedCalibration:
    """Calibrate a tilted detector from one calibrant frame and grade the result.

    ``image`` is the raw frame in MIDAS orientation (``image[row=Z, col=Y]``); it is cleaned by
    :func:`clean_frame` (pass ``mask`` for extra bad pixels, True = bad). ``wavelength_A`` must
    come from the beamline: it is degenerate with the distance and no fit here can recover it.

    ``nominal_tilt_deg``: ``None`` = blind (tilt 0 start); a pair ``(ty, tz)`` in degrees = the
    nominal tilt, used as the start; a single number = the tilt MAGNITUDE with unknown
    direction, which runs 8 starts (45 deg steps) and keeps the one with the lowest fit strain
    (8x the time). A tilt start within about 20 deg / 10 % of the truth converged on the
    45 deg frame; the iterated route needs the start within a few percent in distance.

    ``lsd_um``: a distance hint along the beam in micrometres (e.g. the detector stage readback,
    corrected for any known offset), used INSTEAD of ``make_seed``'s distance. ``make_seed`` assigns
    the innermost visible ring by an index match and got it wrong on 6 of 43 tilted 20-ID frames
    (seed distances 2.4x, 0.85x and 0.68x of the truth); a start whose distance is that far off
    does not converge whatever the tilt start. Within a couple of percent is enough. The beam centre
    still comes from ``make_seed``.

    ``fallback_pv``: if the gate FAILs after the iterated route (it does from some tilt-0
    starts, e.g. the 5 deg frame), also run :func:`autocalibrate_pv` from the seed (a wide
    capture window; blind to tilt up to about 14 deg; minutes, not seconds) and keep the better
    result by the gate.

    PASS means the rings sit where the geometry says to sub-pixel accuracy. It does NOT validate
    ``wavelength_A`` or the distance: they are degenerate and a fit absorbs a wrong wavelength
    (1.9 % too high still gave PASS, with the distance 1.8 % short).

    The returned ``verdict`` is the independent ring-position gate's: PASS (median < 0.3 px
    and p95 < 1 px with most populated cells finding their ring), WARN, or FAIL. Do not use a
    FAIL geometry.
    """
    from ..seed.auto_seed import _ring_table, make_seed
    from ..seed.calibrant import resolve_calibrant
    from .frozen_point import autocalibrate_frozen_point, iterate_frozen_point_until_stable

    t_all = time.time()
    raw = np.asarray(image)
    if raw.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {raw.shape}")
    if wavelength_A <= 0 or px_um <= 0:
        raise ValueError("wavelength_A and px_um must be positive")
    if lsd_um is not None and lsd_um <= 0:
        raise ValueError("lsd_um must be positive")
    img, bad = clean_frame(raw, mask)
    cal = resolve_calibrant(calibrant)
    ring_tth = [r[2] for r in _ring_table(cal, wavelength_A, n_rings=80)]
    notes: List[str] = []

    seed = make_seed(img, wavelength_A=wavelength_A, px_um=px_um, calibrant=calibrant,
                     use_diplib=False)
    if verbose:
        print(f"[tilted] seed: {seed}")
    if getattr(seed, "threshold_rung", 0) > 0:
        notes.append(f"seed found only after relaxing the arc threshold (rung "
                     f"{seed.threshold_rung}); it is a weaker result")
    if lsd_um is not None:
        notes.append(f"seed distance replaced by the hint {lsd_um/1e3:.1f} mm (make_seed gave {seed.Lsd_um/1e3:.1f} mm)")
        seed_lsd = float(lsd_um)
    else:
        seed_lsd = float(seed.Lsd_um)
    first_ring_px = seed_lsd * math.tan(math.radians(ring_tth[0])) / px_um
    pick = dict(point_pick_kwargs) if point_pick_kwargs else {"min_ring_gap_deg": 0.08}

    if nominal_tilt_deg is None:
        starts: List[Tuple[float, float]] = [(0.0, 0.0)]
    elif isinstance(nominal_tilt_deg, (int, float)):
        m = float(nominal_tilt_deg)
        starts = [(m * math.cos(math.radians(a)), m * math.sin(math.radians(a)))
                  for a in range(0, 360, 45)]
        notes.append("tilt magnitude given, direction unknown: 8 starts, best by fit strain")
    else:
        ty0, tz0 = (float(x) for x in nominal_tilt_deg)
        starts = [(ty0, tz0)]

    best = None
    v1_used = None
    for ty0, tz0 in starts:
        v1 = _v1_params(img, wavelength_A, px_um, cal, seed_lsd, seed.BC_y, seed.BC_z,
                        ty0, tz0, first_ring_px)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = iterate_frozen_point_until_stable(v1, img, point_pick_kwargs=pick, verbose=False)
        strain = float(getattr(r.fit, "mean_strain_uE", float("nan")))
        if verbose:
            print(f"[tilted] start (ty,tz)=({ty0:.2f},{tz0:.2f}): iterate converged={r.converged} "
                  f"n_iter={r.n_iter} strain={strain:.0f} microstrain")
        key = strain if math.isfinite(strain) else float("inf")
        if best is None or key < best[0]:
            best = (key, r, ty0, tz0)
            v1_used = v1
    _, it, _, _ = best
    geom_it = _unpack(it.res)
    cands = [("iterate", geom_it, it.res)]
    if polish:
        v1p = _v1_params(img, wavelength_A, px_um, cal, geom_it["Lsd"], geom_it["BC_y"],
                         geom_it["BC_z"], geom_it["ty"], geom_it["tz"], first_ring_px)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rp = autocalibrate_frozen_point(v1p, img, refine_distortion="radial",
                                                point_pick_kwargs=pick, verbose=False)
            cands.append(("iterate+radial pass", _unpack(rp), rp))
        except Exception as exc:                                    # keep the iterate result
            notes.append(f"distortion polish pass failed ({type(exc).__name__}: {exc}); "
                         "iterate geometry kept")
    def _score(cands_):
        out_ = []
        for name_, g_, res_ in cands_:
            gate_ = _gate(img, bad, g_, ring_tth, wavelength_A, px_um)
            if verbose:
                print(f"[tilted] {name_}: {gate_}")
            out_.append((_RANK[gate_.verdict],
                         gate_.median_px if math.isfinite(gate_.median_px) else 9e9,
                         name_, g_, res_, gate_))
        return out_

    scored = _score(cands)
    if fallback_pv and min(t[0] for t in scored) == _RANK["FAIL"]:
        if verbose:
            print("[tilted] gate FAIL after the iterated route: running autocalibrate_pv "
                  "(wide capture window, takes minutes) ...")
        from .single_pv import autocalibrate_pv
        v1pv = _v1_params(img, wavelength_A, px_um, cal, seed_lsd, seed.BC_y, seed.BC_z,
                          0.0, 0.0, first_ring_px)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rv = autocalibrate_pv(v1pv, img, mask=bad.astype(float), n_iter=5, verbose=False)
            scored += _score([("autocalibrate_pv", _unpack(rv), rv)])
        except Exception as exc:
            notes.append(f"autocalibrate_pv fallback failed ({type(exc).__name__}: {exc})")
    scored.sort(key=lambda t: (t[0], t[1]))
    _, _, name, geom, res, gate = scored[0]
    if name == "iterate" and len(cands) > 1:
        notes.append("the radial-distortion pass did not improve the gate; iterate geometry kept")
    if name == "autocalibrate_pv":
        notes.append("the iterated route failed the gate; autocalibrate_pv result kept")
    if not it.converged:
        notes.append("the iterate loop hit max_iter without settling (the start was probably "
                     "outside its capture range)")
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(math.radians(geom["ty"]))
                                                * math.cos(math.radians(geom["tz"]))))))
    verdict = gate.verdict
    if gate.verdict == "FAIL":
        notes.append("do NOT use this geometry. Supply a distance hint (lsd_um; the seed distance is "
                     "wrong in about 1 file in 7) and/or the nominal tilt (nominal_tilt_deg), or stop "
                     "and inspect the ring overlay")
    out = TiltedCalibration(
        verdict=verdict, geometry=geom, tilt_deg=tilt,
        perp_distance_mm=geom["Lsd"] / 1e3 * math.cos(math.radians(tilt)),
        gate=gate, route=name, converged=bool(it.converged), n_iter=int(it.n_iter),
        seed=dict(Lsd_um=float(seed.Lsd_um), BC_y=float(seed.BC_y), BC_z=float(seed.BC_z)),
        seconds=time.time() - t_all, notes=notes, result=res, v1_params=v1_used, bad_mask=bad)
    if verbose:
        print(out)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m midas_calibrate_v2.pipelines.tilted",
                                 description="Calibrate a tilted detector from a calibrant frame and gate the result.")
    ap.add_argument("image", help="TIFF / HDF5 / GE frame (any format read_image supports)")
    ap.add_argument("--wavelength", type=float, required=True, help="Angstrom, from the beamline")
    ap.add_argument("--px-um", type=float, required=True)
    ap.add_argument("--calibrant", default="CeO2")
    ap.add_argument("--nominal-tilt", default=None,
                    help="'M' = tilt magnitude in deg (direction unknown), or 'ty,tz' in deg")
    ap.add_argument("--lsd-mm", type=float, default=None,
                    help="distance hint along the beam in mm (e.g. the stage readback); replaces make_seed's distance")
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--data-loc", default="exchange/data", help="HDF5 dataset path")
    ap.add_argument("--im-trans", default="", help="MIDAS ImTransOpt codes applied when reading, e.g. '2' "
                    "(1 = flip Y, 2 = flip Z, 3 = transpose). The paramstest carries NO ImTransOpt: integrate "
                    "the same transformed frame, not the raw file")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    ap.add_argument("--paramstest", default=None, help="write the v1 paramstest here (not written on FAIL)")
    ap.add_argument("--mask-out", default=None, help="write the bad-pixel mask TIFF for midas-integrate-v2 --mask")
    a = ap.parse_args(argv)
    from ..io.readers import read_image

    im_trans = tuple(int(x) for x in a.im_trans.split(",") if x.strip())
    img, mask = read_image(a.image, data_loc=a.data_loc, im_trans=im_trans, return_mask=True)
    nom: Union[None, float, Tuple[float, float]] = None
    if a.nominal_tilt:
        parts = [float(x) for x in a.nominal_tilt.split(",")]
        nom = parts[0] if len(parts) == 1 else (parts[0], parts[1])
    res = calibrate_tilted(img, wavelength_A=a.wavelength, px_um=a.px_um, calibrant=a.calibrant,
                           mask=mask, nominal_tilt_deg=nom,
                           lsd_um=None if a.lsd_mm is None else a.lsd_mm * 1e3, polish=not a.no_polish)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res.to_dict(), fh, indent=1)
    if a.mask_out:
        res.write_mask(a.mask_out)
    if a.paramstest and res.verdict != "FAIL":
        res.write_paramstest(a.paramstest)
    return 0 if res.verdict == "PASS" else (1 if res.verdict == "WARN" else 2)


if __name__ == "__main__":
    raise SystemExit(main())

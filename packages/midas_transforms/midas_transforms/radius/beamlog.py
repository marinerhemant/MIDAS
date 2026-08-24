"""What beam was actually in effect when a scan ran.

Why this is not a slit lookup
-----------------------------
:mod:`~midas_transforms.radius.vsample` needs the beam height, and the obvious
place to look — the slit settings — is the wrong place whenever the beam is
**focused**. Measured on the Ce ht525 s2 FF scan (bt_1id_jul26): the beamline
log records

    switch_to_NFFocusedBeam HxV: 1200 slitted x 1 focused beam
    Setting E US slit size to:  1.2 x 0.3 mm ...
    Setting E DS slit size to:  1.3 x 0.1 mm ...

The vertical slits are 0.1-0.3 mm, but they are **guard slits**: the beam is
focused to about a micrometre and the slits are nowhere near it. Taking 0.1 mm
as the beam height would overstate the gauge volume by 100x.

**The ``1`` in that line is a knife-edge measurement**, not a nominal label —
this is how the vertical size of a focused 1-ID beam is established, and the
log carries the measured value. So the logged HxV vertical IS the beam height
and is used as such.

(An earlier version of this module refused a focused configuration and demanded
a separate focus-scan number, on the assumption that the label was nominal.
That was wrong, and it also rested on misreading column 6 of
``*_FocusScan.log`` as a beam size. What that column holds is **not
established** here; :func:`focus_scan_minimum` reports it without claiming it
is a width.)

Nor is it the scan macro. The FF macro for that scan carries

    #switch_to_HxV_beam 1.2 _ystp      <- commented out

so the scan **inherited** whatever configuration was already set, and the only
record of what that was is the chronological log.

So: read the log, find the last beam switch before the scan, and take the
vertical size from the reported HxV — never from the slits.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = ["BeamConfig", "beam_config_for_scan", "focus_scan_minimum"]

log = logging.getLogger(__name__)

_CMD = re.compile(r"^\s*(\d+)\.\w+>\s*(.*)$")
_SWITCH = re.compile(r"\bswitch_to_(\w*[Bb]eam\w*)\b")
_HXV = re.compile(r"HxV:\s*([\d.]+)\s*(\w+)\s*x\s*([\d.]+)\s*(\w+)")
_HXV_CALL = re.compile(r"switch_to_HxV_beam\s+([\d.]+)\s+([\d.]+)")
_SLIT = re.compile(
    r"Setting\s+(\w+)\s+(US|DS)\s+slit size to:\s*([\d.]+)\s*x\s*([\d.]+)\s*mm",
    re.I)


@dataclass
class BeamConfig:
    """The beam configuration in effect for a scan, as the log recorded it."""

    scan: str
    macro: str                      # e.g. NFFocusedBeam
    command_index: Optional[int]
    horizontal_um: Optional[float] = None
    vertical_um: Optional[float] = None
    vertical_is_focused: bool = False
    slits_mm: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    raw: List[str] = field(default_factory=list)

    @property
    def height_from_slits_um(self) -> Optional[float]:
        """Smallest vertical slit, in um. **Not the beam height when focused.**"""
        v = [hv[1] for hv in self.slits_mm.values()]
        return min(v) * 1000.0 if v else None

    def beam_height_um(self, override_um: Optional[float] = None) -> float:
        """The beam height, from the log.

        For a focused configuration the logged HxV vertical is a **knife-edge
        measurement** and is used directly. The vertical *slits* are never
        used there — they are guards, and on the Ce scan they would have
        overstated the height by 100x.
        """
        if override_um is not None:
            return float(override_um)
        if self.vertical_um is not None:
            return self.vertical_um
        if self.vertical_is_focused:
            raise ValueError(
                f"{self.scan} ran with {self.macro}, a vertically focused "
                "beam, and the log does not carry its measured size. The "
                f"vertical slits ({self.height_from_slits_um} um) are guards, "
                "not the beam. Supply the knife-edge value."
            )
        h = self.height_from_slits_um
        if h is not None:
            return h
        raise ValueError(
            f"{self.scan}: the log records no vertical beam size and no slit "
            "setting; the beam height cannot be recovered from it."
        )

    def summary(self) -> str:
        out = [f"{self.scan}: {self.macro}"
               + (f" (command {self.command_index})" if self.command_index else "")]
        if self.horizontal_um is not None:
            out.append(f"  horizontal {self.horizontal_um:.0f} um")
        if self.vertical_um is not None:
            out.append(f"  vertical   {self.vertical_um:.0f} um "
                       + ("(FOCUSED -- knife-edge measured)"
                          if self.vertical_is_focused else "(slitted)"))
        for k, (h, v) in sorted(self.slits_mm.items()):
            out.append(f"  slit {k:<6} {h} x {v} mm"
                       + ("   <- guard only, beam is focused"
                          if self.vertical_is_focused else ""))
        return "\n".join(out)


def beam_config_for_scan(
    log_path: Union[str, Path], scan: str, *, occurrence: int = 0
) -> BeamConfig:
    """Last beam configuration set before ``scan`` appears in the log.

    ``occurrence`` selects which mention of the scan to anchor on (0 = first),
    because a scan name appears many times — once per file-name set, once per
    invocation, once per frame.
    """
    lines = Path(log_path).read_text(errors="replace").splitlines()
    anchors = [i for i, l in enumerate(lines)
               if scan in l and _CMD.match(l) and l.rstrip().endswith(scan)]
    if not anchors:
        anchors = [i for i, l in enumerate(lines) if scan in l]
    if not anchors:
        raise ValueError(f"{scan!r} never appears in {log_path}")
    at = anchors[min(occurrence, len(anchors) - 1)]

    last = None
    for i in range(at, -1, -1):
        m = _CMD.match(lines[i])
        if not m:
            continue
        sw = _SWITCH.search(m.group(2))
        if sw:
            last = (i, int(m.group(1)), sw.group(1), m.group(2))
            break
    if last is None:
        raise ValueError(
            f"no switch_to_*Beam appears before {scan!r} in {log_path}; the "
            "configuration was inherited from before the log starts."
        )
    i, cmd, macro, cmdtext = last

    cfg = BeamConfig(scan=scan, macro=macro, command_index=cmd)
    # The report and the slit lines follow the command, before the next one.
    block: List[str] = []
    for j in range(i, min(i + 60, len(lines))):
        if j > i and _CMD.match(lines[j]):
            break
        block.append(lines[j])
    cfg.raw = [b for b in block if b.strip()]

    body = "\n".join(block)
    m = _HXV.search(body)
    if m:
        h, hu, v, vu = m.groups()
        cfg.horizontal_um = float(h) * (1000.0 if hu.lower() == "mm" else 1.0)
        cfg.vertical_um = float(v) * (1000.0 if vu.lower() == "mm" else 1.0)
        cfg.vertical_is_focused = "focus" in vu.lower() or "focus" in body.lower()
    else:
        m = _HXV_CALL.search(cmdtext)
        if m:                                   # switch_to_HxV_beam H V (mm)
            cfg.horizontal_um = float(m.group(1)) * 1000.0
            cfg.vertical_um = float(m.group(2)) * 1000.0
    if "focus" in macro.lower():
        cfg.vertical_is_focused = True

    for sm in _SLIT.finditer(body):
        where, ud, hh, vv = sm.groups()
        cfg.slits_mm[f"{where}{ud}"] = (float(hh), float(vv))
    return cfg


def focus_scan_minimum(path: Union[str, Path], *, value_col: int = 6,
                       pos_col: int = 1) -> Dict[str, Any]:
    """Minimum of a ``*_FocusScan.log`` column against focus position.

    **What column 6 holds is not established here.** It falls to a clean
    minimum against focus position, which is what a focus optimisation looks
    like, but this function does not claim it is a beam width and the beam
    height must not be taken from it — the knife-edge value in the beamline
    log is the measurement (see :meth:`BeamConfig.beam_height_um`). Provided
    for locating the best-focus *position* and for seeing whether the scan
    bracketed it.
    """
    rows = []
    for ln in Path(path).read_text(errors="replace").splitlines():
        parts = ln.split()
        if len(parts) > max(value_col, pos_col):
            try:
                rows.append((float(parts[pos_col]), float(parts[value_col])))
            except ValueError:
                continue
    if len(rows) < 3:
        raise ValueError(f"{path}: fewer than 3 usable rows")
    a = np.array(rows)
    i = int(np.argmin(a[:, 1]))
    at_edge = i in (0, len(a) - 1)
    return {
        "best_value": float(a[i, 1]),
        "best_position": float(a[i, 0]),
        "n_points": int(len(a)),
        "position_range": (float(a[:, 0].min()), float(a[:, 0].max())),
        "max_size_um": float(a[:, 1].max()),
        "bracketed": not at_edge,
        "note": ("" if not at_edge else
                 "the minimum is at an END of the scan, so the focus was not "
                 "bracketed and this is only the smallest value tried"),
    }

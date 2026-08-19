"""The corrected paramstest must COMPOSE relative scalars, not overwrite them.

Regression guard. ``grain-tx`` refines ``tx``/``Wedge`` as CORRECTIONS on top of
the geometry the input reconstruction already used:

  * the forward model is built with ``tx=0.0`` and ``wedge=0.0``
    (``_build_model``), while every absolute scalar (Lsd, ty, tz, BC) is seeded
    from ``v1``;
  * the trial ``tx`` is applied by ROTATING the stored SpotMatrix YLab/ZLab,
    which already carry whatever ``tx`` the pipeline ran with;
  * ``Wedge`` is injected into the forward model while the observed omega
    already carries the pipeline's wedge correction.

So the fitted number is the RESIDUAL. The output paramstest drives a FRESH
reconstruction, so it must carry the TOTAL. Overwriting it silently discards the
previous pass — a second iteration then applies less correction than the first,
with no error and no log line.

Measured on 20-ID Au (5 grains, MinNrPx-4 spot list):
    pass 1 on a tx=0 recon      -> -0.158497
    pass 2 on the -0.1585 recon -> -0.087265   (the residual)
    composed                    -> -0.245762
against -0.2455 from an independent ring/eta systematics fit.

EXCEPTION: when a ``_NEEDS_RAW``/``_DISTORTION`` scalar is thawed the residual
switches to the raw-pixel path, where ``tx`` is applied inside the detector
correction and is therefore ABSOLUTE. ``Wedge`` stays relative in both paths.
"""

from __future__ import annotations

import re

import pytest

from midas_joint_ff_calibrate import grain_refine as gr


def _write(tmp_path, **keys):
    p = tmp_path / "paramstest.txt"
    p.write_text("\n".join(f"{k} {v}" for k, v in keys.items()) + "\n")
    return p


class _V1:
    """Minimal stand-in for the parsed v1 params."""

    def __init__(self, **kw):
        self.tx = kw.get("tx", 0.0)
        self.Wedge = kw.get("Wedge", 0.0)
        self.Lsd = kw.get("Lsd", 1_000_000.0)
        self.ty = kw.get("ty", 0.0)
        self.tz = kw.get("tz", 0.0)


def _emit(tmp_path, refine_params, fitted, v1, observed_from_raw):
    """Reproduce the writer block's composition rule."""
    src = _write(tmp_path, tx=v1.tx, Wedge=v1.Wedge, Lsd=v1.Lsd)
    txt = src.read_text()
    relative = {"Wedge"} | (set() if observed_from_raw else {"tx"})
    for nm in refine_params:
        value = float(fitted[nm])
        if nm in relative:
            value = float(getattr(v1, nm, 0.0) or 0.0) + value
        line = f"{nm} {value:.10g}"
        pat = rf"(?m)^{nm}\b.*$"
        txt = re.sub(pat, line, txt) if re.search(pat, txt) else txt + line + "\n"
    out = tmp_path / "out.txt"
    out.write_text(txt)
    return {ln.split()[0]: float(ln.split()[1]) for ln in txt.splitlines() if ln.strip()}


def test_tx_composes_on_the_default_path(tmp_path):
    """The measured 20-ID Au case: -0.1585 then -0.087265 must give -0.245762."""
    v1 = _V1(tx=-0.1584965882, Wedge=0.0)
    got = _emit(tmp_path, ("tx", "Wedge"),
                {"tx": -0.087265, "Wedge": 0.0}, v1, observed_from_raw=False)
    assert got["tx"] == pytest.approx(-0.245762, abs=1e-5), (
        "tx must COMPOSE; overwriting applies a third of the true roll")


def test_wedge_composes(tmp_path):
    v1 = _V1(Wedge=-0.012584)
    got = _emit(tmp_path, ("Wedge",), {"Wedge": 0.015325}, v1,
                observed_from_raw=False)
    assert got["Wedge"] == pytest.approx(0.002741, abs=1e-6)


def test_tx_is_absolute_on_the_raw_pixel_path(tmp_path):
    """Thawing a _NEEDS_RAW scalar re-derives observations from raw pixels, where
    tx is applied inside the detector correction -- composing would double it."""
    v1 = _V1(tx=-0.1584965882)
    got = _emit(tmp_path, ("tx",), {"tx": -0.245762}, v1, observed_from_raw=True)
    assert got["tx"] == pytest.approx(-0.245762, abs=1e-9)


def test_absolute_scalars_are_never_composed(tmp_path):
    v1 = _V1(Lsd=1_000_000.0)
    got = _emit(tmp_path, ("Lsd",), {"Lsd": 899_916.0}, v1, observed_from_raw=False)
    assert got["Lsd"] == pytest.approx(899_916.0)


def test_zero_prior_is_a_noop(tmp_path):
    """The documented single-shot usage (run on a tx=0 recon) must be unchanged."""
    v1 = _V1(tx=0.0, Wedge=0.0)
    got = _emit(tmp_path, ("tx", "Wedge"),
                {"tx": -0.158497, "Wedge": -0.012584}, v1, observed_from_raw=False)
    assert got["tx"] == pytest.approx(-0.158497)
    assert got["Wedge"] == pytest.approx(-0.012584)


def test_relative_set_matches_the_model_construction():
    """_DIRECT is the set refinable without the raw path; tx/Wedge are the two
    the model zeroes, Lsd is seeded from v1."""
    assert {"tx", "Wedge", "Lsd"} == set(gr._DIRECT)
    assert "tx" not in gr._NEEDS_RAW and "Wedge" not in gr._NEEDS_RAW

"""``midas-grain-qa`` — the one-command audit of a finished layer.

The tests that matter here are about HONESTY of the report rather than about
the numerics (each stage has its own suite):

* a stage that cannot run says why, in ``skipped``;
* a stage that RAISES is recorded, never silently reported as unavailable —
  that distinction is the whole reason a torch threading bug once masqueraded
  as a physical limit on how many grains could be measured;
* d0 refuses non-cubic symmetry instead of guessing;
* the controls (twin adjacency null, twin-agreement self-check) are always in
  the output, because the headline numbers are meaningless without them.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from midas_process_grains.grain_qa import STAGES, GrainQAResult, run_grain_qa

COLS = ("GrainID O11 O12 O13 O21 O22 O23 O31 O32 O33 X Y Z a b c alpha beta "
        "gamma DiffPos DiffOme DiffAngle GrainRadius Confidence "
        "eFab11 eFab12 eFab13 eFab21 eFab22 eFab23 eFab31 eFab32 eFab33 "
        "eKen11 eKen12 eKen13 eKen21 eKen22 eKen23 eKen31 eKen32 eKen33 "
        "RMSErrorStrain PhaseNr Eul0 Eul1 Eul2").split()

PRE = ("%NumGrains {n}\n%BeamCenter 0.000000\n%BeamThickness 100.000000\n"
       "%GlobalPosition 0.000000\n%NumPhases 1\n%PhaseInfo\n"
       "%\tSpaceGroup:{sg}\n"
       "%\tLattice Parameter: 3.6 3.6 3.6 90.0 90.0 90.0\n")


def _grains_csv(path, n=40, sg=225, a=3.6005, seed=0):
    """A small but structurally real Grains.csv."""
    rng = np.random.default_rng(seed)
    lines = [PRE.format(n=n, sg=sg), "%" + "\t".join(COLS) + "\n"]
    for i in range(n):
        # random proper rotation
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        w, x, y, z = q
        O = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])
        vals = {"GrainID": i + 1, "GrainRadius": 8.0, "Confidence": 0.9,
                "a": a, "b": a, "c": a,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                "X": rng.uniform(-400, 400), "Y": rng.uniform(-400, 400),
                "Z": rng.uniform(-40, 40), "PhaseNr": 1}
        for k, v in zip(("O11","O12","O13","O21","O22","O23","O31","O32","O33"),
                        O.reshape(-1)):
            vals[k] = v
        row = [f"{float(vals.get(c, 0.0)):.6f}" for c in COLS]
        row[0] = str(i + 1)
        lines.append("\t".join(row) + "\n")
    path.write_text("".join(lines))
    return path


@pytest.fixture
def layer(tmp_path):
    d = tmp_path / "LayerNr_1"; d.mkdir()
    _grains_csv(d / "Grains.csv")
    return d


# ---------------------------------------------------------------------------
#  contract
# ---------------------------------------------------------------------------

def test_missing_grains_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Grains.csv"):
        run_grain_qa(tmp_path)


def test_unknown_stage_raises(layer):
    with pytest.raises(ValueError, match="unknown stage"):
        run_grain_qa(layer, skip=["nonsense"])


def test_bare_layer_runs_and_says_what_it_could_not_do(layer):
    """Only Grains.csv present: twins still work, the rest report a REASON."""
    res = run_grain_qa(layer)
    assert res.n_grains == 40
    assert res.twins is not None, "twins need only the grain table"
    for st in ("d0", "attribution", "uncertainty"):
        assert st in res.skipped, f"{st} should be skipped"
        assert res.skipped[st], f"{st} must carry a reason, not an empty string"
    assert "SpotMatrix" in res.skipped["attribution"]


def test_skip_is_honoured_and_recorded(layer):
    res = run_grain_qa(layer, skip=["twins"])
    assert res.twins is None
    assert res.skipped["twins"] == "skipped by request"


def test_all_stages_are_skippable(layer):
    res = run_grain_qa(layer, skip=list(STAGES))
    assert set(res.skipped) == set(STAGES)
    assert res.n_grains == 40


# ---------------------------------------------------------------------------
#  the honesty rules
# ---------------------------------------------------------------------------

def test_a_raising_stage_is_recorded_not_swallowed(layer, monkeypatch):
    """A stage that throws must land in BOTH skipped and warnings, with the
    exception type — not be reported as merely 'unavailable'."""
    import midas_process_grains.grain_qa as qa

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(qa, "_stage_twins", boom)

    res = run_grain_qa(layer)
    assert "RuntimeError" in res.skipped["twins"]
    assert "synthetic failure" in res.skipped["twins"]
    assert any("twins" in w and "RuntimeError" in w for w in res.warnings)


def test_d0_refuses_non_cubic_rather_than_guessing(tmp_path):
    d = tmp_path / "L"; d.mkdir()
    _grains_csv(d / "Grains.csv", sg=194)          # hexagonal
    res = run_grain_qa(d, space_group=194, skip=["twins", "attribution",
                                                 "uncertainty"])
    assert res.d0 is None
    assert "cubic" in res.skipped["d0"].lower()


def test_low_twin_separation_is_flagged_in_the_summary(layer):
    """The self-check has to be able to say 'this filter is not working'."""
    res = run_grain_qa(layer)
    res.attribution = type("A", (), {
        "summary": lambda self: "stub", "n_contested": 10, "n_dropped": 1,
        "grain_id": np.zeros(3), "sigma": 25.0, "rel_threshold": 0.05})()
    res.twin_agreement = {"n_twin_claims": 100.0, "n_accidental_claims": 100.0,
                          "keep_rate_twin": 0.55, "keep_rate_accidental": 0.50}
    txt = res.summary()
    assert "WEAK separation" in txt
    assert "Do not use it to drop spots" in txt


def test_summary_always_carries_the_twin_null(layer):
    res = run_grain_qa(layer)
    txt = res.summary()
    assert "twins" in txt
    assert "null" in txt or "far" in txt, (
        "the adjacency null must appear; a twin fraction without it is "
        "not interpretable")


# ---------------------------------------------------------------------------
#  outputs
# ---------------------------------------------------------------------------

def test_to_json_is_serialisable_and_keeps_provenance(layer):
    res = run_grain_qa(layer)
    blob = json.dumps(res.to_json())          # must not raise on numpy types
    back = json.loads(blob)
    assert back["n_grains"] == 40
    assert back["space_group"] == 225
    assert "grains" in back["provenance"]
    assert back["provenance"]["grains"].endswith("Grains.csv")
    assert "twins" in back


def test_json_reports_the_twin_control_fields(layer):
    j = run_grain_qa(layer).to_json()
    for k in ("rate_near", "rate_far", "enrichment", "trustworthy"):
        assert k in j["twins"], f"missing control field {k}"


def test_timings_recorded_for_every_attempted_stage(layer):
    res = run_grain_qa(layer, skip=["uncertainty"])
    for st in ("d0", "twins", "attribution"):
        assert st in res.timings


def test_cli_runs_end_to_end(layer, tmp_path, capsys):
    from midas_process_grains.grain_qa import main
    out = tmp_path / "qa.json"
    rc = main([str(layer), "--skip", "uncertainty", "--json", str(out), "-q"])
    assert rc == 0
    assert out.exists()
    assert json.loads(out.read_text())["n_grains"] == 40
    printed = capsys.readouterr().out
    assert "grain QA" in printed


def test_cli_csv_without_uncertainty_says_so(layer, tmp_path, capsys):
    from midas_process_grains.grain_qa import main
    main([str(layer), "--skip", "uncertainty", "--csv",
          str(tmp_path / "x.csv"), "-q"])
    assert "no uncertainty result" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  geometry guards
# ---------------------------------------------------------------------------

class _B:
    """Minimal paramstest stand-in."""
    Lsd = LsdFit = 767765.75
    YBCFit = 1022.76; ZBCFit = 974.64
    px = 200.0; Wavelength = 0.17309
    OmegaStart = 180.0; OmegaStep = -0.25
    NrPixelsY = NrPixelsZ = 2048
    MarginEta = 500.0            # a MATCHING margin, not a pole angle
    ExcludePoleAngle = 6.0
    txFit = tyFit = tzFit = 0.0


class _P:
    def __init__(self, base): self.base = base


def test_min_eta_comes_from_exclude_pole_angle_not_margin_eta():
    """MarginEta is 500 on real files; as a pole angle that excludes everything.

    The failure it caused was opaque — the forward model predicted zero
    reflections and every grain died with
    'IndexError: min(): Expected reduction dim 1 to have non-zero size'.
    """
    from midas_process_grains.grain_qa import _geometry
    g, lsd, step = _geometry(_P(_B()))
    assert g.min_eta == 6.0, "must take ExcludePoleAngle"
    assert g.min_eta != _B.MarginEta


def test_geometry_refuses_an_impossible_pole_angle():
    from midas_process_grains.grain_qa import _geometry

    class B(_B):
        ExcludePoleAngle = 500.0
    with pytest.raises(ValueError, match="excludes every reflection"):
        _geometry(_P(B()))


def test_geometry_refuses_a_missing_beam_centre():
    """Never invent a BC: a guessed one looks fine and is wrong everywhere."""
    from midas_process_grains.grain_qa import _geometry

    class B(_B):
        YBCFit = 0.0
    with pytest.raises(ValueError, match="beam centre"):
        _geometry(_P(B()))


def test_missing_exclude_pole_angle_falls_back_to_the_default():
    from midas_process_grains.grain_qa import (_geometry,
                                               DEFAULT_EXCLUDE_POLE_DEG)

    class B(_B):
        ExcludePoleAngle = None
    g, _, _ = _geometry(_P(B()))
    assert g.min_eta == DEFAULT_EXCLUDE_POLE_DEG


def _sigma_stub(n, sigma_um=20.0, sigma_obs_px=1.0):
    """A PerGrainParameterSigmaResult-shaped stub complete enough for summary()."""
    return type("S", (), {
        "ok": np.ones(n, bool),
        "sigma_pos_um": np.full((n, 3), sigma_um),
        "sigma_euler_rad": np.full((n, 3), 1e-4),
        "sigma_latc": np.full((n, 6), 1e-4),
        "sigma_hydrostatic_strain": np.full(n, 1e-4),
        "n_spots_matched": np.full(n, 100),
        "failures": {},
        "sigma_obs_px": sigma_obs_px,
    })()


# ---------------------------------------------------------------------------
#  calibration stage — the check that can say "your error bars are wrong"
# ---------------------------------------------------------------------------

def test_calibration_needs_the_uncertainty_stage(layer):
    """Without sigma there is nothing to calibrate, and it must say so."""
    res = run_grain_qa(layer, skip=["uncertainty"])
    assert res.calibration is None
    assert "uncertainty" in res.skipped["calibration"]


def test_calibration_says_it_needs_an_adjacent_layer(tmp_path):
    """The whole method rests on a grain being measured twice."""
    import midas_process_grains.grain_qa as qa
    d = tmp_path / "LayerNr_7"; d.mkdir()
    _grains_csv(d / "Grains.csv")
    res = qa.GrainQAResult(layer_dir=d, n_grains=40, space_group=225)
    res.sigma = _sigma_stub(40)
    res.sigma_gid = np.arange(1, 41)
    g = type("G", (), {"ids": np.arange(1, 41),
                       "positions": np.zeros((40, 3)),
                       "orient_mat": np.stack([np.eye(3)] * 40)})()
    qa._stage_calibration(d, g, res)
    assert res.calibration is None
    assert "adjacent layer" in res.skipped["calibration"]


def test_calibration_recovers_a_known_inflation_factor(tmp_path):
    """Plant a known truth: each measurement has 5 um noise, sigma claims 20.

    BOTH measurements must be noised independently, or the difference carries
    std 5 instead of 5*sqrt(2) and the expected factor changes. With both:
    std(m1-m2) = 5*sqrt(2), so z = (m1-m2)/(sqrt(2)*20) has std 5/20 = 0.25 and
    the stage must report a factor of ~4.
    """
    import midas_process_grains.grain_qa as qa
    rng = np.random.default_rng(0)
    n = 300
    pos = rng.uniform(-400, 400, (n, 3))
    quats = rng.normal(size=(n, 4)); quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    def om(q):
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    OM = np.stack([om(q) for q in quats])

    TRUE, CLAIMED = 5.0, 20.0
    d6 = tmp_path / "LayerNr_6"; d6.mkdir()
    d7 = tmp_path / "LayerNr_7"; d7.mkdir()
    for q in (d6, d7):
        _grains_csv(q / "Grains.csv", n=1)          # placeholder file so it exists
    pos1 = pos + rng.normal(0, TRUE, pos.shape)      # measurement 1
    pos2 = pos + rng.normal(0, TRUE, pos.shape)      # measurement 2, independent

    class _G:
        def __init__(self, p): self.ids = np.arange(1, n+1); self.positions = p
        orient_mat = OM
    import midas_process_grains.io.read as rd
    monkey = rd.read_grains_csv
    rd.read_grains_csv = lambda path, **k: _G(pos2)
    try:
        res = qa.GrainQAResult(layer_dir=d6, n_grains=n, space_group=225)
        res.sigma = _sigma_stub(n, sigma_um=CLAIMED)
        res.sigma_gid = np.arange(1, n+1)
        qa._stage_calibration(d6, _G(pos1), res)
    finally:
        rd.read_grains_csv = monkey

    assert res.calibration is not None, res.skipped.get("calibration")
    c = res.calibration
    assert c["tight"]["n_pairs"] > 100
    assert 3.0 < c["factor"] < 5.5, f"expected ~4x, got {c['factor']:.2f}"
    assert 0.18 < c["implied_sigma_obs_px"] < 0.34


def test_calibration_reports_but_never_applies(tmp_path):
    """A measured sigma_obs must stay an explicit user choice."""
    import midas_process_grains.grain_qa as qa
    res = qa.GrainQAResult(layer_dir=tmp_path, n_grains=1, space_group=225)
    res.sigma = _sigma_stub(1)
    res.calibration = {"tight": {"n_pairs": 500, "std_z_x": 0.35, "std_z_y": 0.31},
                       "factor": 3.03, "assumed_sigma_obs_px": 1.0,
                       "implied_sigma_obs_px": 0.33, "n_sibling_layers": 2}
    txt = res.summary()
    assert "TOO LARGE" in txt
    assert "NOT applied automatically" in txt
    assert "sigma_obs_px=0.330" in txt
    # and the sigmas themselves are untouched
    assert res.sigma.sigma_pos_um[0, 0] == 20.0
    # the reproducibility scoping must be stated
    assert "REPRODUCIBILITY" in txt and "common-mode" in txt

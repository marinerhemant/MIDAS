"""Static checks on the shipped notebook.

Not execution: the notebook writes ~60 MB of synthetic frames and reconstructs
them, which is too slow for the unit suite and needs the compiled engine. Run
it for real before a release:

    cd notebooks && jupyter nbconvert --to notebook --execute --inplace \\
        01_dt_recon_walkthrough.ipynb

What these catch instead is the cheap rot that is invisible until a user opens
the notebook -- which is the worst possible moment. Every code cell must parse,
every ``midas_dt`` name it calls must still exist, and the two warnings about
beamline conventions must still be there, because those are the whole reason a
new instrument can use this at all.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
#: Checks below are per-notebook by topic: the calibration notebook has no
#: beamline-convention or scope section, and the walkthrough has no beam-centre
#: guidance. Applying every check to every notebook would only teach us to
#: weaken the checks.
WALKTHROUGH = [p for p in NOTEBOOKS if "recon_walkthrough" in p.name]
CALIBRATION = [p for p in NOTEBOOKS if "calibration" in p.name]


def _cells(nb: Path, kind: str) -> list[str]:
    doc = json.loads(nb.read_text())
    return ["".join(c["source"]) for c in doc["cells"] if c["cell_type"] == kind]


def test_there_is_a_notebook_to_check():
    """A glob that matches nothing makes every parametrised test below vacuous."""
    assert NOTEBOOKS, f"no notebooks found under {NOTEBOOK_DIR}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_parses(nb):
    for i, src in enumerate(_cells(nb, "code")):
        # Strip IPython magics, which are not Python.
        clean = "\n".join("" if ln.strip().startswith(("%", "!")) else ln
                          for ln in src.splitlines())
        try:
            ast.parse(clean)
        except SyntaxError as exc:
            pytest.fail(f"{nb.name} code cell {i} does not parse: {exc}")


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_public_names_it_calls_still_exist(nb):
    """Guards against a rename landing in the package but not the notebook."""
    import midas_dt

    src = "\n".join(_cells(nb, "code"))
    called = set()
    for line in src.splitlines():
        if line.startswith("from midas_dt import") and "(" not in line:
            called.update(n.strip() for n in
                          line.split("import", 1)[1].split(","))
    called.discard("")
    assert called, "no `from midas_dt import ...` found; has the notebook changed shape?"
    missing = sorted(n for n in called if not hasattr(midas_dt, n))
    assert not missing, f"{nb.name} imports names midas_dt no longer exports: {missing}"


@pytest.mark.parametrize("nb", WALKTHROUGH, ids=lambda p: p.name)
def test_the_beamline_convention_warnings_survive(nb):
    """The two settings that fail silently on a non-1-ID instrument.

    ``negate_omega`` wrong reconstructs a mirror image; ``drop_first_frame``
    wrong shifts every projection by one angular step. Both look fine. A
    notebook aimed at a new beamline that stops warning about them is worse
    than no notebook, so this fails if the warnings are edited out.
    """
    text = "\n".join(_cells(nb, "markdown") + _cells(nb, "code")).lower()
    assert "negate_omega" in text
    assert "drop_first_frame" in text
    assert "mirror" in text, "the consequence of a wrong omega sign is not stated"
    assert "1-id" in text, "the notebook does not say where the defaults come from"


@pytest.mark.parametrize("nb", WALKTHROUGH, ids=lambda p: p.name)
def test_it_states_the_scope_boundary(nb):
    """Continuous rings or scanning-3DXRD. A user pointed at the wrong
    technique wastes a beamtime, so the notebook has to say so."""
    text = "\n".join(_cells(nb, "markdown")).lower()
    assert "continuous" in text
    assert "3dxrd" in text


@pytest.mark.parametrize("nb", WALKTHROUGH, ids=lambda p: p.name)
def test_demo_mode_is_the_default(nb):
    """It must run for someone with no data, or it cannot be demonstrated."""
    src = "\n".join(_cells(nb, "code"))
    assert "USE_DEMO_DATA = True" in src


def test_demo_helper_builds_a_readable_scan(tmp_path):
    """The one dynamic check, kept small: _demo.py must write files the
    package's own reader can open, since that is the point of it."""
    import sys
    sys.path.insert(0, str(NOTEBOOK_DIR))
    from _demo import make_scan

    from midas_dt import DTScan, RawFormat, frames_in_file

    d = make_scan(tmp_path / "demo", n_pixels=48, n_translations=4,
                  n_rotations=6, ring_radii_px=(12.0, 18.0))
    fmt = RawFormat(n_pixels_y=d.n_pixels, n_pixels_z=d.n_pixels,
                    flip_vertical=False)
    first = d.directory / f"{d.stem}_{d.start_nr:06d}.raw"

    # One extra frame on disk: the throwaway, so drop_first_frame=True is right.
    assert frames_in_file(first, fmt) == d.n_rotations + 1

    scan = DTScan.from_stem(d.directory, d.stem, d.start_nr, d.end_nr, fmt=fmt,
                            start_omega=d.start_omega, omega_step=d.omega_step,
                            negate_omega=False, dark_file=d.dark_file,
                            drop_first_frame=True)
    assert scan.n_translations == 4
    assert scan.n_frames == 6
    frame = scan.frame(1, 2)
    assert frame.shape == (48, 48)
    assert frame.max() > frame.min(), "the demo frame carries no signal"


# ------------------------------------------------- calibration notebook only
@pytest.mark.parametrize("nb", CALIBRATION, ids=lambda p: p.name)
def test_calibration_demo_mode_is_the_default(nb):
    assert "USE_DEMO_CALIBRANT = True" in "\n".join(_cells(nb, "code"))


@pytest.mark.parametrize("nb", CALIBRATION, ids=lambda p: p.name)
def test_it_warns_against_inventing_a_beam_centre(nb):
    """The failure that cost 1040 ue against 47 ue on the same frame.

    Passing a made-up BC_guess overrides the auto-seeder and the fit cannot
    travel. A calibration notebook that does not warn about it will produce
    exactly that failure in a user's hands.
    """
    text = "\n".join(_cells(nb, "markdown")).lower()
    assert "bc_guess" in text
    assert "seeder" in text
    assert "1040" in text, "the measured consequence is not quoted"


@pytest.mark.parametrize("nb", CALIBRATION, ids=lambda p: p.name)
def test_it_states_the_hard_strain_rule(nb):
    """A calibration above 100 ue has failed -- not 'is worse'."""
    text = "\n".join(_cells(nb, "markdown") + _cells(nb, "code")).lower()
    assert "100" in text
    assert any(w in text for w in ("failed", "fail"))


@pytest.mark.parametrize("nb", CALIBRATION, ids=lambda p: p.name)
def test_it_tells_macos_users_calibration_will_not_run(nb):
    """Measured: calibrate() segfaults on macOS, in a child process too."""
    text = "\n".join(_cells(nb, "markdown")).lower()
    assert "macos" in text
    assert "segfault" in text


@pytest.mark.parametrize("nb", CALIBRATION, ids=lambda p: p.name)
def test_it_hands_the_geometry_on_rather_than_asking_for_retyping(nb):
    src = "\n".join(_cells(nb, "code"))
    assert "calibration.json" in src
    assert "DTGeometry" in src

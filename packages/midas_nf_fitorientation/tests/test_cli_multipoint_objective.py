"""``midas-nf-fit-multipoint --objective`` dispatches to the right driver.

Regression test for the incident where the CLI always ran the soft/dense
path (``fit_multipoint_run``), which OOMs at full detector resolution
(393 GiB requested on a 47 GiB GPU on 20-ID data) -- the C-equivalent hard
path (``fit_multipoint_hard_run``) existed but had no CLI wrapper. Default
must be "hard"; "soft" must remain reachable for the differentiable path.
"""
from __future__ import annotations

from midas_nf_fitorientation import cli


def test_default_objective_calls_hard_run(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        cli, "fit_multipoint_hard_run",
        lambda paramfile, **kw: calls.update(kind="hard", paramfile=paramfile, kw=kw),
    )
    monkeypatch.setattr(
        cli, "fit_multipoint_run",
        lambda *a, **kw: calls.update(kind="soft"),
    )
    rc = cli.fit_multipoint_main(["params.txt", "4"])
    assert rc == 0
    assert calls["kind"] == "hard"
    assert calls["paramfile"] == "params.txt"
    assert calls["kw"]["n_cpus"] == 4


def test_explicit_soft_objective_calls_soft_run(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        cli, "fit_multipoint_hard_run",
        lambda *a, **kw: calls.update(kind="hard"),
    )
    monkeypatch.setattr(
        cli, "fit_multipoint_run",
        lambda paramfile, **kw: calls.update(kind="soft", paramfile=paramfile, kw=kw),
    )
    rc = cli.fit_multipoint_main(["params.txt", "--objective", "soft"])
    assert rc == 0
    assert calls["kind"] == "soft"
    assert calls["paramfile"] == "params.txt"
    assert "lbfgs_config" in calls["kw"]


def test_bad_objective_value_rejected():
    import pytest
    with pytest.raises(SystemExit):
        cli.fit_multipoint_main(["params.txt", "--objective", "bogus"])

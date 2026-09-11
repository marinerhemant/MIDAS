# Calibrate → Integrate runbook — what is true right now

> Part of the **calibrate-integrate doc set**. Spine: [`README.md`](README.md).
> The spine is the procedure; this file is the state. If they disagree, this one
> is the thing to re-check, because procedures age more slowly than facts.

**Owner:** Hemant Sharma. **Last reviewed 2026-08-22.**

## Current state — the pick-up point

Every line here is recorded in [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) or
[`ENVELOPE.md`](ENVELOPE.md) with the entry that established it. Nothing below is
a plan; it is what has actually been exercised.

| what | state |
|---|---|
| Single panel, end to end | **EXERCISED** on real data (2026-08-19) |
| Mixed calibrant, off-panel beam centre | **EXERCISED** (2026-08-19/20) |
| Unattended, archive scale | **EXERCISED** (2026-08-22) — 252 exposures, 57 beamtimes, 2016–2026, no human in the loop; 197 usable, **143 ring-verified**, 28 core-hours. Per-class results in `ENVELOPE.md` |
| Detector classes: frames read vs calibration run | **sorted, then filled in** (2026-08-22) — read `ENVELOPE.md`; "we can read it" is not "we have calibrated on it", and the table now says which is which per class |
| EIGER2 CdTe 16M | **calibrates, does not verify** — 6/6 usable, 0 verified at 1.078 px scatter. Do not quote an EIGER geometry yet |
| Adversarial eval of the halt conditions | **RUN** (2026-08-19); the halt list in §7 survived it |
| Doc set handed to a context-free model | **RUN** (2026-08-19); what it got wrong is recorded in `ENVELOPE.md` and fixed in the spine |
| Multi-panel / tiled beyond what `ENVELOPE.md` lists | **NOT exercised — stop and ask.** Do not improvise a panel layout |
| Recovering λ from the fit residual | **REFUTED — do not retry.** Lab Notebook §15 |

## Conditions on "healthy"

A calibration is healthy **only against a stated condition**, never on its own:

* held-out calibrant strain **< 100 µε** is a **screen, not a verdict** (amended
  2026-09-08). Above it, **stop and look at the ring overlay** — the raw observation —
  then decompose the residual, because its *structure* decides whether it matters. A
  **coherent radial** residual (every ring off the same way, drifting smoothly with 2θ)
  is a scale/Lsd/λ offset: it scales every axis equally and **cancels in any ratio** —
  an a/b splitting, δ, texture. An **azimuthal cos 2η** residual is the one that maps
  onto an apparent a/b splitting. On one DAC calibration "210 µε total" and "a
  0.015 % δ floor" were both true at once. Judge a calibration against the quantity
  you need; a high strain still disqualifies it for absolute d-spacings or strain.
* ring crests, measured off the raw radial profile rather than the fit's own
  metric — the fit's residual can look good while the rings are misplaced.
* the sentinel check has actually run: an unmasked low sentinel produced
  **1369 µε** on a dataset whose geometry was otherwise fine.

One threshold with no condition attached false-alarms on a heavy run and goes
silent on a broken one, which is why each row above names what it is measured
against.

## Where the next session should start

Read `ENVELOPE.md` first and find the detector class in the table. If it is not
there, that is the work — **stop and ask** rather than improvising, because the
failure mode for an unlisted detector is a plausible calibration, not an error.

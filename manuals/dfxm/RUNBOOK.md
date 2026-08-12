# DFXM Runbook — operational state

> Part of the **DFXM doc set**. The spine is [`README.md`](README.md).
>
> **This is the volatile document.** The handbook is procedure and changes slowly; the
> notebook is evidence and only grows. This file describes *right now* — where things run,
> what a healthy number looks like, and where the last session stopped. **Update §R3 before
> you finish.**

---

## R1. Where it runs

`midas_dfxm` is a **Python library** (no CLI). Reduction is not GPU-bound; the dynamical
forward and capability inverses benefit from a GPU but run on CPU.

| | |
|---|---|
| package | `pip install "midas-dfxm>=0.3.2"` (public on PyPI since 2026-07-29) |
| ESRF frame reduction | `pip install darling-pypi` — **imports as `darling`** (moment reduction reference). `pip install darling` fails: no such distribution. |
| Mac env | project env, `midas_env` in use — `source /Users/hsharma/miniconda3/bin/activate midas_env; export KMP_DUPLICATE_LIB_OK=TRUE` |
| APS host env | shared env by full path `/home/beams12/S1IDUSER/opt/envs/midas/bin/python` |
| GPU prefix (dynamical/capability) | `CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE` |
| outputs | a project / gdata directory you own — **never `/tmp`** |

**Before any run, pass the spine §0 import gate and paste its output.** Every number below
is conditional on it.

Scripts that exercise the real-data and cross-model findings live under
`packages/midas_dfxm/dev/paper/runs/`: `make_real_multibragg.py`, `extract_com.py` (ID06
reduction + registration wall), `cross_model_test.py` (the 0.3 Λ boundary), `tt_kin_vs_dyn.py`,
`tt_kinematic_bias.py`, `dynamical_sensitivity.py` (refraction).

---

## R2. What healthy looks like

See [`phase-5-report.md`](phase-5-report.md) §5b for the measured ranges (pedestal share
~98.5 %, `darling` agreement corr 1.0, injection-recovery gain 0.9998–1.0000, per-pixel σ
~2 mdeg, the 0.43 registration-wall NCC, the 0.3 Λ boundary, the 144 µε refraction gauge).
**There is no single "healthy" number** — each row carries the conditions it was measured
under.

---

## R3. Current pick-up point

> **Every session updates this section before it ends.**

**Last updated: 2026-08-12.**

### R3a. Where the last dataset session stopped

> This is what the spine's index means by "current pick-up point": the state of the last
> *reduction*, not of the documentation. A session driving this doc set on 2026-08-12 read
> the index, opened this file expecting operational continuity, and found editorial history
> instead. Keep the two apart.

| | |
|---|---|
| last dataset worked | `darling.assets.mosaicity_scan()` (bundled ESRF ID03) and the archived ID06 `com_111`/`com_002` pair |
| furthest phase reached | Phase 3 — halted at the registration gate, as designed |
| blocked on | **energy/wavelength for the bundled assets** (§1a HALT). Neither carries an energy field; unblocking needs the beamline or proposal record, not more analysis |
| next step | supply that energy, or move to a deposit that carries one, to open Phases 1 and 4 |
| known-good numbers to check against | `f_ped` 0.9849 (median) → 66.3×; registration best NCC 0.43, native −0.30 |

### R3b. Documentation history

**State.** This doc set was created on 2026-08-12 from the P_merged paper campaign — it is the
DFXM sibling of the `ff-hedm` / `nf-hedm` doc sets. It is seeded from **two campaigns, neither
of them our own beamtime**: reduction findings (Notebook §1–§2) are real-data-proven on the
archived ESRF ID03/ID06 sets; the refraction gauge and the 0.3 Λ boundary are verified against
the dynamical forward and cross-model; the capability inverses (typing, defect model, full-F,
dynamical) are simulation-grounded.

**Added the same day — the archive re-analysis lessons (Notebook §7, rules 11–20).** A second
campaign re-analysed an archived public deposit that had been reduced by another group's
pipeline. Its transferable content was folded in here, **anonymised**: the material,
instrument configuration, scientific claim and collaborator are deliberately absent, and only
the artifact, the discriminating test, the mundane cause and the general rule are kept. What
went where:

| file | added |
|---|---|
| `README.md` | **hard rules 11–20**; 5 new halt conditions; 15 new trap rows |
| `LAB_NOTEBOOK.md` | **Notebook §7a–§7i** (9 findings) and **Notebook §5f–§5l** (7 retractions); §1 table extended to 29 rows so every detailed entry has one |
| `DIAGNOSIS.md` | 11 new symptom entries (5 → 16), each with a test that can exonerate |
| `phase-0-survey.md` | 6 survey fields (flux monitor, magnification provenance, detector gain, per-channel sampling, window bracketing, who reduced it) + read-their-pipeline note |
| `phase-1-configure.md` | Λ ∝ 1/\|F\| per-reflection classification; the two t/Λ traps |
| `phase-2-reduce.md` | §2a′ gain by photon transfer; over-subtraction check; §2f per-pixel width |
| `phase-3-multireflection.md` | the geometric rank-6 ceiling and the published oblique-geometry solution |
| `phase-4-analyse.md` | §4d′ shared-channel ratios; §4d″ statistics on autocorrelated maps |
| `phase-5-report.md` | 8 healthy-range rows; n_eff requirement |
| `.claude/skills/dfxm/SKILL.md` | items 6–8 (integrated width, background/gain, controls that can fail) |

**Also fixed:** the doc set previously named the gated collaborator and two sample materials
in Notebook §6 and `RUNBOOK.md` §R3. Those are removed — the gate is kept, the names are not.

**Consistency pass after the injection (same day).** Five defects the injection exposed, all
fixed:

| # | Defect | Fix |
|---|---|---|
| 1 | `SURVEY_TEMPLATE.md` lagged `phase-0-survey.md` by six fields, so the form told you to record less than the procedure required | 6 fields added; new **Reduction health** table (pedestal share, gain, background θ-correlation, kernel-vs-ROI, per-pixel FWHM, pts/FWHM, injection gain); per-reflection Λ rows; 5 new halt checkboxes |
| 2 | `ENVELOPE.md` §3 called a shared-channel intensity ratio **intrinsically** unusable, and blamed a missing calibration | Re-stated as **conditional and testable** (compute \|F\|² per structure *and* variant at the measured Q); a §2 reflection-choice row added, since picking a forbidden-in-the-other channel is the lever |
| 3 | `ENVELOPE.md` said the full tensor needs "≥2 reflections" | Corrected to **≥3 non-coplanar**, in §2, §3 and §4; the rank-6 ceiling for a coplanar set added to §1, with the lever for it in §2 (see the contract note below) |
| 4 | `ENVELOPE.md` §4 attributed achievable χ²/dof to counting statistics alone | Now names the **measured gain** as the first term of the error model, with the ~2.2× consequence |
| 5 | `ENVELOPE.md` and `SURVEY_TEMPLATE.md` were not reachable from the spine | Both added to the README doc-set index; `ENVELOPE.md` added to the closing pointer |

Also: `ENVELOPE.md` §2/§4 now carry the vibration-saturation trade and "recoverability by
shorter exposures is undetermined from images alone"; §5 gained three did-not-versus-cannot
rows (gain unmeasured = did not; flux monitor unlogged and vibration spectrum unrecorded =
genuine cannots). The `mosaicity_too_broad` row in the DIAGNOSIS symptoms table now names
where its check lives, like every other row.

**Checked against the contract.** `ENVELOPE.md` cites `~/opt/beamreport/DOCS_SPEC.md` §6.
That spec lives in a **separate repository** — not under `$MIDAS` — which is why the citation
is not `$MIDAS`-relative like every other one in this doc set; all three ENVELOPE headers now
say so explicitly. (An earlier version of this section wrongly said no such file existed; that
search had been scoped to the MIDAS tree only.) Reading that specification found one real
misfiling in what had just been written: the **θ-rocking rank ceiling was entered in tier 1
(Fixed) with a suggestion attached** ("use the oblique geometry"). The spec is explicit that
tiers 1 and 3 earn a plain statement and **no suggestion at all**, because only tier 2 has an
answerable counterfactual — and a coplanar reflection set is *configured per run*, so that is
where the lever belongs. The ceiling now states the cap and no remedy in §1, and the
reflection row in §2 owns the coplanarity decision and the oblique-geometry counterfactual.
The rest of the contract checks out: every added row names a value, a unit or an explicit
dimensionless, and a provenance; the file is dated and owned; and the three blank §2 bounds
correctly suppress any exposure/dwell counterfactual.

**Sibling doc sets:** `ff-hedm/README.md` and `nf-hedm/README.md` did not index their own
`ENVELOPE.md` either. Both now do, in the same row position and wording as here, so the three
doc sets agree.

**Open, not blocking:**

1. **No real-data validation of any capability inverse.** The single biggest lever is one
   capability (typing, defect model, or full-F) demonstrated on real DFXM frames. Until
   then, §4 results are labelled simulation-grounded.
2. **APS 6-ID-C is gated** — a different instrument and a bilateral collaboration with its own
   credit rules; confirm before touching that data. Its full instrument geometry and its
   vibration/resolution limits are still not in this doc set. The *general* lessons from that
   work are in Notebook §7 and rules 11–20, and the gate stands.
3. **`ENVELOPE.md` §2 still has three blank bounds** — detector maximum frame rate, goniometer
   travel limits, and the dose at which a sample starts to damage. Until filled, a report
   **will not** propose changing exposure or dwell. Its own checklist records this.
4. **Resolution recovery on an archived scan** (within-frame vibration blur vs between-frame
   drift) is pre-registered but never run. Notebook §7f records the blocker: the discriminating
   datum is a vibration **spectrum**, which an image archive cannot supply.
5. **Two threads are closed to re-opening with the same tooling**: the boundary-orientation
   question (Notebook §5h) is unresolved in *either* direction, and sub-resolution doublets
   (Notebook §5l) need ≳ 12 pts/FWHM before revisiting.
6. **A count-aware segmentation classifier** — per-pixel class probability plus an explicit
   "undecidable at this dose" class, replacing a hard intensity-ratio deadband — is
   pre-registered and unbuilt. Notebook §7i is the motivation.

**Nothing is mid-run.** No jobs on any host belonging to this thread.

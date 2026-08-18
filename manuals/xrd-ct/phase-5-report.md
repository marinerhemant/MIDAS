# Phase 5 — report: provenance, and keeping the caution attached

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Goal:** a report where every number can be re-derived, and where the hedges that were true in
the working notes are still true in the text that leaves the session.

---

## 5.1 The rule that matters most

**A caution in the working notes that does not reach the page is not a caution.**

The failure mode is specific and easy: a result is correctly labelled *provisional* while
working, and then the summary, the email or the slide says it plainly. Nothing was falsified;
the label was simply dropped in the last step.

So: **the same status label appears in the notes and in what you send.** If the number is
provisional in `LAB_NOTEBOOK.md` §6, it is provisional in the abstract.

## 5.2 Every number carries its provenance

For each quantitative claim, be able to answer without re-reading the conversation:

```
value           what it is
file            which output it came from
command         what produced that output
```

**Read numbers from the source file, never from conversational memory.** If you cannot point at
where a number came from, do not state it.

## 5.3 The status vocabulary

| Label | Means | Required before using it |
|---|---|---|
| **verified** | survived a fresh adversarial attempt to refute it | `/verify` run and recorded |
| **provisional** | credible, internally consistent, not adversarially tested | say so, every time |
| **bound** | we can say what it is *not* larger than | the softener travels with it |
| **refuted** | a specific claim was tested and failed | keep it recorded so it is not resurrected |
| **not tested** | outside the scope of what was run | distinguish from refuted, always |

**"Not yet verified" means "not done", not "true."**

## 5.4 What a strain report must state

```
Scale:        RELATIVE (median-referenced) or ABSOLUTE
              -> if absolute: how the distance was confirmed, and against what
Per ring:     5-95% inter-percentile range, MAD, live azimuth count out of total
Cross-ring:   agreement of azimuthal patterns and spatial maps
Per voxel:    map + polynomial r^2
Rings used:   which, and which were DROPPED and why (multiplet / low SNR)
Contrast:     peak/background per ring
Label:        provisional unless /verify has been run
```

A worked example, with the labels intact:

> Azimuthal deviatoric strain of **0.3–0.7 %** (3352–7324 µε, 5–95 % inter-percentile;
> MAD 1228–3354 µε) across six independent reflections. **Relative** strain, referenced to the
> median over azimuths — the sample-to-detector distance was not independently confirmed.
> Consistency across six reflections is what makes this credible. **Provisional: this has not
> been through an adversarial verification.**

## 5.5 What a texture report must state

Whether the answer is a map or a bound:

```
Model:           uniaxial 4-parameter | general GSH (n_coef), and WHY that choice
Symmetry:        space group, Laue class, group order, lattice
Identifiability: n_coef, rows, unknowns/rows ratio, ghost_dimension()
Ladder:          chi2 for null / global / per-voxel WITH parameter counts
Checks:          improvement %, polynomial r^2, per-ring agreement
Control:         plant noise, BOTH background modes, detect + resolve verdicts
Scope:           the fibre axis assumed, and whether the loading geometry confirms it
```

**A texture number without its positive control is not reportable** — not because it would be
wrong, but because it would be uninterpretable. A null could equally mean "no texture" or
"cannot see texture", and those are different scientific statements.

If the control says **detect-only** (finds planted texture globally but does not resolve it per
voxel), report a **sample-average bound**, not a map, and say which rung the null came from.

A worked example of a bound, with its softener:

> No coherent azimuthal texture in the vetted phases. A single sample-average fibre buys
> **0.11 %** residual improvement over a uniform null and a per-voxel model **0.17 %**, against
> a pre-registered refute line of 5 % — so the absence is not a spatial-resolution limit.
> Amplitude² scaling against a planted 25 % puts the pole-figure order parameter at
> **|S| ≲ 0.1**.
> **Softener, which travels with the bound:** the positive control carries only synthetic
> Poisson noise on a flat background, while the real data is dominated by systematic per-frame
> background-model error, which does not average down. **The true bound is looser than this
> scaling implies.**
> **Scope:** refuted for a fibre about the rotation axis; **not tested** for any other axis.

## 5.6 Report the negatives

A refuted result is a result. It saves the next person the compute and stops the same hypothesis
being re-proposed.

`LAB_NOTEBOOK.md` §5 carries four refuted or invalid results and three withdrawn inferences.
None died of new physics — they died of a windowed sum that was mostly background, a
degrees-of-freedom mismatch, a plant that never used its random seed, and an inference that was
simply backwards. **Add yours in the same form:** what was claimed, what killed it, and what
generalises.

## 5.7 Figures — keep the generator with the report

Check the figure generator into the report bundle **the same day** the figure is made. A figure
whose generator has drifted away cannot be re-derived, which means its numbers have no
provenance (§5.2).

## 5.8 The checklist before sending

- [ ] Every number traceable to a file and a command.
- [ ] The distance is confirmed against the data, or the strain is labelled **relative**.
- [ ] Dropped rings listed with the reason.
- [ ] Peak-to-background stated per ring.
- [ ] Strain spread as inter-percentile range + MAD, with live-azimuth counts.
- [ ] Every per-voxel map carries its polynomial `r²`.
- [ ] Any texture claim carries its positive-control result **at the measured contrast**.
- [ ] Every softener from the notes is present in the text being sent.
- [ ] Provisional results labelled provisional **in the sent text**.
- [ ] Refuted attempts recorded in `LAB_NOTEBOOK.md`.
- [ ] Figure generators committed alongside the figures.

## 5.9 If a gate halted the work

Say which one, what was measured, and what would unblock it. Then report everything that was
**not** blocked — on this technique, strain is usually not blocked by anything that stops
texture.

Log the halt, since how often the gates fire on real data is the only evidence they are
load-bearing rather than decorative:

```bash
~/.claude/bin/skill-log --skill xrd-ct --event invoked --verdict INVOKED \
  --subject "<which gate halted the work, or 'ran to completion'>" \
  --evidence <the file or reading that triggered it> \
  --note "<what would unblock it>"
```

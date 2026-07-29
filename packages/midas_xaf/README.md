# midas-xaf — Cross-Axis Faceted HEDM (XAF-HEDM)

Design & simulation toolkit for a one-of-a-kind far-field HEDM measurement
performed through a **cubic diamond anvil cell** with a ~15° conical opening on
all six faces.

## The technique

- Narrow ω wedges (~16° wide) are collected at ω ≈ 0, ±90, 180° through the four
  **equatorial** faces in a given mounting.
- The cell is disassembled and **remounted about an orthogonal axis** (90° about
  the beam) so the previously top/bottom faces reach the equator — giving an
  orthogonal rotation axis in the crystal frame that **fills the first
  mounting's missing cone**.
- Two faces are measured in both mountings; embedded high-Z **fiducials** give an
  independent rigid-body remount registration.
- Both mountings are **merged into a single reciprocal-space reconstruction**.

## What the toolkit answers

Built on the differentiable `midas-diffract` forward model, it quantifies —
across energy / distance / detector / opening-angle / beam-mode / material knobs —
reciprocal-space coverage, spots per grain, Friedel-pair completeness, and above
all **strain determinability** (autograd Jacobian conditioning), including how it
degrades with the face opening angle (the 15° vs 20° v2-cell decision).

## Physics captured

- **Exit-aperture shadowing:** the diffracted beam must clear a face opening
  whose transmitting cone rotates with ω, so different detector sectors go dark
  at different ω (`exit_model="cone"`, the default).
- **Orthogonal-axis merge:** single-mounting vs merged strain determinability is
  compared directly (`metrics.cross_axis_gain`).

Status: **alpha** — Phase 1 (forward + metrics + sweep). Phase 2 (registration +
reconstruction) in progress. See `packages/XAF_HEDM_IMPLEMENTATION_PLAN.md`.

"""Command-line entry points for midas-pdf.

Each ``midas-pdf-*`` script is a thin wrapper around the corresponding
library function. Design principles:

  * Read plain ASCII files: two-column ``r G(r)`` or three-column
    ``r G(r) sigma`` for PDF; same for I(Q).
  * Write structured output to stdout (JSON summary) and optional
    output files.
  * Minimal argparse surface — one main flag per input dataset, one
    per output artefact.

Registered scripts (see ``pyproject.toml``):

  midas-pdf-cif      — CIF inspection + Crystal ↔ CIF round-trip
  midas-pdf-refine   — small-box PDF refinement against a ``.gr``
  midas-pdf-joint    — joint SAXS+PDF (add ``--sans`` for three-way)
"""

__all__ = ["cif_cmd", "refine_cmd", "joint_cmd"]

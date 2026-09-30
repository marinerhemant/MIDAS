# Phase 6 -- report

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

Every number names its file (`report.json` lists them). State, per series:

1. Geometry source and its status (verified / provisional, and why).
2. Windows and the rule that chose them; dropped series and why.
3. Matrix-scale and halo traces (relative; caveats as in phase 5).
4. Detection limit from the injection curve: "no candidate-pattern spot above X counts per frame
   was missed at rate > 20 %", and the fraction of the pattern hidden in matrix bands.
5. Candidate test: every candidate with `n_lines`, `chance_coverage`, M, L, null mean, p, edge flag;
   the negative controls and whether they failed.
6. Verdict words: pass (provisional until independently checked), no pass within the stated
   sensitivity, or unreadable (a control passed). Never "phase X is absent"; never a composition
   from a cell.
7. Leads (single features that coincide with a line) listed as leads.

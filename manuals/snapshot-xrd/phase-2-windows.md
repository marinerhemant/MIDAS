# Phase 2 -- time windows, fixed before spots are opened

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

Windows decide which frames count as "before" and "after" an event. Choosing them after looking
at spots is a forking path; choose them from traces only.

1. Run the per-window pass (`midas-snapshot run`) and open only `windows_W{W}.csv` / the trace
   figure: matrix scale, halo index, spot count.
2. Apply the stated rule (`windows_from_trace`): baseline = median halo over the first
   `baseline_until` frames; onset = first window above baseline + `rise`; before = [0, onset -
   `guard`]; after = the last `after_len` frames. The rule is written into `analysis.json`.
3. Exclude the event itself (melt, solidification, the shot): inside it the matrix spans a range of
   states within one frame, and off-matrix spots there are mostly the matrix itself.
4. Drop series whose before window is too short or whose halo never returns to baseline; report
   them as dropped.
5. Write the windows into the preregistration before running the analysis.

Static series (no event) and position maps have no windows: use `--static` or `--map` (see the
runbook). For short series the window-rule frame counts should scale with the series length.

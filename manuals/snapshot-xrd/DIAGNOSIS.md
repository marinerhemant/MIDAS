# Snapshot diffraction -- diagnosis

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

Each row is a failure that produced a wrong result on real still-frame data before the rule
existed.

| Symptom | Cause | Rule / fix |
|---|---|---|
| Calibrant "matches" in an overlay plot, rings smeared in a profile | overlay contours hide the data; wrong raw orientation | check with a radial profile in each candidate orientation; set `flip` |
| Predicted rings "do not overlap the data" | a lattice parameter was assumed and drawn | fit the matrix scale; never draw a guessed cell |
| Halo profile is flat at 0 or 1 | median of a few-count ring rails at integer values | use the mean (trimmed for halos) |
| A large "off-ring" population appears | the matrix ring list was truncated below the detector edge | use every matrix line to the detector edge |
| Off-ring spots sit just below matrix rings near a melt | the matrix spans a temperature range inside one frame | exclude the melt/solidification windows; classify per spot |
| Summing frames does not lower the detection limit | an absolute count floor scales with the number of summed frames | no count floor; threshold from null images |
| Single photons in a detector corner become very significant spots | background taken as a per-ring mean while the true background varies with azimuth | local background |
| Negative-control cells "pass" | repeated detections of the same feature counted as independent | merge detections into features first |
| Strongest candidate lines never tested | candidate lines limited to the range where spots were found | lines over the full detector range |
| Positive control shows 100 % power, meaning nothing | synthetic d-values injected into a list, bypassing detection and classification | inject into images, run the identical pipeline |
| A candidate passes with its best scale on the window edge | the true cell lies outside that candidate's window | report the edge; scan the cell (with a look-elsewhere null) |
| The top "feature" persists after the event it should vanish with | detector column or module-edge artefact | mask margin; raw-photon before/after test; fixed-pixel check against unrelated data |
| Ring-centroid scale biased by ~1 % on a coarse-pixel geometry | centroid window only 1-2 pixels wide, background taken on ring flanks | window >= 6 pixels in 2-theta (automatic) |
| Halo index is high in solid frames | halo band overlaps a ring | choose bands clear of every ring |
| A matched "feature" has only a few detections and no raw-photon contrast | weak features pass a detection-based absence check | require the raw-photon test and a minimum number of detections |
| Features dominated by one scan or one position | minor phases are sparse and local | report per scan; test leave-one-scan-out |
| A phase is "found" in a block that holds only liquid | a ring window on a broad halo top rises above a straight line through the window edges | robust quadratic background from the flanks, then a width test: a halo is not a ring (automatic) |
| Three rings "agree" but one is off by more than 1 % | a MAD spread with three rings ignores one ring entirely | drop rings off the others by more than max(0.4 %, one pixel in d) before counting (automatic) |
| A real phase disappears; a far-off cell of the same structure type is reported | "same structure, best in different blocks" merged two patterns 14 % apart | merge only if the fitted cells agree within 3 % (automatic) |
| A second cubic phase exactly on the main phase's lines is kept | sub-pattern test done at each phase's own block, at different temperatures | sub-pattern test on accepted rings with a free relative scale (automatic) |
| A phase's cell differs from what its strongest block shows | cell taken from the first block that selected the phase | cell from the most complete block; every block's cell listed (automatic) |
| An extra phase is labelled with a reference that does not describe it | the reference list is finite; the nearest reference catches some of its lines | read `unexplained` in `setup_select.json`: strong lines on no known phase mean the phase set is incomplete |
| A 3-ring bcc replaces an 8-ring fcc in a hot block | the fcc's rings are split by a temperature gradient and narrowly fail the spread gate; bcc 110/220/400 lie exactly on fcc 111/222/422 | a fit whose rings all lie on a more complete, nearly consistent pattern of the same block is vetoed (automatic) |

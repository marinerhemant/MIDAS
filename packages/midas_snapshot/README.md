# midas-snapshot

Still-frame diffraction time series, where the sample does not rotate: monochromatic beam, an
area detector, one frame or a series. The patterns are spotty or sparse, so radial lineouts
lose the signal.

For each window of frames, the pipeline:

1. fits one isotropic scale of a known matrix phase from its rings (a relative thermometer);
2. computes a diffuse-halo index (melt / amorphous indicator);
3. detects spots with a background model that is local and a threshold calibrated on
   noise-only images;
4. places each spot relative to the matrix lines.

Downstream stages:

- choose before/after windows from the traces by a stated rule;
- merge repeated detections into features, and test "present before / absent after" on raw
  photon counts;
- test candidate cells with a null and automatically generated negative controls;
- measure the detection limit by injecting synthetic spots into real frames.

```
midas-snapshot init --frames DIR --geometry params.txt --out OUT --matrix-cif matrix.cif \
                    --halo-band 2.9 3.2 --base-band 1.9 2.2 --tth-max 12
midas-snapshot run OUT/snapshot_config.json
midas-snapshot analyse OUT/snapshot_config.json --candidates cand1.cif cand2.cif --controls
midas-snapshot report OUT/snapshot_config.json
```

What the method cannot do:

- It cannot give an absolute temperature: thermal and uniform elastic dilatation shift peaks
  identically.
- It cannot give a strain tensor from one still.
- It cannot identify a phase from one line.
- A passing candidate identifies a cell and pattern, not a chemistry.

The procedure and its measured limits are in `manuals/snapshot-xrd/`.

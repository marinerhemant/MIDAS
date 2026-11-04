| scenario | condition | expected | fired | result | note |
|---|---|---|---|---|---|
| C | no blocking/limiting (quality, scan_range, resolution) | none | resolution_signal_limited:limiting | **FAIL** | coverage ids (completeness, ...) reported not judged: completeness:note, next_setting:note |
| Cw | refined wedge on clean frames < 0.1 deg | < 0.1 | {'Cw_3': 0.006, 'Cw_4': 0.005, 'Cw_5': 0.012} | **PASS** |  |
| Cw | geometry_consistency not limiting | note or none | none | **PASS** |  |
| C6 | no blocking/limiting finding from ANY check | none | resolution_signal_limited:limiting | **FAIL** |  |
| C6hi | no blocking/limiting finding from ANY check (6 runs, 100 x counts) | none | none | **PASS** |  |
| C6_dmin | NEGATIVE control (low counts, target d_min 0.66): coverage/resolution still limiting | limiting | resolution_signal_limited | **PASS** | the low-count scene is signal-limited |
| C6hi_dmin | no blocking/limiting finding from ANY check (6 runs, 100 x counts, target d_min 0.66) | none | none | **PASS** |  |
| D1 | intensity_drift limiting | limiting | limiting | **PASS** |  |
| D1 | placebo gain < half the real gain | < 0.5 x gain | placebo 0.035 real 0.850 | **PASS** |  |
| D1 | no other limiting finding | none | resolution_signal_limited:limiting | **FAIL** |  |
| D2 | dynamic_range present, fraction above 25 % of the cutoff >= 0.30 | >= 0.30 | note, fraction 0.430 | **PASS** |  |
| D2 | no other limiting finding | none | resolution_signal_limited:limiting | **FAIL** |  |
| D3 | omega_sampling present with median width < 3 frames | < 3 | note, width 0.00 | **PASS** |  |
| D3 | no other limiting finding | none | resolution_signal_limited:limiting | **FAIL** |  |
| D4_unset | scan_range limiting, range 40 +- 1 deg | limiting | limiting, ranges [40.0, 40.0, 40.0] | **PASS** |  |
| D4_unset | no unexpected limiting finding | none | none | **PASS** |  |
| D4_set | scan_range note, range 40 +- 1 deg | note | limiting, ranges [40.0, 40.0, 40.0] | **FAIL** |  |
| D4_set | no unexpected limiting finding | none | none | **PASS** |  |
| D4_set40 | scan_range note, range 40 +- 1 deg | note | note, ranges [40.0, 40.0, 40.0] | **PASS** |  |
| D4_set40 | no unexpected limiting finding | none | none | **PASS** |  |
| D5 | completeness or redundancy limiting | limiting | completeness, redundancy | **PASS** |  |
| D5 | no unexpected limiting finding | none | resolution_signal_limited:limiting | **FAIL** |  |
| D6 | geometry_consistency present | note or limiting | note | **PASS** |  |
| D6 | refined wedge in 0.3..0.5 deg for every run | 0.3-0.5 | {'D6_3': 0.406, 'D6_4': 0.41, 'D6_5': 0.378} | **PASS** |  |
| D6 | no other limiting finding | none | resolution_signal_limited:limiting | **FAIL** |  |
| D7 | contamination names a ring within 0.3 deg of 2theta = 12.0 | ring at 12.0 | rings [12.12] | **PASS** |  |
| D7 | no other limiting finding | none | resolution_signal_limited:limiting | **FAIL** |  |
| D8 | systematic_floor limiting | limiting | note | **FAIL** |  |
| D8 | intensity_drift NOT limiting | not limiting | absent | **PASS** |  |
| D8 | no other limiting finding | none | resolution_signal_limited:limiting | **FAIL** |  |
| D8b | systematic_floor limiting (per-frame sigma 0.25) | limiting | limiting, floor 0.150 | **PASS** |  |
| D8b | intensity_drift NOT limiting | not limiting | note | **PASS** |  |
| D8b | no other limiting finding | none | resolution_signal_limited:limiting | **FAIL** |  |

#!/bin/bash
cd "$(dirname "$0")"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
until grep -q "front end finished" results/_chain_twin2.log 2>/dev/null; do sleep 60; done
for v in tw30 tw10 tw00; do
  find results -maxdepth 1 -name "twin2_${v}_*.csv" -delete
  $PY run_twin2_refine.py $v 0 50 > results/_twin2_$v.log 2>&1 &
done
wait
$PY analyse_twin2.py > results/_twin2_analysis_stdout.log 2>&1
echo twin2 chain finished

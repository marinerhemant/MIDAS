#!/bin/bash
# waits for the three front-end runs, then refines every variant and writes results/ext_analysis.txt
cd "$(dirname "$0")"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
while pgrep -f "run_c2.py S2 0 50 v3" > /dev/null; do sleep 60; done
for v in T1_5 T1_10 T2; do
  find results -name "ext_${v}_*.csv" -delete
  $PY run_ext_refine.py $v 0 50 > results/_ext_$v.log 2>&1
done
$PY analyse_ext.py > results/_ext_analysis_stdout.log 2>&1
echo chain finished

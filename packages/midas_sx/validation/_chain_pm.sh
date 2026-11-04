#!/bin/bash
# A26: wait for the domain-finding chain and the P30 cache, then P11, P30, P00 (two processes each), then analysis
cd "$(dirname "$0")"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
C=/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache
until grep -q "fd chain finished" results/_chain_fd.log 2>/dev/null; do sleep 60; done
until [ -f $C/S2P_immm_b1.03_dom2.npz.exp.json ]; do sleep 30; done
mkdir -p results/pm_merged
for v in P11 P30 P00; do
  find results -maxdepth 1 -name "pm_${v}_[0-9]*.csv" -delete
  $PY run_pm.py $v 0 25 > results/_pm_${v}_a.log 2>&1 &
  $PY run_pm.py $v 25 50 > results/_pm_${v}_b.log 2>&1 &
  wait
done
$PY analyse_pm.py > results/_pm_analysis_stdout.log 2>&1
echo pm chain finished

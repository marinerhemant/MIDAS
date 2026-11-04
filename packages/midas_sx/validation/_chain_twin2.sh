#!/bin/bash
# A24: front end (v3: exact U, exact geometry) for the three twin variants once their caches exist, two chunks per variant at most
cd "$(dirname "$0")"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
C=/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache
for t in 30 10 00; do
  until [ -f $C/S2O_immm_n100000_o1_tw$t.npz.exp.json ]; do sleep 20; done
  while [ $(pgrep -f "run_c2.py S2O 0" | wc -l) -ge 2 ]; do sleep 30; done
  TWIN=$t nohup $PY run_c2.py S2O 0 25 v3 > results/_fe_tw${t}_a.log 2>&1 &
  TWIN=$t nohup $PY run_c2.py S2O 25 50 v3 > results/_fe_tw${t}_b.log 2>&1 &
  sleep 5
done
while pgrep -f "run_c2.py S2O" > /dev/null; do sleep 60; done
echo front end finished

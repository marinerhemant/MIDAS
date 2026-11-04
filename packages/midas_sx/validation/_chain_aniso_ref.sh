#!/bin/bash
cd "$(dirname "$0")"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
until grep -q "front end finished" results/_chain_aniso_fe.log 2>/dev/null; do sleep 60; done
while pgrep -f "run_aniso_refine.py T0" > /dev/null; do sleep 20; done
find results -name "aniso_S2A_*.csv" -delete
$PY run_aniso_refine.py S2A 0 17 > results/_aniso_S2A_a.log 2>&1 &
$PY run_aniso_refine.py S2A 17 34 > results/_aniso_S2A_b.log 2>&1 &
$PY run_aniso_refine.py S2A 34 50 > results/_aniso_S2A_c.log 2>&1 &
wait
$PY analyse_aniso.py > results/_aniso_analysis_stdout.log 2>&1
echo aniso chain finished

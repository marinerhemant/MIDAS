#!/bin/bash
# S2A: build the signal cache, then the v3 front end in three parallel chunks (PREREGISTER A22)
cd "$(dirname "$0")"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
$PY run_c2.py S2A make-cache > results/_mk_S2A.log 2>&1
$PY run_c2.py S2A 0 17 v3 > results/_fe_S2A_a.log 2>&1 &
$PY run_c2.py S2A 17 34 v3 > results/_fe_S2A_b.log 2>&1 &
$PY run_c2.py S2A 34 50 v3 > results/_fe_S2A_c.log 2>&1 &
wait
echo front end finished

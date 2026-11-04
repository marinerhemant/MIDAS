#!/bin/bash
cd "$(dirname "$0")"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
find results -maxdepth 1 -name "fd_[0-9]*.*" -delete; find results -maxdepth 1 -name "fd_single_*" -delete
$PY run_finddomains.py 0 25 > results/_fd_a.log 2>&1 &
$PY run_finddomains.py 25 50 > results/_fd_b.log 2>&1 &
wait
SINGLE=1 $PY run_finddomains.py 0 13 > results/_fd_s1.log 2>&1 &
SINGLE=1 $PY run_finddomains.py 13 25 > results/_fd_s2.log 2>&1 &
wait
echo fd chain finished

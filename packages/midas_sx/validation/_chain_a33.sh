#!/bin/zsh
# PREREGISTER A33: waits for the A30 cache builds to finish (RAM), then cache -> tables -> arms
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
until [ -f results/_a30_cache_done.txt ]; do sleep 30; done
JOINT=30 $PY run_c2.py S2OA make-cache >> results/_a33.log 2>&1
JOINT=30 $PY run_c2.py S2OA 0 30 v3 >> results/_a33.log 2>&1
$PY run_joint_refine.py 0 30 >> results/_a33.log 2>&1
echo done > results/_a33_done.txt

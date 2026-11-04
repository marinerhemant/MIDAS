#!/bin/zsh
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
for v in P11 P30 P00; do $PY diag_bg_oracle.py $v 50 55 >> results/_a34_$v.log 2>&1; done
echo done > results/_a34_done.txt

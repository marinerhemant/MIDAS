#!/bin/zsh
# PREREGISTER A29: sequential, one process at a time (Mac RAM limit)
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
for v in P00 P11 P30; do
  $PY diag_blend_offset.py $v 50 60 >> results/_a29_$v.log 2>&1
done
echo done >> results/_a29_done.txt

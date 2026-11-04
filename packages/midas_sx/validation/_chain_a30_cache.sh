#!/bin/zsh
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
for o in 4 6; do
  ORI=$o $PY run_c2.py S2A make-cache >> results/_a30_cache.log 2>&1
  ORI=$o TWIN=30 $PY run_c2.py S2O make-cache >> results/_a30_cache.log 2>&1
done
echo done > results/_a30_cache_done.txt

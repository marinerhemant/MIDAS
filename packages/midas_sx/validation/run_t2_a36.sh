#!/bin/zsh
# PREREGISTER A36 T2: held-out seeds 70-89; the A27 chain (run_pm2.py + refit_pm.py arm Tr) with background mean (baseline, same seeds) and quadratic. usage: run_t2_a36.sh <mean|quadratic> <P11|P30|P00>
BG=$1; VAR=$2
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation" || exit 1
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
TAB=pm2_merged; [ "$BG" = quadratic ] && TAB=pm2q_merged
BG=$BG $PY run_pm2.py $VAR 70 90 > results/_t2_${BG}_${VAR}.log 2>&1
$PY refit_pm.py $VAR 70 90 $TAB Tr >> results/_t2_${BG}_${VAR}.log 2>&1

#!/bin/bash
# A27: structure arms on the A26 tables (A1) and on the pair-model tables (B0, B1). Everything runs unattended; results/pmr_*.csv, results/pm2_*.csv.
cd "$(dirname "$0")"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
mkdir -p results/pm2_merged
for v in P11 P30 P00; do find results -maxdepth 1 -name "pm2_${v}_[0-9]*.csv" -delete; find results -maxdepth 1 -name "pmr_*_${v}_[0-9]*.csv" -delete; done
job_a1() { v=$1; lo=$2; hi=$3; $PY refit_pm.py $v $lo $hi pm_merged Ur,Tr > results/_a1_${v}_$lo.log 2>&1; }
job_b() { v=$1; lo=$2; hi=$3; $PY refit_pm.py $v $lo $hi pm2_merged T,Ur,Tr > results/_b_${v}_$lo.log 2>&1; }
# worker 1: tables with the pair model for P30, then their refits; worker 2: P11
( $PY run_pm2.py P30 0 50 > results/_pm2_P30.log 2>&1; job_b P30 0 25 & job_b P30 25 50 & wait ) &
( $PY run_pm2.py P11 0 50 > results/_pm2_P11.log 2>&1; job_b P11 0 25 & job_b P11 25 50 & wait ) &
# worker 3: robust refits of the existing A26 tables (A1), two processes per variant
( job_a1 P30 0 25 & job_a1 P30 25 50 & wait; job_a1 P11 0 25 & job_a1 P11 25 50 & wait; job_a1 P00 0 25 & job_a1 P00 25 50 & wait ) &
wait
# P00 pair-model tables last (the null), then its refits
$PY run_pm2.py P00 0 50 > results/_pm2_P00.log 2>&1
job_b P00 0 25 & job_b P00 25 50 & wait
echo a27 chain finished

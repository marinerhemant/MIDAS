#!/bin/zsh
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
until [ $(ls results/a35_obs 2>/dev/null | wc -l) -ge 10 ] && ! pgrep -f "diag_bg_oracle.py P11 60" > /dev/null; do sleep 30; done
$PY refit_a35.py 60 70 > results/_a35_fit.log 2>&1
$PY analyse_a35.py > results/a35_analysis.txt 2>&1
cat results/a35_analysis.txt

#!/bin/zsh
# PREREGISTER A36 T3: untwinned S2, orientation 1, v3 chain, noise seeds 0-24 (frames seed 1000+seed), background mean vs quadratic on the SAME frames. Waits for the T2a runs to finish.
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation" || exit 1
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
while pgrep -f run_t2a_a36 > /dev/null; do sleep 20; done
TAG=_t3 BG=mean $PY run_c2.py S2 0 25 v3 > results/_t3_mean.log 2>&1 &
TAG=_t3 BG=quadratic $PY run_c2.py S2 0 25 v3 > results/_t3_quad.log 2>&1 &
wait

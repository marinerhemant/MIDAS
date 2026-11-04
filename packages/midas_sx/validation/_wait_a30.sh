#!/bin/zsh
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation"
until [ -f results/_a30_aniso_done.txt ] && [ -f results/_a30_twin_done.txt ]; do sleep 30; done
/Users/travel_user/miniforge3/envs/midas_env/bin/python analyse_a30.py > results/a30_analysis.txt 2>&1
cat results/a30_analysis.txt

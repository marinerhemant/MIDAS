#!/bin/zsh
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation"
until [ -f results/_a33_done.txt ]; do sleep 30; done
/Users/travel_user/miniforge3/envs/midas_env/bin/python analyse_joint.py > results/joint_analysis.txt 2>&1
cat results/joint_analysis.txt

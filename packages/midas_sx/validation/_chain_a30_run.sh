#!/bin/zsh
# PREREGISTER A30: one stream per structure; orientation 4 then 6 (waits for the cache of each)
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation"
PY=/Users/travel_user/miniforge3/envs/midas_env/bin/python
CACHE=/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache
stream=$1
for o in 4 6; do
  if [ "$stream" = "aniso" ]; then
    until [ -f $CACHE/S2A_bilayer_like_aniso_n100000_o$o.npz.truth.pkl ]; do sleep 20; done
    ORI=$o $PY run_c2.py S2A 0 25 v3 >> results/_a30_aniso_o$o.log 2>&1
    ORI=$o $PY run_aniso_refine.py S2A 0 25 >> results/_a30_aniso_o$o.log 2>&1
  else
    until [ -f $CACHE/S2O_immm_n100000_o${o}_tw30.npz.truth.pkl ]; do sleep 20; done
    ORI=$o TWIN=30 $PY run_c2.py S2O 0 25 v3 >> results/_a30_twin_o$o.log 2>&1
    ORI=$o $PY run_twin2_refine.py tw30 0 25 >> results/_a30_twin_o$o.log 2>&1
  fi
done
echo done > results/_a30_${stream}_done.txt

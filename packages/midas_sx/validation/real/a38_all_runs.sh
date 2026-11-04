#!/bin/zsh
# A38: the omega-scan runs (01 02 03 05 06); run 04 is a PHI scan (rotation about a tilted axis): not supported by the package, left out and said so.
cd "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx/validation/real" || exit 1
D="/Volumes/T5 EVO/open_data/thiourea_zenodo_18910433/x/2_T130K_18MPC001"; O="/Volumes/T5 EVO/open_data/thiourea_zenodo_18910433/out"; CIF="$D/work_FixUisoH/18MPC001T130K.cif"
RUNS=${RUNS:-"02 03 05 06"}
for r in ${=RUNS}; do
  /Users/travel_user/miniforge3/envs/midas_env/bin/python -I a38_run.py "$D" mo_18MPC001_rampT160K_${r}_ "$D/mo_18MPC001_rampT160K_xa_${r}_0001.sfrm" "$CIF" "$O/r$r" "$@" > "$O/r$r.log" 2>&1
done
echo done > "$O/all_runs.done"

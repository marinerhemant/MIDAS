#!/bin/bash
# Far-field chain for ONE layer of the TARGET phase, with the checks that the reference beamtime found necessary.
#   export EXP=<experiment>; source ../kit_env.sh; ./target_ff.sh <stage> <scan_folder> <file_nr>
# stages (run them in order, read each one's output before the next):
#   prep     background pseudo-dark, parameter file, TARGET ring selection (rings that do not sit on a PRIMARY line), ring-thresh,
#            then zip + hkl + peakfit + transforms. Peakfit is the long step (2.7 h for one 1442-frame layer on 2 A6000, reference beamtime).
#   seed     choose the seed ring (OverAllRingToIndex) and write the seed list. Reports |F|^2 of each ring when TARGET_CIF is set.
#   index    binning (CPU) + indexing + refinement + process_grains -> $RUN/real/results/LayerNr_1/Grains.csv
#   nulls    3 omega-scrambled null arms from the same spots (run_arm.py), same seed ring, same chain
#   readout  grains above the null (ff_readout.py): the only way to know whether the grains are more than chance
# DRY=1 prints the long commands instead of running them. The FIRST stages still run for real: a DRY prep writes a ~12 GB zip under $RUN/ungated and
# takes ~11 minutes (midas-ring-thresh is single-threaded, ~9 GB RAM). Nothing is fitted, so InputAll.csv does not exist afterwards. Override the seed ring with SEED_RING=<n>, the completeness gate with GATE=<x>.
# Every number below that the study fixed is a default, not a law: seed budget 50..20000, ring window 200..1100 px, overlap 7 px.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../kit_env.sh" || exit 1
USAGE="usage: target_ff.sh <prep|seed|index|nulls|readout> <scan_folder> <file_nr>   e.g. target_ff.sh prep S2_70wt_8kg_line_HEDM_cathode_redo 575"
STAGE=${1:-}; FOLDER=${2:-}; NR=${3:-}
[ -n "$STAGE" ] && [ -n "$FOLDER" ] && [ -n "$NR" ] || { echo "$USAGE"; exit 1; }
kit_check || exit 1
RUN=$WORK/ff/${FOLDER}_$NR; LAY=$RUN/real/results/LayerNr_1; GATE=${GATE:-0.5}
DATA=$RAW/varexD/$FOLDER; STEM=$FOLDER
FILE=$DATA/${STEM}_$(printf %06d $NR).vrx.h5
set_gate() {    # set_gate <arm dir>: the completeness gate lives in Parameters.txt (Completeness) AND in the layer's paramstest.txt (MinMatchesToAcceptFrac); prep wrote both with the prep-time GATE
  local d=$1
  sed -i.bak "s/^Completeness .*/Completeness $GATE/" $d/Parameters.txt
  sed -i.bak "s/^MinMatchesToAcceptFrac .*/MinMatchesToAcceptFrac $GATE;/" $d/results/LayerNr_1/paramstest.txt
  # a stage is skipped when its output exists: a re-run at another gate must clear the products of indexing onward and the derived paramstest copies
  rm -rf $d/results/LayerNr_1/{Output,Results,Grains.csv,SpotMatrix.csv,GrainIDsKey.csv,paramstest_index_comp.txt,paramstest_refine_comp.txt,paramstest_pg.txt,paramstest_pg_comp.txt}
  echo "completeness gate set to $GATE in $d"
}
run() { if [ "${DRY:-0}" = 1 ]; then echo "[dry] (run from $PWD) $*"; else eval "$@"; fi; }
mkdir -p "$RUN"; cd "$RUN"

params() {       # params <outfile> <ring-thresh> <overall-ring>
  local extra=""
  [ -f "$(dirname "$CAL_PARAMS")/residual_corr.bin" ] && extra="ResidualCorrectionMap=$(dirname "$CAL_PARAMS")/residual_corr.bin"
  [ -n "${TARGET_CIF:-}" ] && extra="$extra PhaseCIF=$TARGET_CIF"
  $PY -B -I "$KIT/ff/make_params.py" "$CAL_PARAMS" "$1" SpaceGroup=$TARGET_SG "LatticeConstant=$TARGET_LAT" \
    FileStem=$STEM StartFileNrFirstLayer=$NR RawFolder=$DATA/ MarginOme=0.5 Completeness=$GATE Wedge=0 \
    Dark=$RUN/bgdark.h5 darkLoc=/exchange/dark RingThresh="$2" OverAllRingToIndex=$3 $extra
}

case $STAGE in
prep)
  say "pseudo-dark (per-pixel temporal median of 120 frames): without it no zipper threshold works on dense rings"
  [ -f bgdark.h5 ] || $PY -B -I "$KIT/ff/make_bg_dark.py" "$FILE" bgdark.h5 120 12 || exit 1
  say "hkl pass (all rings), to read ring radii"
  mkdir -p ungated/results; ( cd ungated; params Parameters.txt 1:300 1
    $P/midas-pipeline run --scan-mode ff --params Parameters.txt --result results --layers 1-1 --only zip_convert --only hkl --n-cpus $NCPUS > step.log 2>&1 ) || { echo ZIP_FAILED; exit 1; }
  say "ring selection: TARGET rings 200..1100 px that are not within 7 px of a PRIMARY line"
  export RUN KIT_ENERGY=$ENERGY_KEV KIT_LSD=${LSD_MM} KIT_PX=$PX_UM KIT_PSG=$PRIMARY_SG KIT_PLAT="$PRIMARY_LAT"
  RINGS=$($PY -B -I - <<'PYE'
import os, numpy as np, midas_hkls as mh
px = float(os.environ["KIT_PX"]); lam = 12.398419843 / float(os.environ["KIT_ENERGY"]); lsd = float(os.environ["KIT_LSD"])
rows = [l.split() for l in open(os.environ["RUN"] + "/ungated/results/LayerNr_1/hkls.csv").read().splitlines() if l.strip()]
h = rows[0]; ir, irad = h.index("RingNr"), h.index("Radius"); R = {}
for r in rows[1:]: R.setdefault(int(r[ir]), float(r[irad]) / px)
lat = [float(x) for x in os.environ["KIT_PLAT"].split()]
rf = mh.generate_hkls(mh.SpaceGroup.from_number(int(os.environ["KIT_PSG"])), mh.Lattice(*lat), d_min=1.0, d_max=12.0)
dP = np.array(sorted({x.d_spacing for x in rf})); RP = lsd * 1000 / px * np.tan(2 * np.arcsin(lam / (2 * dP)))
print(",".join(str(k) for k, v in sorted(R.items()) if 200 < v < 1100 and np.abs(RP - v).min() >= 7))
PYE
  )
  [ -n "$RINGS" ] || { echo "NO_TARGET_RINGS: every target ring sits on a primary line or outside 200..1100 px"; exit 1; }
  echo "TARGET rings selected: $RINGS"; echo "$RINGS" > rings.txt; OVR=${RINGS%%,*}
  RT0=$(echo $RINGS | tr ',' '\n' | awk '{printf "%s%s:300", (n++?",":""), $1}')
  # the zip embeds the parameters it was built with: rebuild it with the FULL ring list BEFORE ring-thresh (else the tool sees one ring)
  ( cd ungated; params Parameters.txt "$RT0" $OVR; rm -rf results; mkdir results
    $P/midas-pipeline run --scan-mode ff --params Parameters.txt --result results --layers 1-1 --only zip_convert --only hkl --n-cpus $NCPUS > step2.log 2>&1 ) || { echo ZIP2_FAILED; exit 1; }
  say "ring-thresh (verbatim output is the RingThresh)"
  $P/midas-ring-thresh ungated/results/LayerNr_1/${STEM}_$(printf %06d $NR).MIDAS.zip --result-folder ungated/results/LayerNr_1 --n-frames 40 2>&1 | grep -v Warning > ringthresh.txt
  RT=$(sed -n '/Paste into/,$p' ringthresh.txt | grep '^RingThresh' | sed 's/#.*//' | awk '{printf "%s%s:%s", (n++?",":""), $2, $3}')
  [ "$(echo "$RT" | tr ',' '\n' | wc -l)" -ge 3 ] || { echo "RINGTHRESH_TOO_FEW: $RT"; exit 1; }
  THR=$(echo $RT | tr ',' '\n' | awk -F: 'NR==1{m=$2} $2<m{m=$2} END{print m}'); echo "RingThresh $RT ; preproc floor = $THR"; echo $THR > preproc_thr.txt
  echo "$RT" > RingThresh.txt; echo "RingThresh (verbatim from midas-ring-thresh, also in ringthresh.txt under 'Paste into the parameter file'): $RT"
  # OverAllRingToIndex $OVR below is a PLACEHOLDER (the first target ring); the seed stage chooses the real seed ring
  mkdir -p real/results; ( cd real; params Parameters.txt "$RT" $OVR )
  say "peakfit + transforms (LONG: hours). Run it with nohup on the host, never inline over a fragile ssh"
  ( cd real; run "$P/midas-pipeline run --scan-mode ff --params Parameters.txt --result results --layers 1-1 --preproc-thresh $THR --n-cpus $NCPUS --only zip_convert --only hkl --only peakfit --only transforms --only cross_det_merge --only global_powder > prep.log 2>&1; echo prep exit=\$?" )
  [ "${DRY:-0}" = 1 ] && echo "DRY RUN: nothing was fitted. To fit: rerun without DRY=1 under nohup (peakfit ~2.7 h on 2 GPUs); until then $LAY/InputAll.csv does not exist and the seed stage cannot run."
  ;;
seed)
  [ -f "$LAY/InputAll.csv" ] || { echo "no $LAY/InputAll.csv: the prep stage has not run to the end (a DRY=1 prep never makes it). Run: nohup ./ff/target_ff.sh prep $FOLDER $NR > $WORK/prep.out 2>&1 &   then check $RUN/real/prep.log"; exit 1; }
  export LAYER=$LAY SEED_RING=${SEED_RING:-}
  $PY -B -I - <<'PYE'
import os, sys, numpy as np, pandas as pd
L = os.environ["LAYER"]
try:
    from midas_pipeline.diagnostics.seed_ring import ring_seed_table, advise_seed_ring
    tab = ring_seed_table(L + "/hkls.csv", L + "/InputAll.csv"); adv = advise_seed_ring(tab, None)
    print("ring  F2rel%  spots  est.seeds"); [print(f"{r['ring']:4d} {'' if r['f2_rel'] is None else '%6.1f' % (100*r['f2_rel'])} {r['n_spots']:7d} {r['n_seeds_est']:8d}") for r in tab]
    rec = adv["recommended"]; print("recommended seed ring:", rec, "|", adv["message"] or "(needs F2: set TARGET_CIF so hkls.csv carries it)")
except ImportError:
    print("midas_pipeline.diagnostics.seed_ring not in this env (needs the wave-3 release): choose the ring by hand, see README 'seed ring'")
    rec = None
ring = int(os.environ["SEED_RING"]) if os.environ["SEED_RING"] else rec
if ring is None: sys.exit("set SEED_RING=<n> (a ring with a moderate |F|^2 and 50..20000 seeds); the default seed ring is the FIRST selected ring and may be a weak reflection")
ia = pd.read_csv(L + "/InputAll.csv", sep=r"\s+", usecols=["SpotID", "RingNumber", "Omega"])
ids = ia[(ia.RingNumber == ring) & (ia.Omega >= -90) & (ia.Omega <= 90)].SpotID.astype(int).values
p = L + "/SpotsToIndex.csv"
if os.path.exists(p) and not os.path.exists(p + ".orig"): os.rename(p, p + ".orig")
elif os.path.exists(p): os.unlink(p)
open(p, "w").write("\n".join(map(str, ids)) + "\n"); open(os.path.dirname(os.path.dirname(os.path.dirname(L))) + "/seed_ring.txt", "w").write(str(ring) + "\n")
print(f"seed ring {ring}: {len(ids)} seeds written to {p}")
PYE
  ;;
index)
  THR=$(cat preproc_thr.txt); SR=$(cat seed_ring.txt 2>/dev/null || true); [ -n "$SR" ] || { echo "run the seed stage first"; exit 1; }
  set_gate $RUN/real
  ( cd real; sed -i.bak "s/^OverAllRingToIndex.*/OverAllRingToIndex $SR/" Parameters.txt
    run "$P/midas-pipeline run --scan-mode ff --params Parameters.txt --result results --layers 1-1 --preproc-thresh $THR --n-cpus $NCPUS --indexer-backend c-omp --refine-backend c-omp --binning-device cpu --no-refresh-params --resume auto > index.log 2>&1; echo index exit=\$?" )
  echo "A zero-grain arm exits 1 in process_grains < 0.13.2 (a crash on the empty list, fixed in the wave-3 release); 'grains = 0' is then the result."
  ;;
nulls)
  THR=$(cat preproc_thr.txt); RINGS=$(cat rings.txt); SR=$(cat seed_ring.txt)
  for k in 1 2 3; do
    rm -rf $RUN/null$k      # a re-run starts the arm from scratch (its stages would otherwise be skipped)
    run "$PY -B -I $KIT/ff/run_arm.py --src $LAY --params $RUN/real/Parameters.txt --out $RUN/null$k --null-seed $k --f 1.0 --rings $RINGS --band-px 12 > $RUN/null$k.prep.log 2>&1"
    [ "${DRY:-0}" = 1 ] || LAYER=$RUN/null$k/results/LayerNr_1 RING=$SR $PY -B -I - <<'PYE'
import os
import pandas as pd
L = os.environ["LAYER"]; ring = int(os.environ["RING"]); p = L + "/SpotsToIndex.csv"
# the arm's own scrambled omega decides which ring spots fall in the seed half-range, exactly as for the real arm
ia = pd.read_csv(L + "/InputAll.csv", sep=r"\s+", usecols=["SpotID", "RingNumber", "Omega"])
ids = ia[(ia.RingNumber == ring) & (ia.Omega >= -90) & (ia.Omega <= 90)].SpotID.astype(int).values
if os.path.exists(p): os.unlink(p)
open(p, "w").write("\n".join(map(str, ids)) + "\n")
PYE
    [ "${DRY:-0}" = 1 ] || set_gate $RUN/null$k      # run_arm rebuilt this arm's paramstest from the gate baked into the zip at prep
    ( cd $RUN/null$k; run "$P/midas-pipeline run --scan-mode ff --params Parameters.txt --result results --layers 1-1 --preproc-thresh $THR --n-cpus $((NCPUS/3)) --indexer-backend c-omp --refine-backend c-omp --binning-device cpu --no-refresh-params --resume auto > index.log 2>&1; echo null$k exit=\$?" )
  done ;;
readout)
  $PY -B -I "$KIT/ff/ff_readout.py" "$RUN" ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac

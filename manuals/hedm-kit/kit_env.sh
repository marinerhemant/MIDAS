# Sourced by every kit script.  Usage:   export EXP=<name>   (a folder under experiments/)   then   source kit_env.sh
#   EXP     experiment folder name under $KIT/experiments/ (its experiment.env defines RAW, GROUP, conventions)
#   WORK    where YOUR outputs go. Default /data/$USER/<EXP>_work if /data is writable (large local disk, but local to
#           THIS host: results made on toro are not visible on chutoro), else $HOME/<EXP>_work. FF needs ~2x the raw
#           layer size per layer (zarr), NF ~10 GB per layer.
#   NCPUS   cores for the c-omp stages (default: nproc, capped at 64). The hosts are SHARED: export NCPUS=16 before sourcing unless you need more.
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP=${EXP:?set EXP=<experiment> (a folder under $KIT/experiments), e.g. export EXP=<experiment>}
EXPDIR=$KIT/experiments/$EXP
[ -r "$EXPDIR/experiment.env" ] || { echo "ERROR: no $EXPDIR/experiment.env"; return 1 2>/dev/null || exit 1; }
source "$EXPDIR/experiment.env"            # RAW, GROUP, IMTRANS, SKIPFRAME, NPIX, PX_UM, ENERGY_KEV, ENERGY_SOURCE, ...
if [ -z "${WORK:-}" ]; then
  if [ -w /data ]; then WORK=/data/$USER/${EXP}_work; else WORK=$HOME/${EXP}_work; fi
fi
NCPUS=${NCPUS:-$(n=$(nproc); [ "$n" -gt 64 ] && n=64; echo "$n")}
P=${MIDAS_ENV:-/home/beams12/S1IDUSER/opt/envs/midas}/bin     # shared MIDAS env, world-readable
PY=$P/python
export PATH=$P:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE PYTHONDONTWRITEBYTECODE=1   # never write __pycache__ into the (shared, read-only) kit

kit_check() {
  [ -x "$PY" ] || { echo "ERROR: shared env not found at $P (are you on toro or chutoro?)"; return 1; }
  [ -r "$RAW" ] || { echo "ERROR: cannot read $RAW -- your account must be in group $GROUP (run: id)"; return 1; }
  mkdir -p "$WORK" || { echo "ERROR: cannot create WORK=$WORK"; return 1; }
  echo "kit $KIT | exp $EXP | work $WORK | user $(whoami)@$(hostname -s) | $NCPUS cpus | $(df -h "$WORK" | awk 'NR==2{print $4" free"}')"
}

# least-busy GPUs by UTILISATION (not free memory), as a comma list of the first N
pick_gpus() {
  local n=${1:-1}
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | sort -t, -k2n -k3n | head -"$n" | cut -d, -f1 | tr -d ' ' | paste -sd, -
}

say() { echo; echo "######## $* ########"; date; }

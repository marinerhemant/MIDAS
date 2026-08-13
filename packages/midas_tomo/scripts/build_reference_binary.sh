#!/usr/bin/env bash
# Build a REFERENCE MIDAS_TOMO from the pristine, pre-fork TOMO/src/.
#
# This is what the packaging-parity test compares against. The question it
# answers is the only one that matters for a repackage: "does the packaged
# build produce the same numbers as the code it was forked from?"
#
# Why not use the committed TOMO/bin/MIDAS_TOMO: that binary is macOS arm64,
# built in 2023 by a different compiler. Comparing against it would conflate
# a packaging regression with a platform difference. Building both from
# source on one machine with identical flags isolates the packaging change.
#
# Usage:  ./dev/build_reference_binary.sh <path-to-MIDAS-checkout> [outdir]
set -euo pipefail

MIDAS_ROOT="${1:?usage: $0 <path-to-MIDAS-checkout> [outdir]}"
OUTDIR="${2:-$(pwd)/dev/refbin}"
SRC="$MIDAS_ROOT/TOMO/src"

if [ ! -f "$SRC/tomo_gridrec.c" ]; then
    echo "ERROR: $SRC does not look like TOMO/src (no tomo_gridrec.c)." >&2
    exit 1
fi

mkdir -p "$OUTDIR"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$SRC"/tomo_init.c "$SRC"/tomo_gridrec.c "$SRC"/tomo_utils.c \
   "$SRC"/tomo_cleanup.c "$SRC"/tomo_heads.h "$WORK/"

# The legacy build derives this from git; the content does not affect results.
cat > "$WORK/midas_version.h" <<'EOF'
#ifndef MIDAS_VERSION_H
#define MIDAS_VERSION_H
#define MIDAS_VERSION "reference-prefork"
#define MIDAS_GIT_HASH ""
#define MIDAS_GIT_DATE ""
#define MIDAS_VERSION_STRING "MIDAS_TOMO reference (pre-fork)"
#endif
EOF

# EXACTLY the flags packages/midas_tomo/CMakeLists.txt applies. Optimisation
# level changes FP instruction selection, so a mismatch here would make the
# comparison meaningless -- it would be measuring the flags, not the code.
#
# gnu99, not c99: CMake's C_STANDARD 99 leaves C_EXTENSIONS ON by default and
# so emits -std=gnu99. Under strict -std=c99 the build fails outright, because
# M_PI is a POSIX/GNU extension rather than standard C99 and -DPI=M_PI then
# expands to an undeclared identifier.
FLAGS="-fPIC -O3 -w -g -std=gnu99 -fopenmp -DPI=M_PI"

: "${FFTW_PREFIX:=}"
INC=""; LIB=""
if [ -n "$FFTW_PREFIX" ]; then
    INC="-I$FFTW_PREFIX/include"
    LIB="-L$FFTW_PREFIX/lib"
fi

# shellcheck disable=SC2086
gcc $FLAGS -I"$WORK" $INC \
    "$WORK"/tomo_init.c "$WORK"/tomo_gridrec.c "$WORK"/tomo_utils.c \
    "$WORK"/tomo_cleanup.c \
    $LIB -lfftw3f -lhdf5 -lm -o "$OUTDIR/MIDAS_TOMO_REF"

echo "Built $OUTDIR/MIDAS_TOMO_REF from $SRC"
echo "Point the parity test at it:"
echo "  export MIDAS_TOMO_REFERENCE_BIN=$OUTDIR/MIDAS_TOMO_REF"

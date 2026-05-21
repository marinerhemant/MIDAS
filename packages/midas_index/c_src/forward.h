/* forward.h — MIDAS shared diffraction forward model (midas_ckernel).
 *
 * ONE theoretical-spot simulator shared by the c-omp indexer (orientation
 * search, IndexerUnified.c) and the c-omp refiner (position+strain fit,
 * FitUnified.c). It replaces two near-duplicate copies that had drifted:
 *
 *   - indexer  : CalcDiffrSpots()      (IndexerUnified.c, RealType, 16-col out,
 *                                        RingsToReject fraction tally)
 *   - refiner  : CalcDiffractionSpots()(CalcDiffractionSpots.c, double, 9-col
 *                                        out, BigDetector mask, GCr position)
 *
 * Design rulings (see midas_fit_grain/dev/PORT_PLAN_c_refine_backend.md §9):
 *
 *  R1. fp64 everywhere. `RealType` is fixed to `double` (the indexer already
 *      builds with -DRealType=double for its parity gates).
 *
 *  R2. `v = sin(theta·deg2rad)·|G_hkl|` is computed from the UN-ROTATED
 *      reciprocal vector magnitude, exactly as the indexer precomputes it
 *      (IndexerUnified.c:2628-2631). This is what preserves the locked
 *      500/500 bit-level indexer parity. The legacy refiner used |OM·G_hkl|
 *      (ULP-different, mathematically identical); the newly-ported refiner
 *      adopts the indexer form and is its own reference.
 *
 *  R3. Output column layout is the indexer's (cols 0-9, 14, 15), FROZEN
 *      because IndexerUnified.c's CompareSpots / DoIndexing read those exact
 *      offsets. Refiner-only outputs (the scaled reciprocal vector GCr used by
 *      the spatial position objective) live in APPENDED columns 16-18, so the
 *      indexer is bit-unaffected. See the column map below.
 *
 *  R4. Both the indexer's RingsToReject fraction tally AND the refiner's
 *      BigDetector active-area mask are supported, gated by NULL/zero args, so
 *      a single body serves both callers with no behavioral change to either.
 */
#ifndef MIDAS_CKERNEL_FORWARD_H
#define MIDAS_CKERNEL_FORWARD_H

#ifndef RealType
#define RealType double
#endif

/* Wide enough for the indexer's frozen 0-15 layout plus refiner GCr (16-18). */
#define MIDAS_CK_NCOLS 19

/* ---------------------------------------------------------------------------
 * hkls table — CANONICAL (indexer) layout. One row per reflection:
 *   [0..2] hc,kc,lc   reciprocal-lattice (cartesian) vector components
 *   [3]    ringnr     (double-cast int)
 *   [4]    Ds         |G| used to scale the output GCr unit vector
 *   [5]    theta      Bragg angle, degrees
 *   [6]    RingRadius detector ring radius (µm)
 *   [7]    sin(theta·deg2rad)
 *   [8]    v   = [7]·|G_hkl|     (precomputed; see R2)
 *   [9]    vSq = v·v
 * A refiner-native table ([0..2]=G, [3]=Ds, [4]=theta, [5]=RingRadius,
 * [6]=RingNr) is converted to this layout by midas_ck_hkls_from_refiner().
 *
 * Output spot row (MIDAS_CK_NCOLS wide):
 *   [0]  OrientID   (caller sets via orient_id arg)
 *   [1]  spotid     (running, per call)
 *   [2]  indexhkl   (source hkl row)
 *   [3]  distance
 *   [4]  yl         lab-frame detector y (µm)
 *   [5]  zl         lab-frame detector z (µm)
 *   [6]  omega      (deg)
 *   [7]  eta        (deg)
 *   [8]  theta      (deg)
 *   [9]  ringnr
 *   [10] (caller-filled: displaced y)   left untouched
 *   [11] (caller-filled: displaced z)   left untouched
 *   [12] (caller-filled: eta@displaced) left untouched
 *   [13] (reserved)                     left untouched
 *   [14] sinOme
 *   [15] cosOme
 *   [16] GCr0  = Ds·Gc0/|Gc|   (refiner spatial objective)
 *   [17] GCr1
 *   [18] GCr2
 * ------------------------------------------------------------------------- */

/* BigDetector active-area mask context (refiner). All-zero / NULL => no mask. */
typedef struct {
  int big_det_size;            /* BigDetSize; 0 => mask disabled */
  const unsigned int *mask;    /* BigDetector bitset, or NULL */
  double pixelsize;            /* µm per pixel */
} MidasCkBigDet;

/* Shared CalcOmega — precomputed-v form (takes v, vSq; outputs cos/sinOmes).
 * Exposed so the indexer's FF Friedel-pair helpers reuse it (no duplicate). */
void midas_ck_calc_omega(RealType x, RealType y, RealType z, RealType v,
                         RealType vSq, RealType omegas[4], RealType etas[4],
                         RealType cosOmes[4], RealType sinOmes[4], int *nsol);

/* Shared CalcSpotPosition (yl,zl from RingRadius+eta). */
void midas_ck_calc_spot_position(RealType RingRadius, RealType eta,
                                 RealType *yl, RealType *zl);

/* Unified forward simulator. Returns 0 on success.
 *
 *   OrientMatrix   3x3 orientation (row-major via [3][3])
 *   distance       sample-to-detector (µm)
 *   RingRadii      per-ring radius table indexed by ringnr (indexer's
 *                  param-file values), or NULL => use hkls[*][6] (refiner).
 *                  Indexer and refiner draw RingRadius from DIFFERENT sources
 *                  (param file vs hkl file); this gate keeps both bit-faithful.
 *   hkls           canonical-layout table, n_hkls rows
 *   OmegaRange     [NOmegaRanges][2] (deg)
 *   BoxSizes       [NOmegaRanges][4]  (yl_lo,yl_hi,zl_lo,zl_hi)
 *   ExcludePoleAngle (deg)
 *   ringsToReject  / nRingsToReject   indexer fraction-tally (NULL/0 => none)
 *   bigdet         BigDetector mask, or NULL (no mask)
 *   orient_id      value written to col 0
 *   spots          [>=2*n_hkls][MIDAS_CK_NCOLS] caller-allocated output
 *   nspots         OUT: number of spots emitted
 *   nSpotsFracCalc OUT: spots NOT on a rejected ring (NULL => not computed)
 */
int midas_ck_calc_diffraction_spots(
    RealType OrientMatrix[3][3], RealType distance, const RealType *RingRadii,
    double **hkls, int n_hkls, RealType OmegaRange[][2], RealType BoxSizes[][4],
    int NOmegaRanges, RealType ExcludePoleAngle, const int *ringsToReject,
    int nRingsToReject, const MidasCkBigDet *bigdet, int orient_id,
    RealType **spots, int *nspots, int *nSpotsFracCalc);

/* Convert a refiner-native hkls row-set to the canonical layout in `out`
 * (caller-allocated, n_hkls x 10). `in` rows: [0..2]=G,[3]=Ds,[4]=theta(deg),
 * [5]=RingRadius,[6]=RingNr. */
void midas_ck_hkls_from_refiner(double **in, int n_hkls, double **out);

#endif /* MIDAS_CKERNEL_FORWARD_H */

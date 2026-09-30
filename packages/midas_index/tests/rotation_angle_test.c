/* rotation_angle_test.c — drive the C indexer's candidate-sweep angle directly.
 *
 * CalcRotationAngle / RotationAngleFor are `static` in IndexerUnified.c, so
 * this TU #includes the whole translation unit (main renamed out of the way),
 * as comparespots_weight_test.c does. It reads lines
 *
 *     sg a b c alpha beta gamma h k l
 *
 * from stdin and prints the sweep angle for each, one per line, going through
 * the real entry point: SGNum / ABCABG / RingHKLint are set exactly as
 * ReadParams and the hkls.csv reader set them, then CalcRotationAngle(1) is
 * called. tests/test_rotation_angle_laue.py compares every line against
 * midas_hkls' Seitz operators.
 *
 * Build (from packages/midas_index):
 *   cc -std=gnu99 -fopenmp -O2 -I c_src -I <builddir> \
 *      tests/rotation_angle_test.c c_src/MIDAS_Math.c \
 *      c_src/GetMisorientation.c c_src/forward.c -lm -o rot_angle
 */
#define main midas_indexer_main_under_test
#include "IndexerUnified.c"
#undef main

int main(void) {
  char line[512];
  while (fgets(line, sizeof line, stdin)) {
    int sg, h, k, l;
    double abc[6];
    if (sscanf(line, "%d %lf %lf %lf %lf %lf %lf %d %d %d", &sg, &abc[0],
               &abc[1], &abc[2], &abc[3], &abc[4], &abc[5], &h, &k, &l) != 10)
      continue;
    SGNum = sg;
    for (int i = 0; i < 6; i++) ABCABG[i] = abc[i];
    RingHKLint[1][0] = h;
    RingHKLint[1][1] = k;
    RingHKLint[1][2] = l;
    printf("%.6f\n", CalcRotationAngle(1));
  }
  return 0;
}

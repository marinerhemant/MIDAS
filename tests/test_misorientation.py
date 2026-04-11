#!/usr/bin/env python3
"""
Comprehensive test suite for GetMisorientation.c / calcMiso.py

Tests cover:
  A. Symmetry operator correctness (group closure, norms, rotation angles)
  B. Euler / OrientMat / Quat conversion correctness
  C. Misorientation correctness by crystal system
  D. Symmetry reduction validation
  E. Numerical robustness
  F. Batch API validation (after C library is built)
  G. C vs Python cross-validation (after ctypes wrapper is built)

Run:  python tests/test_misorientation.py
"""
import sys, os, json, math, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from calcMiso import (
    Euler2OrientMat, OrientMat2Euler, OrientMat2Quat,
    GetMisOrientationAngle, GetMisOrientationAngleOM,
    MakeSymmetries, BringDownToFundamentalRegionSym, QuaternionProduct,
    normalize
)

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0

def check(condition, name, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name}")
        if detail:
            print(f"        {detail}")

def skip(name, reason=""):
    global SKIP_COUNT
    SKIP_COUNT += 1

# ──────────────────────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────────────────────

def canonicalize_quat(q):
    """Canonical form: first nonzero component is positive."""
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([1.0, 0, 0, 0])
    q = q / n
    for i in range(4):
        if abs(q[i]) > 1e-10:
            if q[i] < 0:
                q = -q
            break
    return q

def hamilton_product(a, b):
    """Quaternion Hamilton product a*b."""
    return np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    ])

def quat_to_angle_axis(q):
    """Decode quaternion to (angle_deg, axis)."""
    q = canonicalize_quat(q)
    w = min(1.0, max(-1.0, abs(q[0])))
    angle = 2.0 * math.acos(w)
    s = math.sin(angle / 2.0)
    if s < 1e-10:
        axis = np.array([0, 0, 1.0])
    else:
        axis = q[1:4] / s
        if q[0] < 0:
            axis = -axis
    return math.degrees(angle), axis

def quat_to_rotmat(q):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    q = np.array(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def make_rotation_quat(angle_deg, axis):
    """Create quaternion for rotation of angle_deg about axis."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), math.sin(half)*axis[0],
                     math.sin(half)*axis[1], math.sin(half)*axis[2]])

def group_closure_check(quats):
    """Check if quaternions form a closed group. Returns list of violations."""
    n = len(quats)
    normed = [canonicalize_quat(q) for q in quats]
    violations = []
    for i in range(n):
        for j in range(n):
            prod = hamilton_product(normed[i], normed[j])
            prod = canonicalize_quat(prod)
            found = any(np.allclose(prod, normed[k], atol=1e-3) for k in range(n))
            if not found:
                violations.append((i, j, prod))
    return violations


# ══════════════════════════════════════════════════════════════
#  SECTION A: Symmetry operator correctness
# ══════════════════════════════════════════════════════════════

def test_A_symmetry_operators():
    print("\n=== A. Symmetry Operator Correctness ===")

    # A1. Group closure for all current tables
    print("  A1. Group closure...")
    for sg, expected_n, label in [
        (1, 1, "Triclinic"), (10, 2, "Monoclinic"), (62, 4, "Orthorhombic"),
        (139, 8, "Tetragonal high"), (150, 6, "Trigonal type1"),
        (194, 12, "Hexagonal high"), (225, 24, "Cubic high"),
    ]:
        n, sym = MakeSymmetries(sg)
        violations = group_closure_check(sym[:n])
        check(len(violations) == 0,
              f"A1: Group closure SG {sg} ({label})",
              f"{len(violations)} violations" if violations else "")

    # A2. Operator counts at SG boundaries
    print("  A2. Operator counts at SG boundaries...")
    # These test EXPECTED behavior after fixes; some will fail against current code
    expected_counts = {
        1: 1, 2: 1,       # Triclinic
        3: 2, 15: 2,      # Monoclinic
        16: 4, 74: 4,     # Orthorhombic
        75: 4, 88: 4,     # Tetragonal LOW  (currently returns 8 — known bug)
        89: 8, 142: 8,    # Tetragonal HIGH
        143: 3, 148: 3,   # Trigonal LOW    (currently returns 6 — known bug)
        149: 6, 167: 6,   # Trigonal HIGH
        168: 6, 176: 6,   # Hexagonal LOW  (currently returns 12 — known bug)
        177: 12, 194: 12, # Hexagonal HIGH
        195: 12, 206: 12, # Cubic LOW      (currently returns 24 — known bug)
        207: 24, 230: 24, # Cubic HIGH
    }
    for sg, exp in expected_counts.items():
        n, _ = MakeSymmetries(sg)
        check(n == exp,
              f"A2: MakeSymmetries({sg}) count",
              f"got {n}, expected {exp}")

    # A3. Unit quaternion norms
    print("  A3. Unit quaternion norms...")
    for sg in [1, 10, 62, 139, 150, 194, 225]:
        n, sym = MakeSymmetries(sg)
        all_unit = True
        for i in range(n):
            norm = np.linalg.norm(sym[i])
            if abs(norm - 1.0) > 1e-4:
                all_unit = False
                break
        check(all_unit, f"A3: All quaternions unit norm for SG {sg}",
              f"sym[{i}] norm={norm:.6f}" if not all_unit else "")

    # A4. OrtSym[1] should be 180° about X (not 90°)
    print("  A4. OrtSym fix...")
    n, sym = MakeSymmetries(62)
    q1 = sym[1]
    angle, axis = quat_to_angle_axis(q1)
    check(abs(angle - 180.0) < 1.0,
          "A4: OrtSym[1] is 180° rotation",
          f"got {angle:.1f}°")
    check(abs(np.linalg.norm(q1) - 1.0) < 1e-4,
          "A4: OrtSym[1] is unit quaternion",
          f"norm={np.linalg.norm(q1):.6f}")

    # A5. Trigonal Type 1 vs Type 2 axes
    print("  A5. Trigonal Type 1 vs Type 2...")
    n1, sym1 = MakeSymmetries(150)  # Type 1
    n2, sym2 = MakeSymmetries(149)  # Type 2
    # Extract 180° axes
    axes_1 = []
    axes_2 = []
    for i in range(n1):
        angle, ax = quat_to_angle_axis(sym1[i])
        if abs(angle - 180) < 1:
            axes_1.append(np.degrees(np.arctan2(ax[1], ax[0])))
    for i in range(n2):
        angle, ax = quat_to_angle_axis(sym2[i])
        if abs(angle - 180) < 1:
            axes_2.append(np.degrees(np.arctan2(ax[1], ax[0])))
    axes_1.sort()
    axes_2.sort()
    # Type 1 should have axes at -30, 30, 90; Type 2 at -60, 0, 60
    check(len(axes_1) == 3, "A5: Trigonal Type 1 has 3 two-fold axes",
          f"got {len(axes_1)}")
    check(len(axes_2) == 3, "A5: Trigonal Type 2 has 3 two-fold axes",
          f"got {len(axes_2)}")
    if len(axes_1) == 3 and len(axes_2) == 3:
        # Check they're different
        check(not np.allclose(sorted(axes_1), sorted(axes_2), atol=5),
              "A5: Trigonal Type 1 and 2 have different 2-fold axes",
              f"Type1={axes_1}, Type2={axes_2}")

    # A6. Monoclinic axis from lattice params
    print("  A6. Monoclinic dynamic axis (post-fix test)...")
    # This tests MakeSymmetriesWithLattice which doesn't exist yet.
    # For now, document expected behavior.
    n, sym = MakeSymmetries(10)
    angle, axis = quat_to_angle_axis(sym[1])
    check(abs(angle - 180.0) < 1.0,
          "A6: MonoSym[1] is 180° rotation",
          f"got {angle:.1f}°")
    # Document current axis (X = {0,1,0,0})
    current_axis_x = abs(axis[0]) > 0.9 and abs(axis[1]) < 0.1 and abs(axis[2]) < 0.1
    current_axis_y = abs(axis[1]) > 0.9 and abs(axis[0]) < 0.1 and abs(axis[2]) < 0.1
    # After fix, default should be Y (b-unique)
    check(current_axis_y,
          "A6: MonoSym default should be b-unique (Y-axis)",
          f"axis=[{axis[0]:.3f},{axis[1]:.3f},{axis[2]:.3f}]")


# ══════════════════════════════════════════════════════════════
#  SECTION B: Conversion correctness
# ══════════════════════════════════════════════════════════════

def test_B_conversions():
    print("\n=== B. Euler / OrientMat / Quat Conversions ===")
    np.random.seed(42)

    # B1. Euler→OM→Euler round-trip
    print("  B1. Euler→OM→Euler round-trip (100 random)...")
    max_err = 0
    for _ in range(100):
        e = [np.random.uniform(0, 2*np.pi),
             np.random.uniform(0.01, np.pi-0.01),  # avoid gimbal lock
             np.random.uniform(0, 2*np.pi)]
        om = Euler2OrientMat(e)
        e2 = OrientMat2Euler(np.array(om).reshape(3,3))
        om2 = Euler2OrientMat(list(e2))
        err = np.max(np.abs(np.array(om) - np.array(om2)))
        max_err = max(max_err, err)
    check(max_err < 1e-10,
          f"B1: Euler→OM→Euler round-trip",
          f"max_err={max_err:.2e}")

    # B2. OM→Quat→OM round-trip
    print("  B2. OM→Quat→OM round-trip (100 random)...")
    max_err = 0
    for _ in range(100):
        e = [np.random.uniform(0, 2*np.pi),
             np.random.uniform(0, np.pi),
             np.random.uniform(0, 2*np.pi)]
        om = np.array(Euler2OrientMat(e))
        q = OrientMat2Quat(list(om))
        om2 = quat_to_rotmat(q).flatten()
        err = np.max(np.abs(om - om2))
        max_err = max(max_err, err)
    check(max_err < 1e-10,
          f"B2: OM→Quat→OM round-trip",
          f"max_err={max_err:.2e}")

    # B3. Known identity
    print("  B3. Known identity...")
    om = np.array(Euler2OrientMat([0, 0, 0])).reshape(3,3)
    check(np.allclose(om, np.eye(3), atol=1e-12),
          "B3: (0,0,0) Euler → identity matrix")

    # B4. Determinant and orthogonality
    print("  B4. Determinant and orthogonality (100 random)...")
    max_det_err = 0
    max_orth_err = 0
    for _ in range(100):
        e = [np.random.uniform(0, 2*np.pi),
             np.random.uniform(0, np.pi),
             np.random.uniform(0, 2*np.pi)]
        om = np.array(Euler2OrientMat(e)).reshape(3,3)
        det_err = abs(np.linalg.det(om) - 1.0)
        orth_err = np.max(np.abs(om @ om.T - np.eye(3)))
        max_det_err = max(max_det_err, det_err)
        max_orth_err = max(max_orth_err, orth_err)
    check(max_det_err < 1e-12,
          f"B4: All OMs have det=+1",
          f"max_det_err={max_det_err:.2e}")
    check(max_orth_err < 1e-12,
          f"B4: All OMs are orthogonal (R^T R = I)",
          f"max_orth_err={max_orth_err:.2e}")

    # B5. Gimbal lock (phi=0)
    print("  B5. Gimbal lock edge cases...")
    for phi in [0.0, math.pi]:
        e = [1.5, phi, 2.3]
        om = Euler2OrientMat(e)
        e2 = OrientMat2Euler(np.array(om).reshape(3,3))
        check(not any(np.isnan(e2)),
              f"B5: No NaN at phi={phi:.4f}",
              f"result={e2}")
        om2 = Euler2OrientMat(list(e2))
        err = np.max(np.abs(np.array(om) - np.array(om2)))
        check(err < 1e-10,
              f"B5: Round-trip at phi={phi:.4f}",
              f"err={err:.2e}")


# ══════════════════════════════════════════════════════════════
#  SECTION C: Misorientation by crystal system
# ══════════════════════════════════════════════════════════════

def test_C_misorientation():
    print("\n=== C. Misorientation Correctness by Crystal System ===")

    # C1. Identity misorientation
    print("  C1. Identity misorientation...")
    for sg, label in [(225, "Cubic"), (194, "Hex"), (139, "Tet"), (62, "Ort"), (1, "Tric")]:
        e = [1.0, 0.5, 2.0]
        angle, _ = GetMisOrientationAngle(e, e, sg)
        check(abs(angle) < 1e-10,
              f"C1: Identity miso SG {sg} ({label})",
              f"got {angle:.2e}")

    # C2. Cubic Sigma-3 twin (60° about <111>)
    print("  C2. Known misorientations...")
    om_id = Euler2OrientMat([0, 0, 0])
    q_sigma3 = make_rotation_quat(60.0, [1, 1, 1])
    om_sigma3 = list(quat_to_rotmat(q_sigma3).flatten())
    angle, _ = GetMisOrientationAngleOM(om_id, om_sigma3, 225)
    check(abs(math.degrees(angle) - 60.0) < 0.01,
          "C2: Cubic Sigma-3 twin = 60°",
          f"got {math.degrees(angle):.4f}°")

    # C3. Cubic Sigma-5 twin (36.87° about <100>)
    q_sigma5 = make_rotation_quat(36.87, [1, 0, 0])
    om_sigma5 = list(quat_to_rotmat(q_sigma5).flatten())
    angle, _ = GetMisOrientationAngleOM(om_id, om_sigma5, 225)
    check(abs(math.degrees(angle) - 36.87) < 0.1,
          "C3: Cubic Sigma-5 twin ≈ 36.87°",
          f"got {math.degrees(angle):.4f}°")

    # C4. Hexagonal: 30° about c-axis
    q_hex30 = make_rotation_quat(30.0, [0, 0, 1])
    om_hex30 = list(quat_to_rotmat(q_hex30).flatten())
    angle, _ = GetMisOrientationAngleOM(om_id, om_hex30, 194)
    check(abs(math.degrees(angle) - 30.0) < 0.01,
          "C4: Hexagonal 30° about c",
          f"got {math.degrees(angle):.4f}°")

    # C5. Orthorhombic: 180° about one axis should be 0 (symmetry-equivalent)
    q_ort180y = make_rotation_quat(180.0, [0, 1, 0])
    om_ort180y = list(quat_to_rotmat(q_ort180y).flatten())
    angle, _ = GetMisOrientationAngleOM(om_id, om_ort180y, 62)
    check(abs(math.degrees(angle)) < 0.01,
          "C5: Orthorhombic 180° about Y = identity (symmetry)",
          f"got {math.degrees(angle):.4f}°")

    # C6. Triclinic: raw angle, no symmetry reduction
    q_raw = make_rotation_quat(45.0, [1, 0, 0])
    om_raw = list(quat_to_rotmat(q_raw).flatten())
    angle, _ = GetMisOrientationAngleOM(om_id, om_raw, 1)
    check(abs(math.degrees(angle) - 45.0) < 0.01,
          "C6: Triclinic raw angle = 45°",
          f"got {math.degrees(angle):.4f}°")

    # C7. Low vs high Laue monotonicity
    print("  C7. Low vs high Laue monotonicity...")
    np.random.seed(123)
    e1 = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
    e2 = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
    om1 = Euler2OrientMat(e1)
    om2 = Euler2OrientMat(e2)
    pairs = [
        (87, 139, "Tet low vs high"),
        (148, 150, "Trig low vs high"),
        (176, 194, "Hex low vs high"),
        (200, 225, "Cub low vs high"),
    ]
    for sg_low, sg_high, label in pairs:
        a_low, _ = GetMisOrientationAngleOM(om1, om2, sg_low)
        a_high, _ = GetMisOrientationAngleOM(om1, om2, sg_high)
        check(a_low >= a_high - 1e-10,
              f"C7: {label}: miso(low) >= miso(high)",
              f"low={math.degrees(a_low):.4f}° high={math.degrees(a_high):.4f}°")

    # C8. Regression baseline for cubic SG 225
    print("  C8. Regression baseline for cubic SG 225...")
    baseline_path = os.path.join(os.path.dirname(__file__), 'regression_baseline_cubic225.json')
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        for i, (pair, ref) in enumerate(zip(baseline['miso_inputs_euler'], baseline['miso_results'])):
            e1, e2 = pair
            angle, _ = GetMisOrientationAngle(e1, e2, 225)
            ref_angle = ref['angle_rad']
            check(abs(angle - ref_angle) < 1e-12,
                  f"C8: Regression pair {i}",
                  f"got {angle:.15f}, expected {ref_angle:.15f}")
    else:
        skip("C8: Regression baseline file not found")


# ══════════════════════════════════════════════════════════════
#  SECTION D: Symmetry reduction validation
# ══════════════════════════════════════════════════════════════

def test_D_symmetry_reduction():
    print("\n=== D. Symmetry Reduction Validation ===")
    np.random.seed(456)

    # D1. Symmetry-equivalent orientations give same misorientation
    print("  D1. Symmetry-equivalent orientations...")
    for sg, label in [(225, "Cubic"), (194, "Hex"), (62, "Ort")]:
        n, sym = MakeSymmetries(sg)
        e_ref = [1.2, 0.8, 3.1]
        e_test = [0.5, 1.2, 2.3]
        angle_ref, _ = GetMisOrientationAngle(e_ref, e_test, sg)
        # Apply a symmetry operation to e_test
        om_test = np.array(Euler2OrientMat(e_test))
        q_test = OrientMat2Quat(list(om_test))
        q_sym = QuaternionProduct(list(q_test), list(sym[1]))  # apply sym[1]
        om_sym = list(quat_to_rotmat(q_sym).flatten())
        e_sym = list(OrientMat2Euler(np.array(om_sym).reshape(3,3)))
        angle_sym, _ = GetMisOrientationAngle(e_ref, e_sym, sg)
        check(abs(angle_ref - angle_sym) < 1e-8,
              f"D1: Sym-equiv miso same for SG {sg} ({label})",
              f"ref={math.degrees(angle_ref):.6f}° sym={math.degrees(angle_sym):.6f}°")

    # D2. Commutativity
    print("  D2. Commutativity...")
    for sg in [225, 194, 139, 62, 10, 1]:
        e1 = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
        e2 = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
        a12, _ = GetMisOrientationAngle(e1, e2, sg)
        a21, _ = GetMisOrientationAngle(e2, e1, sg)
        check(abs(a12 - a21) < 1e-10,
              f"D2: Commutativity SG {sg}",
              f"miso(1,2)={math.degrees(a12):.6f}° miso(2,1)={math.degrees(a21):.6f}°")

    # D3. Misorientation <= max disorientation angle
    print("  D3. Max disorientation bounds...")
    # Max disorientation angles for fundamental zone of each Laue group
    max_angles_deg = {
        225: 62.9,   # cubic m-3m (Mackenzie: 62.80 deg)
        194: 94.0,   # hexagonal 6/mmm (Mackenzie: ~93.84 deg)
        139: 99.0,   # tetragonal 4/mmm (~98.4 deg max disorientation)
        62:  121.0,  # orthorhombic mmm (120 deg)
        10:  180.0,  # monoclinic
        1:   180.0,  # triclinic
    }
    for sg, max_deg in max_angles_deg.items():
        all_below = True
        for _ in range(50):
            e1 = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
            e2 = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
            angle, _ = GetMisOrientationAngle(e1, e2, sg)
            if math.degrees(angle) > max_deg + 0.1:
                all_below = False
                break
        check(all_below,
              f"D3: All misos <= {max_deg}° for SG {sg}",
              f"got {math.degrees(angle):.2f}°" if not all_below else "")

    # D4. Triangle inequality
    print("  D4. Triangle inequality...")
    for sg in [225, 194, 62]:
        violations = 0
        for _ in range(50):
            e1 = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
            e2 = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
            e3 = [np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
            a12, _ = GetMisOrientationAngle(e1, e2, sg)
            a23, _ = GetMisOrientationAngle(e2, e3, sg)
            a13, _ = GetMisOrientationAngle(e1, e3, sg)
            if a13 > a12 + a23 + 1e-8:
                violations += 1
        check(violations == 0,
              f"D4: Triangle inequality SG {sg}",
              f"{violations}/50 violations")


# ══════════════════════════════════════════════════════════════
#  SECTION E: Numerical robustness
# ══════════════════════════════════════════════════════════════

def test_E_numerical():
    print("\n=== E. Numerical Robustness ===")

    # E1. acos clamping (near-identity quaternion)
    print("  E1. acos clamping...")
    # Construct an orientation matrix that's nearly identity but has slight errors
    om_near_id = list(np.eye(3).flatten() + np.random.randn(9) * 1e-15)
    try:
        angle, _ = GetMisOrientationAngleOM(om_near_id, om_near_id, 225)
        check(not math.isnan(angle), "E1: Near-identity no NaN", f"angle={angle}")
    except Exception as e:
        check(False, "E1: Near-identity no exception", str(e))

    # E2. Non-mutation of inputs
    print("  E2. Non-mutation...")
    e1 = [1.0, 0.5, 2.0]
    e2 = [0.3, 1.1, 4.5]
    e1_copy = list(e1)
    e2_copy = list(e2)
    GetMisOrientationAngle(e1, e2, 225)
    check(e1 == e1_copy and e2 == e2_copy,
          "E2: GetMisOrientationAngle does not modify inputs")

    # E3. Very small misorientation
    print("  E3. Very small misorientation...")
    e_base = [1.0, 0.5, 2.0]
    e_perturbed = [1.0 + 1e-8, 0.5, 2.0]
    angle, _ = GetMisOrientationAngle(e_base, e_perturbed, 225)
    check(not math.isnan(angle) and angle >= 0,
          "E3: Very small misorientation is non-negative and not NaN",
          f"angle={angle:.2e}")

    # E4. Antipodal quaternions (q and -q are same rotation)
    print("  E4. Antipodal quaternion equivalence...")
    e1 = [1.0, 0.5, 2.0]
    e2 = [0.3, 1.1, 4.5]
    om1 = Euler2OrientMat(e1)
    om2 = Euler2OrientMat(e2)
    q1 = OrientMat2Quat(om1)
    q2 = OrientMat2Quat(om2)
    # Negate q2
    q2_neg = [-x for x in q2]
    om2_neg = list(quat_to_rotmat(q2_neg).flatten())
    a1, _ = GetMisOrientationAngleOM(om1, om2, 225)
    a2, _ = GetMisOrientationAngleOM(om1, om2_neg, 225)
    check(abs(a1 - a2) < 1e-10,
          "E4: q and -q give same misorientation",
          f"diff={abs(a1-a2):.2e}")


# ══════════════════════════════════════════════════════════════
#  SECTION F: Batch API (requires C library)
# ══════════════════════════════════════════════════════════════

def test_F_batch():
    print("\n=== F. Batch API Validation ===")
    try:
        from calcMiso import GetMisOrientationAngleBatch, GetMisOrientationAngleOMBatch
    except ImportError:
        skip("F: Batch API not available")
        print("  SKIPPED: Batch API not available")
        return

    np.random.seed(789)

    # F1. Batch OM matches scalar
    print("  F1. Batch OM matches scalar (100 pairs)...")
    n = 100
    oms1 = np.array([Euler2OrientMat([np.random.uniform(0, 2*np.pi),
                                       np.random.uniform(0, np.pi),
                                       np.random.uniform(0, 2*np.pi)])
                      for _ in range(n)])
    oms2 = np.array([Euler2OrientMat([np.random.uniform(0, 2*np.pi),
                                       np.random.uniform(0, np.pi),
                                       np.random.uniform(0, 2*np.pi)])
                      for _ in range(n)])
    batch_angles = GetMisOrientationAngleOMBatch(oms1, oms2, 225)
    max_diff = 0
    for i in range(n):
        scalar_angle, _ = GetMisOrientationAngleOM(list(oms1[i]), list(oms2[i]), 225)
        max_diff = max(max_diff, abs(batch_angles[i] - scalar_angle))
    check(max_diff < 1e-10,
          "F1: Batch OM matches scalar",
          f"max_diff={max_diff:.2e}")

    # F2. Batch quat matches scalar
    print("  F2. Batch quat matches scalar (100 pairs)...")
    quats1 = np.array([OrientMat2Quat(list(om)) for om in oms1])
    quats2 = np.array([OrientMat2Quat(list(om)) for om in oms2])
    batch_angles_q = GetMisOrientationAngleBatch(quats1, quats2, 225)
    max_diff_q = max(abs(batch_angles_q[i] - batch_angles[i]) for i in range(n))
    check(max_diff_q < 1e-10,
          "F2: Batch quat matches batch OM",
          f"max_diff={max_diff_q:.2e}")

    # F3. Batch with n=1
    print("  F3. Batch with n=1...")
    single = GetMisOrientationAngleOMBatch(oms1[:1], oms2[:1], 225)
    ref, _ = GetMisOrientationAngleOM(list(oms1[0]), list(oms2[0]), 225)
    check(abs(single[0] - ref) < 1e-12,
          "F3: Batch n=1 matches scalar")


# ══════════════════════════════════════════════════════════════
#  SECTION G: C vs Python cross-validation
# ══════════════════════════════════════════════════════════════

def test_G_cross_validation():
    print("\n=== G. C vs Python Cross-Validation ===")
    # Check if ctypes wrapper is available
    try:
        # Will be available after Phase 4 of the plan
        import ctypes
        lib_path = None
        for candidate in [
            os.path.join(os.path.dirname(__file__), '..', 'build', 'lib', 'libmidas_orientation.so'),
            os.path.join(os.path.dirname(__file__), '..', 'build', 'lib', 'libmidas_orientation.dylib'),
        ]:
            if os.path.exists(candidate):
                lib_path = candidate
                break
        if lib_path is None:
            raise FileNotFoundError("libmidas_orientation not found")
        has_c_lib = True
    except (ImportError, FileNotFoundError):
        has_c_lib = False

    if not has_c_lib:
        skip("G: C library not yet built")
        print("  SKIPPED: C library not yet built")
        return

    # TODO: Add C vs Python comparison tests once library is built


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Misorientation test suite")
    parser.add_argument('--sections', nargs='+', default=['A','B','C','D','E','F','G'],
                        help='Which test sections to run (default: all)')
    args = parser.parse_args()

    sections = {
        'A': test_A_symmetry_operators,
        'B': test_B_conversions,
        'C': test_C_misorientation,
        'D': test_D_symmetry_reduction,
        'E': test_E_numerical,
        'F': test_F_batch,
        'G': test_G_cross_validation,
    }

    for s in args.sections:
        if s.upper() in sections:
            sections[s.upper()]()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed, {SKIP_COUNT} skipped")
    print(f"{'='*60}")

    if FAIL_COUNT > 0:
        print("\nFailed tests indicate either known bugs (pre-fix) or regressions (post-fix).")
        print("Known pre-fix failures: A2 (low Laue counts), A4 (OrtSym), A5 (Trig types), A6 (Mono axis)")

    sys.exit(1 if FAIL_COUNT > 0 else 0)

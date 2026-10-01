#!/usr/bin/env python3
"""MIDAS logo concepts, v2: the suite, not just the spots.

Idea: X-rays in, structure out. A thin ray crosses a sample cell and leaves
as gold (the Midas touch: raw data turned into something solid).

  1a  ray -> wireframe cube -> gold voxel -> gold ray out
  1b  ray changes colour as it crosses the cube (simplest, best at 16 px)
  4   stacked slices pierced by a gold line (tomography-leaning alternative)

Writes SVGs (transparent master + dark/light previews) and PNG renders.
Run with the miniforge python (needs cairosvg for PNG only):
    /Users/travel_user/miniforge3/bin/python make_logo.py
"""
import math
import os

GOLD, GOLD_LT, GOLD_DK = "#E0AA45", "#F2CB7C", "#B07F22"
INK_D, INK_L = "#E9EDF0", "#1C2830"   # ink for dark / light backgrounds
BG_D, BG_L = "#1F262B", "#FFFFFF"
OUT = os.path.dirname(os.path.abspath(__file__))
C30 = math.cos(math.radians(30))


def proj(x, y, z, cx, cy, u):
    """Isometric projection of a point of the cube [-1,1]^3."""
    return (cx + (x - y) * C30 * u, cy + (x + y) * 0.5 * u - z * u)


def poly(pts, **kw):
    d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + "Z"
    attrs = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
    return f'<path d="{d}" {attrs}/>'


def line(p, q, ink, sw, op=1.0):
    return (f'<path d="M{p[0]:.2f},{p[1]:.2f} L{q[0]:.2f},{q[1]:.2f}" stroke="{ink}" '
            f'stroke-width="{sw}" stroke-opacity="{op}" stroke-linecap="round" fill="none"/>')


def cube_wire(cx, cy, u, ink, sw, inner_op=0.55):
    """Hexagon outline plus the three visible inner edges."""
    hexa = [proj(*v, cx, cy, u) for v in
            [(1, -1, 1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)]]
    c = proj(1, 1, 1, cx, cy, u)
    out = [poly(hexa, fill="none", stroke=ink, stroke_width=sw, stroke_linejoin="round")]
    for v in [(1, -1, 1), (-1, 1, 1), (1, 1, -1)]:
        out.append(line(c, proj(*v, cx, cy, u), ink, sw, inner_op))
    return "\n".join(out), hexa


def gold_cube(cx, cy, u, s):
    """Solid cube of half-size s (fraction of u) in three gold tones."""
    P = lambda x, y, z: proj(x * s, y * s, z * s, cx, cy, u)
    top = poly([P(1, -1, 1), P(-1, -1, 1), P(-1, 1, 1), P(1, 1, 1)], fill=GOLD_LT)
    left = poly([P(-1, 1, 1), P(-1, 1, -1), P(1, 1, -1), P(1, 1, 1)], fill=GOLD)
    right = poly([P(1, 1, 1), P(1, 1, -1), P(1, -1, -1), P(1, -1, 1)], fill=GOLD_DK)
    return "\n".join([top, left, right])


# ------------------------------------------------------------------ marks
def mark_1a(ink, W=400, cy=190):
    cx, u, sw = W / 2, 70.0, 5.0
    wire, hexa = cube_wire(cx, cy, u, ink, sw)
    s = 0.42                                       # gold cube half-size, fraction of the wire cube
    half = 2 * C30 * u * s                         # x half-extent of the gold cube
    ray_in = line((18, cy), (cx - half - 4, cy), ink, sw)
    ray_out = line((cx + half + 4, cy), (W - 18, cy), GOLD, sw)
    dot = f'<circle cx="{W-18}" cy="{cy}" r="{sw*1.5}" fill="{GOLD}"/>'
    src = f'<circle cx="18" cy="{cy}" r="{sw*1.5}" fill="{ink}"/>'
    return "\n".join([wire, ray_in, ray_out, gold_cube(cx, cy, u, s), src, dot])


def mark_icon(ink, W=400, cy=200):
    cx, u, sw = W / 2, 84.0, 13.0
    wire, hexa = cube_wire(cx, cy, u, ink, sw, inner_op=0.5)
    return "\n".join([wire, gold_cube(cx, cy, u, 0.52)])


def mark_1b(ink, W=400, cy=200):
    cx, u, sw = W / 2, 70.0, 6.0
    wire, hexa = cube_wire(cx, cy, u, ink, sw, inner_op=0.45)
    xl, xr = hexa[2][0], hexa[0][0]
    ray_in = line((18, cy), (xl, cy), ink, sw)
    ray_out = line((xl, cy), (W - 18, cy), GOLD, sw + 1.5)
    dot = f'<circle cx="{W-18}" cy="{cy}" r="{sw*1.6}" fill="{GOLD}"/>'
    src = f'<circle cx="18" cy="{cy}" r="{sw*1.6}" fill="{ink}"/>'
    return "\n".join([wire, ray_in, ray_out, src, dot])


def mark_4(ink, bg, W=400, cy=200):
    cx, u, sw = W / 2, 62.0, 4.5
    out = []
    zs = [-1.0, 0.0, 1.0]
    for z in zs:                                  # bottom slice first
        yc = cy - z * 1.15 * u
        pts = [proj(x, y, 0, cx, yc, u) for x, y in [(1.15, -1.15), (-1.15, -1.15), (-1.15, 1.15), (1.15, 1.15)]]
        out.append(poly(pts, fill=bg, fill_opacity=0.82, stroke=ink, stroke_width=sw, stroke_linejoin="round"))
        out.append(f'<circle cx="{cx}" cy="{yc:.1f}" r="{sw*1.9}" fill="{GOLD}"/>')
    top = cy - 1.15 * u
    bot = cy + 1.15 * u
    out.append(line((cx, top - 38), (cx, bot + 38), GOLD, sw))
    # re-draw the dots over the line
    for z in zs:
        out.append(f'<circle cx="{cx}" cy="{cy - z*1.15*u:.1f}" r="{sw*1.9}" fill="{GOLD}"/>')
    return "\n".join(out)


# ------------------------------------------------------------------ wordmark (monoline, matches stroke)
def wordmark(ink, H=40.0, sw=6.5, gap=16.0):
    parts, x = [], 0.0

    def stroke(d):
        parts.append(f'<path d="{d}" fill="none" stroke="{ink}" stroke-width="{sw}" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')

    w = 46                                                           # M
    stroke(f"M{x},{H} L{x},0 L{x+w/2},{H*0.62} L{x+w},0 L{x+w},{H}")
    x += w + gap
    stroke(f"M{x},0 L{x},{H}")                                       # I
    x += gap
    r = H / 2                                                        # D
    stroke(f"M{x},0 L{x},{H} M{x},0 L{x+8},0 A{r},{r} 0 0 1 {x+8},{H} L{x},{H}")
    x += 8 + r + gap * 0.5
    w = 44                                                           # A
    stroke(f"M{x},{H} L{x+w/2},0 L{x+w},{H} M{x+w*0.2},{H*0.68} L{x+w*0.8},{H*0.68}")
    x += w + gap * 0.75
    rr, rx = H / 4, H / 4 * 1.25                                     # S
    cxs = x + rx
    a0, a1 = math.radians(-35), math.radians(145)
    sx, sy = cxs + rx * math.cos(a0), rr + rr * math.sin(a0)
    ex, ey = cxs + rx * math.cos(a1), 3 * rr + rr * math.sin(a1)
    stroke(f"M{sx:.2f},{sy:.2f} A{rx},{rr} 0 1 0 {cxs},{2*rr} A{rx},{rr} 0 1 1 {ex:.2f},{ey:.2f}")
    x += 2 * rx
    return "\n".join(parts), x, H


# ------------------------------------------------------------------ assembly
def doc(w, h, body, bg=None):
    rect = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">\n{rect}\n{body}\n</svg>\n')


def stacked(mark_fn, ink, bg_for_fill, bg=None):
    wm, ww, wh = wordmark(ink)
    if mark_fn is mark_icon:
        return doc(400, 400, mark_fn(ink), bg)
    body = (f'<g>{mark_fn(ink, bg_for_fill) if mark_fn is mark_4 else mark_fn(ink)}</g>'
            f'<g transform="translate({200-ww/2:.1f},372)">{wm}</g>')
    return doc(400, 432, body, bg)


def main():
    import cairosvg
    variants = {"1a": mark_1a, "icon": mark_icon, "1b": mark_1b, "4": mark_4}
    names = []
    for tag, ink in (("ink-dark", INK_L), ("ink-light", INK_D)):   # transparent masters
        svg = stacked(mark_1a, ink, None, None)
        open(os.path.join(OUT, f"master_1a_{tag}.svg"), "w").write(svg)
        svg = doc(400, 400, mark_icon(ink), None)
        open(os.path.join(OUT, f"master_icon_{tag}.svg"), "w").write(svg)
    for k, fn in variants.items():
        for tag, ink, bgc in (("dark", INK_D, BG_D), ("light", INK_L, BG_L)):
            name = f"{k}_{tag}"
            svg = stacked(fn, ink, bgc, bgc)
            with open(os.path.join(OUT, name + ".svg"), "w") as fh:
                fh.write(svg)
            cairosvg.svg2png(bytestring=svg.encode(), write_to=os.path.join(OUT, name + ".png"),
                             output_width=800)
            names.append(name)
    # small-size check: icon-only at 32 and 16 px on both backgrounds, upscaled nearest for viewing
    for k, fn in variants.items():
        for tag, ink, bgc in (("dark", INK_D, BG_D), ("light", INK_L, BG_L)):
            body = fn(ink, bgc) if fn is mark_4 else fn(ink)
            svg = doc(400, 400, body, bgc)
            for px in (64, 32, 16):
                cairosvg.svg2png(bytestring=svg.encode(),
                                 write_to=os.path.join(OUT, f"small_{k}_{tag}_{px}.png"),
                                 output_width=px, output_height=px)
    print("wrote", names)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MIDAS full lockup (README / slides): beam -> grained cube -> gold grain diffracts -> detector rings.

Reuses the primitives of make_logo.py. The simple icon (cube + gold cube) stays the
small-size / sticker mark; this one is the descriptive version.
Run: /Users/travel_user/miniforge3/bin/python make_full.py
"""
import math
import os
import random

import make_logo as ml
from make_logo import GOLD, INK_D, INK_L, BG_D, BG_L, C30, proj, poly, line, doc, wordmark

OUT = os.path.dirname(os.path.abspath(__file__))


def clip(polyg, a, b, c):
    out = []
    for i in range(len(polyg)):
        p, q = polyg[i], polyg[(i + 1) % len(polyg)]
        dp, dq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if dp <= 0:
            out.append(p)
        if dp * dq < 0:
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def voronoi_square(sites):
    cells = []
    for i, s in enumerate(sites):
        pg = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for j, t in enumerate(sites):
            if i != j:
                pg = clip(pg, t[0] - s[0], t[1] - s[1],
                          (t[0] ** 2 + t[1] ** 2 - s[0] ** 2 - s[1] ** 2) / 2)
        cells.append(pg)
    return cells


def face_sites(rng, n, forced=None):
    sites = [forced] if forced else []
    while len(sites) < n:
        p = (rng.uniform(-0.95, 0.95), rng.uniform(-0.95, 0.95))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 0.55 for q in sites):
            sites.append(p)
    return sites


def full_mark(ink, cx=250, cy=190, u=70.0, seed=3):
    rng = random.Random(seed)
    sw = 5.0
    out = []
    # face maps: (a,b) in [-1,1]^2 -> 3-D point on each visible face
    faces = {
        "left": lambda a, b: (a, 1, b),     # y = +1 face; ray hits it at (0, 1, 0.5)
        "right": lambda a, b: (1, a, b),    # x = +1 face
        "top": lambda a, b: (a, b, 1),      # z = +1 face
    }
    gold_seed = (0.0, 0.5)                  # on the left face -> screen point (cx-C30*u, cy)
    O = proj(0, 1, 0.5, cx, cy, u)
    assert abs(O[1] - cy) < 1e-6
    for name, f in faces.items():
        sites = face_sites(rng, 4, gold_seed if name == "left" else None)
        for i, cell in enumerate(voronoi_square(sites)):
            pts = [proj(*f(a, b), cx, cy, u) for a, b in cell]
            if name == "left" and i == 0:
                out.append(poly(pts, fill=GOLD))
            else:
                out.append(poly(pts, fill="none", stroke=ink, stroke_width=sw * 0.45,
                                stroke_opacity=0.55, stroke_linejoin="round"))
    # cube silhouette + the three visible edges
    hexa = [proj(*v, cx, cy, u) for v in
            [(1, -1, 1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)]]
    ctr = proj(1, 1, 1, cx, cy, u)
    for v in [(1, -1, 1), (-1, 1, 1), (1, 1, -1)]:
        out.append(line(ctr, proj(*v, cx, cy, u), ink, sw, 0.9))
    out.append(poly(hexa, fill="none", stroke=ink, stroke_width=sw, stroke_linejoin="round"))
    # incident beam, direct beam, beamstop
    xd = 640
    out.append(line((22, cy), (O[0], cy), ink, sw))
    out.append(f'<circle cx="22" cy="{cy}" r="{sw*1.5}" fill="{ink}"/>')
    out.append(line((O[0] + 6, cy), (xd, cy), ink, sw * 0.5, 0.45))
    # detector: oblique concentric ellipses (Debye rings), spots on them
    k = 0.46
    for ry in (62, 104):
        out.append(f'<ellipse cx="{xd}" cy="{cy}" rx="{ry*k:.1f}" ry="{ry}" fill="none" '
                   f'stroke="{ink}" stroke-opacity="0.5" stroke-width="{sw*0.55}"/>')
    out.append(f'<circle cx="{xd}" cy="{cy}" r="{sw*1.2}" fill="{ink}"/>')
    pt = lambda ry, t: (xd + ry * k * math.cos(math.radians(t)), cy + ry * math.sin(math.radians(t)))
    gold_hits = [(62, -62), (62, 118), (104, -74), (104, 104)]
    for ry, t in gold_hits:                  # diffracted beams from the gold grain
        x, y = pt(ry, t)
        out.append(line(O, (x, y), GOLD, sw * 0.55, 0.9))
    for ry, t in gold_hits:
        x, y = pt(ry, t)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{sw*1.35}" fill="{GOLD}"/>')
    for ry, t in [(62, 30), (62, -128), (62, 160), (104, -28), (104, 32), (104, -160), (104, 146)]:
        x, y = pt(ry, t)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{sw*0.8}" fill="{ink}" fill-opacity="0.85"/>')
    return "\n".join(out)


FONT = "/Users/travel_user/miniforge3/fonts/IBMPlexSans-Text.ttf"   # IBM Plex Sans, SIL OFL


def text_path(text, size, tracking, ink, opacity=1.0):
    """Outline `text` (no kerning) into an SVG <path>; returns (svg, width_px)."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont
    f = TTFont(FONT)
    gs, cmap, upm = f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm
    sc = size / upm
    d, x = [], 0.0
    for ch in text:
        g = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[g].draw(TransformPen(pen, (sc, 0, 0, -sc, x, 0)))
        d.append(pen.getCommands())
        x += gs[g].width * sc + tracking
    x -= tracking
    return f'<path d="{" ".join(d)}" fill="{ink}" fill-opacity="{opacity}"/>', x


def lockup_body(ink, tagline=True, W=760):
    """Return (svg fragment, width, height) of the full lockup."""
    wm, ww, wh = wordmark(ink, H=44, sw=7, gap=18)
    body = full_mark(ink) + f'<g transform="translate({W/2-ww/2:.1f},362)">{wm}</g>'
    H = 430
    if tagline:
        tp, tw = text_path(TAGLINE, 17, 1.4, ink, 0.72)
        body += f'<g transform="translate({W/2-tw/2:.1f},448)">{tp}</g>'
        H = 470
    return body, W, H


def lockup(ink, bg=None, tagline=True, W=760):
    body, W, H = lockup_body(ink, tagline, W)
    return doc(W, H, body, bg)


TAGLINE = "Microstructural Imaging using Diffraction Analysis Software"


def main():
    import cairosvg
    for tag, ink, bg in (("dark", INK_D, BG_D), ("light", INK_L, BG_L)):
        svg = lockup(ink, bg)
        open(os.path.join(OUT, f"full_{tag}.svg"), "w").write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=os.path.join(OUT, f"full_{tag}.png"),
                         output_width=1520)
    for tag, ink in (("ink-dark", INK_L), ("ink-light", INK_D)):
        open(os.path.join(OUT, f"master_full_{tag}.svg"), "w").write(lockup(ink, None))
    print("ok")


if __name__ == "__main__":
    main()

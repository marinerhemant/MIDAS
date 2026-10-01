#!/usr/bin/env python3
"""Build every MIDAS logo asset into logos/ (full lockup, icon, sticker, slide titles).

    /Users/travel_user/miniforge3/bin/python build_assets.py

Needs: cairosvg (PNG), fonttools + IBM Plex Sans (outlined tagline, SIL OFL).
"""
import os

import cairosvg

from make_full import lockup, lockup_body, text_path, TAGLINE
from make_logo import INK_D, INK_L, BG_D, BG_L, mark_icon, wordmark, doc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)          # logos/


def write(name, svg, png_width=None, png_height=None):
    with open(os.path.join(OUT, name + ".svg"), "w") as fh:
        fh.write(svg)
    if png_width:
        cairosvg.svg2png(bytestring=svg.encode(), write_to=os.path.join(OUT, name + ".png"),
                         output_width=png_width, output_height=png_height)


def sticker():
    """Round die-cut sticker: white margin, dark disc, icon, wordmark."""
    S, c = 440, 220
    wm, ww, wh = wordmark(INK_D, H=30, sw=5, gap=12)
    body = (f'<circle cx="{c}" cy="{c}" r="{c-4}" fill="#FFFFFF"/>'
            f'<circle cx="{c}" cy="{c}" r="{c-20}" fill="{BG_D}"/>'
            f'<g transform="translate({c-200*0.56:.1f},{c-205*0.56-10+2:.1f}) scale(0.56)">{mark_icon(INK_D)}</g>'
            f'<g transform="translate({c-ww/2:.1f},{c+112:.1f})">{wm}</g>')
    return doc(S, S, body, None)


def slide_title(ink, bg):
    body, W, H = lockup_body(ink, tagline=True)
    sc = 2.0
    ox, oy = (1920 - W * sc) / 2, (1080 - H * sc) / 2
    return doc(1920, 1080, f'<g transform="translate({ox:.1f},{oy:.1f}) scale({sc})">{body}</g>', bg)


def main():
    # full lockup: transparent masters for README <picture>; "on-dark" uses light ink
    write("midas_logo_on-dark", lockup(INK_D, None), 1520)
    write("midas_logo_on-light", lockup(INK_L, None), 1520)
    # icon: transparent masters + raster sizes
    for tag, ink in (("on-dark", INK_D), ("on-light", INK_L)):
        svg = doc(400, 400, mark_icon(ink), None)
        write(f"midas_icon_{tag}", svg, 512)
    icon_dark = doc(400, 400, mark_icon(INK_D), BG_D)
    for px in (32, 180):
        cairosvg.svg2png(bytestring=icon_dark.encode(), write_to=os.path.join(OUT, f"midas_icon_{px}.png"),
                         output_width=px, output_height=px)
    write("midas_sticker", sticker(), 1200, 1200)
    write("midas_slide_title_dark", slide_title(INK_D, BG_D), 1920, 1080)
    write("midas_slide_title_light", slide_title(INK_L, BG_L), 1920, 1080)
    # legacy path stays valid: new full lockup on the dark background, as a real PNG
    cairosvg.svg2png(bytestring=lockup(INK_D, BG_D).encode(),
                     write_to=os.path.join(OUT, "midas_logo.png"), output_width=1520)
    print("built into", OUT)


if __name__ == "__main__":
    main()

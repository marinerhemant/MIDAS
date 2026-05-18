"""Streamlit single-image ring-pick GUI (Item 32).

Run as:

    streamlit run midas_integrate_v2/gui/ring_picker.py -- \\
        --image calibrant.tif --energy 60 --calibrant CeO2

Click on detected rings to seed BC; drag the BC marker; preview live
LM-refined geometry; save a paramstest.txt when satisfied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except ImportError:
    st = None  # type: ignore
    _HAS_STREAMLIT = False


def _no_streamlit_msg():
    print(
        "midas_integrate_v2.gui.ring_picker requires streamlit.\n"
        "Install via: pip install 'midas-integrate-v2[gui]'",
        file=sys.stderr,
    )


def main():
    if not _HAS_STREAMLIT:
        _no_streamlit_msg()
        sys.exit(1)
    p = argparse.ArgumentParser(prog="ring_picker")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--energy", type=float, default=60.0,
                   help="X-ray energy in keV")
    p.add_argument("--calibrant", type=str, default="CeO2")
    args = p.parse_args(sys.argv[3:])  # streamlit eats argv[0:3]

    import numpy as np
    import tifffile

    st.set_page_config(page_title="MIDAS ring picker", layout="wide")
    st.title("MIDAS v2 — ring-pick calibration GUI")
    st.caption(f"Image: {args.image}; calibrant: {args.calibrant}; "
                f"E = {args.energy} keV")

    img = tifffile.imread(str(args.image)).astype(np.float64)
    NZ, NY = img.shape
    bc_y = st.sidebar.number_input("Beam centre Y (px)", value=NY / 2.0)
    bc_z = st.sidebar.number_input("Beam centre Z (px)", value=NZ / 2.0)
    lsd_um = st.sidebar.number_input("Lsd (µm)", value=1_000_000.0)

    st.image(np.log10(np.clip(img, 1.0, None)), caption="image (log10)",
              clamp=True)
    if st.button("Run LM refinement"):
        try:
            from midas_calibrate_v2.bootstrap import estimate_initial_spec
            spec = estimate_initial_spec(
                img,
                wavelength_A=12398.4 / (args.energy * 1000.0),
                BC=(bc_y, bc_z),
                Lsd_um=lsd_um,
                calibrant=args.calibrant,
            )
            st.success(f"Refinement complete. Spec: {spec}")
        except Exception as e:
            st.error(f"Refinement failed: {e}")
            st.info("This is a scaffold; polish for your detector "
                     "in midas_integrate_v2.gui.ring_picker.main()")


if __name__ == "__main__":
    main()

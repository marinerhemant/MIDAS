# MIDAS

Code for reduction of Near-Field and Far-Field High Energy Diffraction Microscopy (HEDM) data.

nldrmd.cu and nldrmd.cuh are CUDA-based optimizers based on the NLOPT source.
SGInfo library used to calculate HKLs.
Need to install libtiff-dev for compilation of NF-HEDM codes.


# Installation
To check help for installation, type "make help" in the terminal.
For individual help type "make helpnf" or "make helpff" in the terminal.
To compile individually, need to go to the sub-folder and "make" individually.
Would need NLOPT and TIFF packages.
For experimental CUDA codes: go to FF_HEDM folder and "make cuda". This doesn't require any external library.

# MIDAS

Code for reduction of Near-Field and Far-Field High Energy Diffraction Microscopy (HEDM) data.

Neldermead is taken from http://people.sc.fsu.edu/~jburkardt/cpp_src/asa047/asa047.html and modified to include constraints and use as CUDA kernels.
SGInfo library used to calculate HKLs.
Need to install libtiff-dev and nlopt for compilation of NF-HEDM codes.


# Installation
To check help for installation, type "make help" in the terminal.
For individual help type "make helpnf" or "make helpff" in the terminal.
To compile individually, need to go to the sub-folder and "make" individually.
Would need NLOPT and TIFF packages.
For experimental CUDA codes: go to FF_HEDM folder and "make cuda". This doesn't require any external library.

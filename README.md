# MIDAS

Code for reduction of Near-Field and Far-Field High Energy Diffraction Microscopy (HEDM) data.

Neldermead is taken from http://people.sc.fsu.edu/~jburkardt/cpp_src/asa047/asa047.html and modified to include constraints and use as CUDA kernels.

SGInfo library used to calculate HKLs.

Downloads swift binary, NLOPT, LIBTIFF, NETCDF (curl, hdf5, zlib) for compilation of N(F)F-HEDM codes.

# Installation
Go to each sub-folder: NF_HEDM / FF_HEDM and type "make MACHINE_NAME". This will install shortcuts in ${HOME}/.MIDAS directory.

MACHINE_NAMEs supported:
local_dep (local deployment)
orthros
lcrc_cloud
biocluster
nersc_edison
purdue_rice
purdue_conte

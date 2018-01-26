# MIDAS

Code for reduction of Near-Field and Far-Field High Energy Diffraction Microscopy (HEDM) data.

Neldermead is taken from http://people.sc.fsu.edu/~jburkardt/cpp_src/asa047/asa047.html and modified to include constraints and use as CUDA kernels.

SGInfo library used to calculate HKLs.

Some misorientation functions are taken from the ODFPF package from Cornell (https://anisotropy.mae.cornell.edu/onr/Matlab/matlab-functions.html).

Downloads swift binary, NLOPT, LIBTIFF, NETCDF (curl, hdf5, zlib) for compilation of N(F)F-HEDM codes.

# Installation
Go to each sub-folder: NF_HEDM / FF_HEDM and type "make MACHINE_NAME". This will install shortcuts in ${HOME}/.MIDAS directory.

MACHINE_NAMEs supported for compiling:
* local
* orthros
* lcrc_cloud
* biocluster
* nersc
* purdue
* lanl
* lcrc
* stampede

MACHINE_NAMEs supported during run:
* local
* orthrosall
* orthrosregular
* orthrosextra
* orthrosnew
* lcrc_cloud
* edison_debug
* edison_realtime
* cori_realtime
* biocluster
* conte
* rice
* halstead
* darwin
* blues_batch
* blues_haswell
* bebop
* stamped_normal
* stampede_largemem

If using local run give nCPUs to use during run.

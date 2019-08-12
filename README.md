# MIDAS

Code for reduction of Near-Field and Far-Field High Energy Diffraction Microscopy (HEDM) data.

Neldermead is taken from http://people.sc.fsu.edu/~jburkardt/cpp_src/asa047/asa047.html and modified to include constraints and use as CUDA kernels.

SGInfo library used to calculate HKLs.

Some misorientation functions are taken from the ODFPF package from Cornell (https://anisotropy.mae.cornell.edu/onr/Matlab/matlab-functions.html).

Downloads jre, swift binary, NLOPT, LIBTIFF, NETCDF (curl, hdf5, zlib) for compilation of N(F)F-HEDM codes.

More details at https://github.com/marinerhemant/MIDAS/wiki

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
* notchpeak

If using local run give nCPUs to use during run.

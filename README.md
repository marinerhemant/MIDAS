# MIDAS

**** V7 Released ****


Code for reduction of High Energy Diffraction Microscopy (HEDM) data developed at Advanced Photon Source.

In case of problems, contatct [Hemant Sharma](mailto:hsharma@anl.gov?subject=[MIDAS]%20From%20Github).

[SGInfo](http://cci.lbl.gov/sginfo/) library used to calculate HKLs.

Some misorientation functions are taken from the [ODFPF](https://anisotropy.mae.cornell.edu/onr/Matlab/matlab-functions.html) package from Cornell.

Downloads [NLOPT](https://nlopt.readthedocs.io/en/latest/), [LIBTIFF](http://www.libtiff.org/), [FFTW](http://www.fftw.org/), [HDF5](https://www.hdfgroup.org/solutions/hdf5/), [BLOSC](https://github.com/Blosc/c-blosc), [BLOSC-2](https://github.com/Blosc/c-blosc2), [ZLIB](https://zlib.net/), [LIBZIP](https://libzip.org/) for compilation of N(F)F-HEDM codes.


**Installation and Compilation Instructions:**

Download the source code from [GitHub](https://github.com/marinerhemant/MIDAS).

`bash
git clone https://github.com/marinerhemant/MIDAS.git
`

Change to the MIDAS directory.

`bash
cd MIDAS
`

Run the build script.

`bash
./build.sh
`

Install the python requirements.

`bash
pip install -r requirements.txt
`

**More details at** [MIDAS-WIKI](https://github.com/marinerhemant/MIDAS/wiki) 

Installation instructions on the WIKI are old. Please refer to the build.sh script for the latest instructions.

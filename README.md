# MIDAS

**** V8 Released ****


Code for reduction of High Energy Diffraction Microscopy (HEDM) data developed at Advanced Photon Source.

In case of problems, contatct [Hemant Sharma](mailto:hsharma@anl.gov?subject=[MIDAS]%20From%20Github).

[SGInfo](http://cci.lbl.gov/sginfo/) library used to calculate HKLs.

Some misorientation functions are taken from the [ODFPF](https://anisotropy.mae.cornell.edu/onr/Matlab/matlab-functions.html) package from Cornell.

Downloads [NLOPT](https://nlopt.readthedocs.io/en/latest/), [LIBTIFF](http://www.libtiff.org/), [FFTW](http://www.fftw.org/), [HDF5](https://www.hdfgroup.org/solutions/hdf5/), [BLOSC](https://github.com/Blosc/c-blosc), [BLOSC-2](https://github.com/Blosc/c-blosc2), [ZLIB](https://zlib.net/), [LIBZIP](https://libzip.org/) for compilation of N(F)F-HEDM codes.


**Installation and Compilation Instructions:**

Download the source code from [GitHub](https://github.com/marinerhemant/MIDAS).

```
git clone https://github.com/marinerhemant/MIDAS.git
```

Change to the MIDAS directory.

```
cd MIDAS
```

**To compile on Linux, skip to step 5.**

**To compile on MACOS, please use the following steps:**

1. Install Homebrew if not already installed.
2. Install the required libraries using Homebrew:
   ```
   brew install llvm libomp gcc cmake jemalloc
   ```
3. Set the environment variables for the compiler:
   ```
    echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
    echo 'export LDFLAGS="-L/opt/homebrew/opt/llvm/lib $LDFLAGS"' >> ~/.zshrc
    echo 'export CPPFLAGS="-I/opt/homebrew/opt/llvm/include $CPPFLAGS"' >> ~/.zshrc
    echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib $LDFLAGS"' >> ~/.zshrc
    echo 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include $CPPFLAGS"' >> ~/.zshrc
    echo 'CC=/opt/homebrew/opt/gcc/bin/gcc-15' >> ~/.zshrc
    echo 'export CC' >> ~/.zshrc
    echo 'CXX=/opt/homebrew/opt/gcc/bin/g++-15' >> ~/.zshrc
    echo 'export CXX' >> ~/.zshrc
   ```
4. Reload your shell configuration:
   ```
   source ~/.zshrc
   ```
5. Run the build script:
   ```
   ./build.sh
   ```
6. Install the Python requirements:
   ```
   pip install -r requirements.txt
   ```

**More details at** [MIDAS-WIKI](https://github.com/marinerhemant/MIDAS/wiki) 

Installation instructions on the WIKI are old. Please refer to the build.sh script for the latest instructions.

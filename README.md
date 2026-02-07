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

**To compile on Windows, please use the following steps after installing wsl UBUNTU:**

```
sudo ./build_wsl_windows.sh
```

This will compile all the codes in the MIDAS directory.

**To compile on LINUX, skip to step 5.**

**To compile on MACOS, please use the following steps:**

1. Install Homebrew if not already installed.
    If you don't have Homebrew installed, you can install it by running the following command in your terminal:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
    Follow the on-screen instructions to complete the installation.
    * In case you don't have SUDO access, you can install Homebrew in your home directory by following these steps:
    ```bash
    mkdir homebrew && curl -L https://github.com/Homebrew/brew/tarball/main | tar xz --strip-components 1 -C homebrew
    eval "$(homebrew/bin/brew shellenv)"
    brew update --force --quiet
    chmod -R go-w "$(brew --prefix)/share/zsh"
   ```
    Also place the Homebrew binary in your PATH by adding the following line to your `~/.zshrc` file:
    ```bash
    echo 'eval $(/opt/homebrew/bin/brew shellenv)'  >> ~/.zshrc
    ```
    Then, reload your shell configuration:
    ```bash
    source ~/.zshrc
    ```
2. Install the required libraries using Homebrew:
   ```
   brew install llvm libomp gcc cmake jemalloc
   ```
3. Set the environment variables for the compiler (copy paste the following commands in your terminal):
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
   To disable CUDA:
   ```
   ./build.sh --cuda OFF
   ```
   * In case of MACOS, if there is an error during build for the libtiff directory, please follow these steps:
   ```
   Edit the cmake/deps/LIBTIFF_custom.cmake file to comment lines 44, 45, and 48 by adding # in the beginning. It should look like:
      #  list(APPEND _libtiff_fetch_extra_args
      #       PATCH_COMMAND patch -p1 -N --fuzz=0 < ${CMAKE_CURRENT_LIST_DIR}/libtiff-disable-doc.patch
         # -N: ignore already applied patches
         # --fuzz=0: apply only if exact match
      #   )
   Edit the build/_deps/libtiff-src/CMakeLists.txt to comment line number 157 by adding # in the beginning. It should look like:
        #add_subdirectory(doc)
   Then run build.sh again. DO NOT DELETE THE build directory after this step!!!!!
   ```
6. Install the Python requirements:
   ```
   conda env create -f environment.yml
   ```

7. Activate the conda environment:
   ```
   conda activate midas_env
   ```

8. To build only a certain code (e.g. PeaksFittingOMPZarrRefactor):
   ```
   cd build # navigate to build directory
   cmake --build . --target PeaksFittingOMPZarrRefactor
   ```

**MANUALS in the [manuals](https://github.com/marinerhemant/MIDAS/tree/master/manuals) directory**

**Somewhat DEPRECATED! More details at** [MIDAS-WIKI](https://github.com/marinerhemant/MIDAS/wiki) 

Installation instructions on the WIKI are old. Please refer to the instructions above for the latest method.

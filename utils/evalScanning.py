from ctypes import *
import numpy as np
import os

home = os.path.expanduser('~')
paramFN = 'ps.txt' # This files contains all the info required to evaluate the function.
scanningFCNs = ctypes.CDLL(home+'/opt/MIDAS/FF_HEDM/bin/ScanningFunctionsSO.so')
scanningFCNs.populate_arrays(paramFN.encode('ASCII'))
# This writes a file /dev/shm/x.bin, containing 9 parameters for each voxel. They are arranged as each voxel position has 3 euler angles, 6 values for lattice parameter.
# The number of (double) values in x.bin is 9*nrVoxels.
# For the euler angles, there is a parameter called OrientTol, which defines how much euler angles can deviate in degrees.
# For the lattice parameter: ABCTol defines how much in % the first 3 parameters are allowed to deviate, ABGTol defines how much the next 3 parameters are allowed to deviate.
# A number of other helper arrays are also written to help in analysis.

# To change x, we can modify the /dev/shm/x.bin file and call the following function, which will return the value of the function calculated as 
function_val = scanningFCNs.evaluateF()

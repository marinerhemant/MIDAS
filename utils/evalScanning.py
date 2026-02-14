import ctypes
import os
import numpy as np

try:
    import midas_config
    scanningFCNs = ctypes.CDLL(os.path.join(midas_config.MIDAS_BIN_DIR, 'ScanningFunctionsSO.so'))
except (ImportError, OSError):
    home = os.path.expanduser('~')
    scanningFCNs = ctypes.CDLL(home+'/opt/MIDAS/FF_HEDM/bin/ScanningFunctionsSO.so')
paramFN = 'ps.txt' # This files contains all the parameters required to evaluate the function.
scanningFCNs.populate_arrays(paramFN.encode('ASCII'))

# This writes a file /dev/shm/x.bin, containing 9 parameters for each voxel. They are arranged as each voxel position has 3 euler angles, 6 values for lattice parameter.
# The number of (double) values in x.bin is 9*nrVoxels. nrVoxels is the number of rows in Grain2Voxels.csv file
# For the euler angles, there is a parameter called OrientTol, which defines how much euler angles can deviate in degrees.
# For the lattice parameter: ABCTol defines how much in % the first 3 parameters are allowed to deviate, ABGTol defines how much the next 3 parameters are allowed to deviate.
# A number of other helper arrays are also written to help in analysis.

# To change x, we can read the /dev/shm/x.bin file and pass the array to our function scanningFCNs.evaluateF to get a function value. The code below shows an example where x is modified multiple times and re-evaluated.

x = np.fromfile('/dev/shm/x.bin',dtype=np.double,count=(3511*9))
scanningFCNs.evaluateF.argtypes = [ctypes.POINTER(ctypes.c_double)]
scanningFCNs.evaluateF.restype = ctypes.c_double

function_val = scanningFCNs.evaluateF(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
print(function_val)

for voxNr in range(3511):
	x[voxNr*9+0] *= 1.001
	x[voxNr*9+1] *= 1.001
function_val = scanningFCNs.evaluateF(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
print(function_val)

for voxNr in range(3511):
	x[voxNr*9+0] *= 1.001
	x[voxNr*9+1] *= 1.001
function_val = scanningFCNs.evaluateF(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
print(function_val)

for voxNr in range(3511):
	x[voxNr*9+0] *= 1.001
	x[voxNr*9+1] *= 1.001
function_val = scanningFCNs.evaluateF(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
print(function_val)

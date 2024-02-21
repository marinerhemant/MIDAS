'''
# Ang file has some header starting with #, then data in EulerAngles (0,1,2), x,y, confidence, phaseNr
# Mic file needs header starting with % TriEdgeSize (can be 0), NumPhases (1), GlobalPosition (can be 0) and the descriptor:
# OrientationRowNr	OrientationID	RunTime	X	Y	TriEdgeSize	UpDown	Eul1	Eul2	Eul3	Confidence	PhaseNr
# OrientationRowNr is the FF grainID, but we will give it a random number.
# We can put first 3 as random numbers, TriEdgeSize as 1, UpDown as +-1 based on rowNr
'''
import pandas as pd
import numpy as np

fn = '/Users/hsharma/Desktop/analysis/kenesei_nov20/ebsd/ebsd2hedm_aligned/ebsd_inv_320.ang'
fn = '/Users/hsharma/Desktop/analysis/kenesei_nov20/ebsd/GES_HEDM1_resampledEBSD_layer1.ang'
outfn = '/Users/hsharma/Desktop/analysis/kenesei_nov20/ebsd/ebsd2hedm_aligned/mic/GES_HEDM1_resampledEBSD_layer1.ang.mic'
data = np.genfromtxt(fn,comments='#')
print(data.shape)
data2 = np.zeros((data.shape[0],12))
data2[:,3:5] = data[:,3:5]
data2[:,7:10] = data[:,:3]
# data2[:,10:] = data[:,5:]
data2[:,10:] = data[:,6:]
data2[:,5] = 1
data2[0::2,6] = 1
data2[1::2,6] = -1
header = """%TriEdgeSize 0.000000
%NumPhases 1
%GlobalPosition 0.000000
%OrientationRowNr	OrientationID	RunTime	X	Y	TriEdgeSize	UpDown	Eul1	Eul2	Eul3	Confidence	PhaseNr
"""
outf = open(outfn,'w')
outf.write(header)
np.savetxt(outf,data2,fmt="%.6f",delimiter="\t")
print(data2[5000,:])
print(data2[5001,:])

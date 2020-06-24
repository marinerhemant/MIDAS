import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
import h5py
import glob
from scipy.spatial.transform import Rotation as R

# Read file
fn = 'MicrostructureWithoutTwins_CPFE_Results_ElementWise_binary.vtk'
reader = vtk.vtkGenericDataObjectReader()
reader.SetFileName(fn)
reader.Update()
dataset = dsa.WrapDataObject(reader.GetOutput())

Positions = dataset.CellData['Centroid']
IDsToKeep = (dataset.CellData['MeshQuality-Flag'] == 1) & (Positions[:,0] > 15) & (Positions[:,0] < 285) & (Positions[:,1] > 15) & (Positions[:,1] < 285) & (Positions[:,2] > 15) & (Positions[:,2] < 285)
grainIDs = dataset.CellData['GrainIDs'][IDsToKeep]
PSA = dataset.CellData['PlasticStrainAccumulation-Cycle-4-OutputStep-20'][IDsToKeep]
PSED = dataset.CellData['PlasticStrainEnergyDensity-Cycle-4-OutputStep-20'][IDsToKeep]
nGrains = np.max(grainIDs) + 1
numCells, = PSED.shape
Positions = dataset.CellData['Centroid'][IDsToKeep]
Strains = dataset.CellData['ElasticStrainTensor-Cycle-4-OutputStep-20'][IDsToKeep]
OMs = dataset.CellData['OrientationTensor-Cycle-4-OutputStep-20'][IDsToKeep]
OM2 = dataset.CellData['OrientationTensor-Cycle-1-OutputStep-1'][IDsToKeep]

for grainNr in range(1,nGrains):
	print([grainNr,np.max(PSED[grainIDs==grainNr]),np.mean(PSED[grainIDs==grainNr]),np.min(PSED[grainIDs==grainNr]),np.std(PSED[grainIDs==grainNr])])#OMs[np.argmax(PSED[grainIDs==grainNr])]])

import matplotlib.pyplot as plt
maxIDX = np.argmax(PSED)
orig_orient = R.from_matrix(OM2[maxIDX]).as_euler('zxz',degrees=True)
maxPSEDs = np.zeros(80)
nrs = np.zeros(80)
for cycle in range(1,5):
	for step in range(1,21):
		thisKey = 'PlasticStrainEnergyDensity-Cycle-' + str(cycle) + '-OutputStep-' + str(step)
		maxPSEDs[(cycle-1)*20+step-1] = dataset.CellData[thisKey][IDsToKeep][maxIDX]
		thisKey2 = 'OrientationTensor-Cycle-' + str(cycle) + '-OutputStep-' + str(step)
		new_orient = R.from_matrix(dataset.CellData[thisKey2][IDsToKeep][maxIDX]).as_euler('zxz',degrees=True)
		miso = new_orient - orig_orient
		print (miso)
		nrs[(cycle-1)*20+step-1] = (cycle-1)*20+step-1
plt.scatter(maxPSEDs,nrs)
plt.show()

# Let's now take the last output and create a bin file, write out pos, orient, strain(rotated to crystal)
# ~ fnout = 'MIDAS_Input_OrigOrientOrigStrain.bin'
fnout = 'MIDAS_Input_Cycle4OutputStep20.bin'
fout = open(fnout,'w')
outarr = np.zeros((numCells,18))
for ctr in range(numCells):
	# ~ OMThis = R.from_matrix(OM2[ctr]).as_matrix()
	OMThis = R.from_matrix(OMs[ctr]).as_matrix() # comment if wanted original orientations
	StrainThis = Strains[ctr] # comment this and next few lines if wanted zero strains
	StrainsThis = np.array([[StrainThis[0],StrainThis[1],StrainThis[2]],[StrainThis[1],StrainThis[3],StrainThis[4]],[StrainThis[2],StrainThis[4],StrainThis[5]]])
	StrainRotated = np.matmul(np.matmul(OMThis,StrainsThis),OMThis.T)
	outarr[ctr][12] = StrainRotated[0,0]
	outarr[ctr][13] = StrainRotated[0,1]
	outarr[ctr][14] = StrainRotated[0,2]
	outarr[ctr][15] = StrainRotated[1,1]
	outarr[ctr][16] = StrainRotated[1,2]
	outarr[ctr][17] = StrainRotated[2,2]
	outarr[ctr][0] = Positions[ctr,0]
	outarr[ctr][1] = Positions[ctr,1]
	outarr[ctr][2] = Positions[ctr,2]
	outarr[ctr][3] = OMThis[0,0]
	outarr[ctr][4] = OMThis[0,1]
	outarr[ctr][5] = OMThis[0,2]
	outarr[ctr][6] = OMThis[1,0]
	outarr[ctr][7] = OMThis[1,1]
	outarr[ctr][8] = OMThis[1,2]
	outarr[ctr][9] = OMThis[2,0]
	outarr[ctr][10] = OMThis[2,1]
	outarr[ctr][11] = OMThis[2,2]

outarr.astype(np.float64).tofile(fout)
fout.close()

# HDF file
outfn = h5py.File(fn.replace('vtk','h5'),'w')
for key in dataset.CellData.keys():
	thisDataset = outfn.create_dataset(key,data=np.array(dataset.CellData[key]).astype(np.float32),compression='gzip')
outfn.close()

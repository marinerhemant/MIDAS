import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
import h5py
import glob


fn = 'MicrostructureWithoutTwins_CPFE_Results_ElementWise_binary.vtk'

reader = vtk.vtkGenericDataObjectReader()
reader.SetFileName(fn)
reader.Update()
dataset = dsa.WrapDataObject(reader.GetOutput())
outfn = h5py.File(fn.replace('vtk','h5'),'w')
for key in dataset.CellData.keys():
	thisDataset = outfn.create_dataset(key,data=np.array(dataset.CellData[key]).astype(np.float32),compression='gzip')
outfn.close()

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

# Let's now take the last output and create a bin file, write out pos, orient, strain
fnout = 'MIDAS_Input_Cycle4OutputStep20.bin'
fout = open(fnout,'w')
outarr = np.zeros((numCells,18))
for ctr in range(numCells):
	outarr[ctr][0] = Positions[ctr,0]
	outarr[ctr][1] = Positions[ctr,1]
	outarr[ctr][2] = Positions[ctr,2]
	outarr[ctr][3] = OMs[ctr,0,0]
	outarr[ctr][4] = OMs[ctr,0,1]
	outarr[ctr][5] = OMs[ctr,0,2]
	outarr[ctr][6] = OMs[ctr,1,0]
	outarr[ctr][7] = OMs[ctr,1,1]
	outarr[ctr][8] = OMs[ctr,1,2]
	outarr[ctr][9] = OMs[ctr,2,0]
	outarr[ctr][10] = OMs[ctr,2,1]
	outarr[ctr][11] = OMs[ctr,2,2]
	outarr[ctr][12] = Strains[ctr,0]
	outarr[ctr][13] = Strains[ctr,1]
	outarr[ctr][14] = Strains[ctr,2]
	outarr[ctr][15] = Strains[ctr,3]
	outarr[ctr][16] = Strains[ctr,4]
	outarr[ctr][17] = Strains[ctr,5]

outarr.astype(np.float64).tofile(fout)

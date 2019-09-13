# Code to take tiffs and convert to hdf for nf

############# Parameters #####################
startNr = 244 # First File Nr for the tiff data
nDistances = 3
Lsd = [5000,8000,11000] # Microns
LatC = [4.08,4.08,4.08,90,90,90] # Angstorm and Degrees
SpaceGroup = 225 
energy = 61.332 # keV
omegaStart = 90 # Degrees, for Aero change direction
nrFilesPerDistance = 30 # Look up in parfile
omegaStep = -0.25 # Degrees, change direction for Aero
nrPixels = 2048 # should be divisible by 4 and 8
samx = -0.1 # Microns
samy = 592.3 # Microns
samz = 0.1 # Microns
px = 1.48 # Microns
folder = '/mnt/chromeos/removable/UNTITLED/NFData/Au_nf/'
fstem = 'Au_nf'
outfn = '/home/hemantmariner/Desktop/preuss_nov18/Dataset.hdf'
LoGMaskRadius = 15 # Pixels
BlanketSubtraction = 20 # Counts
DoLoGFilter = 1 # Switch, 0 or 1
GaussFiltRadius = 3.0 # Pixels, can be double use either 3 or 4, this is the radius for LoG filter
WriteReducedImage = 1 # Switch 0 or 1
MedianFilterRadius = 2 # Int only 1 or 2
NrOfDeblurIterations = 0 # Typically 100 or 0 to disable

###############################################

from PIL import Image
import matplotlib
import math
import sys, os
import numpy as np
import h5py
import time


f = h5py.File(outfn,'w')
f.create_dataset('implements',data='exchange')
f.create_group('exchange')
f.create_group('measurement')
f['measurement'].create_group('instrument')
f['measurement'].create_dataset('experiment_type',data=str('near field'))
f['measurement']['instrument'].create_group('monochr')
f['measurement']['instrument']['monochr'].create_dataset('energy',data=str(energy))
f['measurement']['instrument']['monochr']['energy'].attrs['units']='keV'
f['measurement']['instrument']['monochr'].create_dataset('wavelength',data=str(12.398/energy))
f['measurement']['instrument']['monochr']['wavelength'].attrs['units']='Angstrom'
omegaEnd = omegaStart + (nrFilesPerDistance)*omegaStep
omegas_t = np.arange(omegaStart,omegaEnd,omegaStep)
omegas = []
for nr in range(nDistances):
	omegas.append(omegas_t)
f['measurement']['instrument'].create_group('detector')
f['measurement']['instrument']['detector'].create_group('geometry')
f['measurement']['instrument']['detector']['geometry'].create_dataset('translation',data=Lsd)
f['measurement']['instrument']['detector']['geometry']['translation'].attrs['units']='microns'
f['measurement']['instrument']['detector']['geometry'].create_dataset('nr_pixels',data=nrPixels)
f['measurement']['instrument']['detector']['geometry'].create_dataset('pixel_size',data=px)
f['measurement']['instrument']['detector']['geometry']['pixel_size'].attrs['units']='microns'
f['exchange'].create_group('data')
f['exchange']['data'].attrs['data_units'] = '16 bit'
f['measurement'].create_dataset('experiment_name',data='xzhang_apr17')
f['measurement'].create_group('sample')
f['measurement']['sample'].create_group('geometry')
f['measurement']['sample']['geometry'].create_dataset('omegas',data=omegas)
f['measurement']['sample']['geometry']['omegas'].attrs['units']='degrees'
f['measurement']['sample'].create_group('crystal')
f['measurement']['sample']['crystal'].create_dataset('lattice_parameter',data=LatC)
f['measurement']['sample']['crystal']['lattice_parameter'].attrs['units']='Angstorm Degrees'
f['measurement']['sample']['crystal'].create_dataset('space_group',data=SpaceGroup)
f['measurement']['sample'].create_dataset('sample_name',data=fstem)
f['measurement']['sample'].create_group('translation')
f['measurement']['sample']['translation'].create_dataset('samx',data=samx)
f['measurement']['sample']['translation']['samx'].attrs['units']='microns'
f['measurement']['sample']['translation'].create_dataset('samy',data=samy)
f['measurement']['sample']['translation']['samy'].attrs['units']='microns'
f['measurement']['sample']['translation'].create_dataset('samz',data=samz)
f['measurement']['sample']['translation']['samz'].attrs['units']='microns'
f['measurement'].create_dataset('nr_files_per_distance',data=nrFilesPerDistance)
f['measurement'].create_dataset('nr_distances',data=nDistances)
f.create_group('analysis')
f['analysis'].create_group('median_images')
f['analysis'].create_group('max_images')
f['analysis'].create_group('max_median_images')
f.create_group('parameters')
f['parameters'].create_dataset('LoGMaskRadius',data=LoGMaskRadius)
f['parameters'].create_dataset('BlanketSubtraction',data=BlanketSubtraction)
f['parameters'].create_dataset('DoLoGFilter',data=DoLoGFilter)
f['parameters'].create_dataset('GaussFiltRadius',data=GaussFiltRadius)
f['parameters'].create_dataset('MedianFilterRadius',data=MedianFilterRadius)
f['parameters'].create_dataset('NrOfDeblurIterations',data=NrOfDeblurIterations)
f['analysis'].create_group('reduced_images')
f['analysis'].create_group('reduced_positions')
f['analysis']['reduced_positions'].create_group('y')
f['analysis']['reduced_positions'].create_group('z')
f['analysis']['reduced_positions'].create_group('peak_id')
f['analysis']['reduced_positions'].create_group('intensity')
zeroArrInit = []
for nr in range(nrFilesPerDistance):
	zeroArrInit.append(0)
zeroArr = []
for nr in range(nDistances):
	zeroArr.append(zeroArrInit)
zeroArr = np.array(zeroArr)
f['analysis']['reduced_positions'].create_dataset('num_pixels_intensity',data=zeroArr)

totNrFiles = nrFilesPerDistance*nDistances
sizeObsSpots = nrPixels*nrPixels/32
f['analysis'].create_dataset('reduced_file',(sizeObsSpots,totNrFiles),chunks=(sizeObsSpots,1),compression='gzip',compression_opts=4,dtype=np.int32)

frameNr = 0
fileNr = startNr
for dist in range(0,nDistances):
	for omega in omegas_t:
		fn = fstem + '_' + str(fileNr).zfill(6) + '.tif'
		im = Image.open(folder+fn)
		imarr = np.array(im,dtype=np.uint16)
		f['exchange']['data'].create_dataset(str(frameNr),data=imarr,compression='gzip',chunks=(nrPixels/8,nrPixels/8))
		f['analysis']['reduced_images'].create_dataset(str(frameNr),(nrPixels,nrPixels),dtype=np.uint16,compression='gzip',chunks=(nrPixels/8,nrPixels/8),compression_opts=4)
		print 'Read file: ' + fn
		frameNr += 1
		fileNr += 1

f.close()

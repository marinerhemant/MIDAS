import zarr
import skimage
import cc3d
import numpy as np
from numba import jit
import argparse, sys, os
import time

t0 = time.time()
@jit(nopython=True)
def applymask(data,mask):
    data2 = np.zeros(data.shape,dtype=np.uint16)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if mask[0,j,k] == 0:
                    data2[i,j,k] = 1
    return data2

def calcREta4mYZ(y,z):
    r = np.sqrt(z*z+y*y)
    eta = np.arccos(z/r)*180/np.pi
    if (y>0):
        eta -= 1
    return r,eta

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(description='''blobPeaksearch.py Code to run "dumb" peaksearch on zip files. Can use large amounts of RAM''', 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-dataFN', type=str, required=True, help='Zarr zip file with the data.')
parser.add_argument('-resultFolder', type=str, required=False, default='.', help='Folder where the zip file exists and results will exist.')
args, unparsed = parser.parse_known_args()
dataFN = args.dataFN
resultFolder = args.resultFolder

if resultFolder == '.':
    resultFolder = os.getcwd()

zf = zarr.open(f'{resultFolder}/{dataFN}','r')
data = zf['exchange/data'][:]
mask = zf['exchange/mask'][:]

data2 = applymask(data,mask)

labels_out = cc3d.connected_components(data2)
props = skimage.measure.regionprops(labels_out,intensity_image=data)
ome_start = zf['measurement/process/scan_parameters/start'][0]
ome_step = zf['measurement/process/scan_parameters/step'][0]
yCen = zf['analysis/process/analysis_parameters/YCen'][0]
zCen = zf['analysis/process/analysis_parameters/ZCen'][0]

iter = 1
outarr = np.empty((len(props),14))
for prop in props:
    if prop.num_pixels > 2:
        thisOme = ome_start + ome_step*prop.weighted_centroid[0]
        thisY = prop.weighted_centroid[2]
        thisZ = prop.weighted_centroid[1]
        integratedIntensity = np.sum(prop.image)
        iMax = prop.intensity_max
        minOme = ome_start + ome_step*prop.bbox[0]
        maxOme = ome_start + ome_step*prop.bbox[3]
        sigR = 1
        sigEta = 1
        nPx = prop.num_pixels
        nPxTot = nPx
        r,eta = calcREta4mYZ(-thisY+yCen,thisZ-zCen)
        outarr[iter-1] = np.array([iter,integratedIntensity,thisOme,thisY,thisZ,iMax,minOme,maxOme,sigR,sigEta,nPx,nPxTot,r,eta])
        iter +=1

outarr = outarr[:iter-1]

np.savetxt(f'{resultFolder}/Result_StartNr_1_EndNr_{data.shape[0]}.csv',outarr,fmt='%0.6f',delimiter=' ',
           header='SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px) IMax MinOme(degrees) MaxOme(degress) SigmaR SigmaEta NrPx NrPxTot Radius(px) Eta(px)')

print(f'Time elapsed: {time.time()-t0}')
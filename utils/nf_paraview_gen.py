#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

##
# HDF5 (compressed output) file arrangement:
#            Group            Dataset
#		|--- GrainID       -- GrainNr            -- Nonsense value: -15, always uint
#		|--- EulerAngles  |-- EulerAngle1        -- Nonsense value: -15, range +-2pi
#		|                 |-- EulerAngle2        -- Nonsense value: -15, range +-2pi
#		|                 |-- EulerAngle3        -- Nonsense value: -15, range +-2pi
#		|--- Confidence    -- ConfidenceValue    -- Nonsense value: -15, range 0...1
#		|--- PhaseNumber   -- PhaseNr            -- Nonsense value: -15, range 1...n
##

fillVal = -15 # this value can be used to filter out nonsense values.

### Only modify the following arguments:
### Also look at where variable FileName is defined to see if the file arrangement is different
####
sampleName = 'ss709_AR1_nf2_R1'
filestem = 'MicrostructureText_Layer'
outfn = 'MicOut'
spaceGroup = 225 # This is used for misorientation calculation
startnr = 1
endnr = 10
minConfidence = 0.1
zspacing = -2
xyspacing = 2  # X and Y spacing are equal
xExtent = 1200 # Maximum Extent of xValues in um
			   # (this should be a bit larger than your sample diameter or edge length)
yExtent = 1200 # Maximum Extent of yValues in um
			   # (this should be a bit larger than your sample diameter or edge length)
####

import math
import sys, os
import numpy as np
import h5py
import time
from numba import jit
from math import cos, sin, tan, sqrt, asin, acos, atan
rad2deg = 57.2957795130823

Dims = np.array([0,0,0])
Dims = Dims.astype(int)
Dims[1] = int(xExtent/abs(xyspacing))
Dims[2] = int(yExtent/abs(xyspacing))
Dims[0] = (endnr - startnr + 1)
print('Dimensions of final array:')
print(Dims)
startPos = 0
grainIDs = np.zeros((Dims))
Euler1 = np.zeros((Dims))
Euler2 = np.zeros((Dims))
Euler3 = np.zeros((Dims))
Confidence = np.zeros((Dims))
PhaseNr = np.zeros((Dims))
dataNr = 0;
outarr = np.zeros((Dims[1],Dims[2],7))
outarr = outarr.astype(float)
dimarr = np.array([Dims[1],Dims[2],abs(xyspacing)])
dimarr = dimarr.astype(int)

TricSym = np.array([[1.00000,   0.00000,   0.00000,   0.00000],[1.00000,   0.00000,   0.00000,   0.00000]])

MonoSym = np.array([[1.00000,   0.00000,   0.00000,   0.00000],[0.00000,   1.00000,   0.00000,   0.00000]])

OrtSym = np.array([[1.00000,   0.00000,   0.00000,   0.00000],[1.00000,   1.00000,   0.00000,   0.00000],[0.00000,   0.00000,   1.00000,   0.00000],[0.00000,   0.00000,   0.00000,   1.00000]])

TetSym = np.array([[1.00000,   0.00000,   0.00000,   0.00000],[0.70711,   0.00000,   0.00000,   0.70711],[0.00000,   0.00000,   0.00000,   1.00000],[0.70711,  -0.00000,  -0.00000,  -0.70711],[0.00000,   1.00000,   0.00000,   0.00000],[0.00000,   0.00000,   1.00000,   0.00000],[0.00000,   0.70711,   0.70711,   0.00000],[0.00000,  -0.70711,   0.70711,   0.00000]])

TrigSym = np.array([[1.00000,   0.00000,   0.00000,   0.00000],[0.50000,   0.00000,   0.00000,   0.86603],[0.50000,  -0.00000,  -0.00000,  -0.86603],[0.00000,   0.50000,  -0.86603,   0.00000],[0.00000,   1.00000,   0.00000,   0.00000],[0.00000,   0.50000,   0.86603,   0.00000]])

HexSym = np.array([[1.00000,   0.00000,   0.00000,   0.00000],[0.86603,   0.00000,   0.00000,   0.50000],[0.50000,   0.00000,   0.00000,   0.86603],[0.00000,   0.00000,   0.00000,   1.00000],[0.50000,  -0.00000,  -0.00000,  -0.86603],[0.86603,  -0.00000,  -0.00000,  -0.50000],[0.00000,   1.00000,   0.00000,   0.00000],[0.00000,   0.86603,   0.50000,   0.00000],[0.00000,   0.50000,   0.86603,   0.00000],[0.00000,   0.00000,   1.00000,   0.00000],[0.00000,  -0.50000,   0.86603,   0.00000],[0.00000,  -0.86603,   0.50000,   0.00000]])

CubSym = np.array([[1.00000,   0.00000,   0.00000,   0.00000],[0.70711,   0.70711,   0.00000,   0.00000],[0.00000,   1.00000,   0.00000,   0.00000],[0.70711,  -0.70711,   0.00000,   0.00000],[0.70711,   0.00000,   0.70711,   0.00000],[0.00000,   0.00000,   1.00000,   0.00000],[0.70711,   0.00000,  -0.70711,   0.00000],[0.70711,   0.00000,   0.00000,   0.70711],[0.00000,   0.00000,   0.00000,   1.00000],[0.70711,   0.00000,   0.00000,  -0.70711],[0.50000,   0.50000,   0.50000,   0.50000],[0.50000,  -0.50000,  -0.50000,  -0.50000],[0.50000,  -0.50000,   0.50000,   0.50000],[0.50000,   0.50000,  -0.50000,  -0.50000],[0.50000,   0.50000,  -0.50000,   0.50000],[0.50000,  -0.50000,   0.50000,  -0.50000],[0.50000,  -0.50000,  -0.50000,   0.50000],[0.50000,   0.50000,   0.50000,  -0.50000],[0.00000,   0.70711,   0.70711,   0.00000],[0.00000,  -0.70711,   0.70711,   0.00000],[0.00000,   0.70711,   0.00000,   0.70711],[0.00000,   0.70711,   0.00000,  -0.70711],[0.00000,   0.00000,   0.70711,   0.70711],[0.00000,   0.00000,   0.70711,  -0.70711]])

@jit('void(int64[:],float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:,:])',nopython=True,nogil=True)
def Euler2Quat(nEls,eul1,eul2,eul3,quats):
	OrientMat = np.zeros(9)
	Quat = np.zeros(4)
	for frameNr in range(nEls[0]):
		for xPos in range(nEls[1]):
			for yPos in range(nEls[2]):
				psi = eul1[frameNr,xPos,yPos]
				phi = eul2[frameNr,xPos,yPos]
				theta = eul3[frameNr,xPos,yPos]
				if (psi == fillVal):
					quats[frameNr,xPos,yPos,0] = fillVal
					quats[frameNr,xPos,yPos,1] = fillVal
					quats[frameNr,xPos,yPos,2] = fillVal
					quats[frameNr,xPos,yPos,3] = fillVal
				cps = cos(psi)
				cph = cos(phi)
				cth = cos(theta)
				sps = sin(psi)
				sph = sin(phi)
				sth = sin(theta)
				OrientMat[0] = cth * cps - sth * cph * sps
				OrientMat[1] = -cth * cph * sps - sth * cps
				OrientMat[2] = sph * sps
				OrientMat[3] = cth * sps + sth * cph * cps
				OrientMat[4] = cth * cph * cps - sth * sps
				OrientMat[5] = -sph * cps
				OrientMat[6] = sth * sph
				OrientMat[7] = cth * sph
				OrientMat[8] = cph
				trace = OrientMat[0] + OrientMat[4] + OrientMat[8];
				if(trace > 0):
					s = 0.5/sqrt(trace+1.0);
					Quat[0] = 0.25/s;
					Quat[1] = (OrientMat[7]-OrientMat[5])*s;
					Quat[2] = (OrientMat[2]-OrientMat[6])*s;
					Quat[3] = (OrientMat[3]-OrientMat[1])*s;
				else:
					if (OrientMat[0]>OrientMat[4] and OrientMat[0]>OrientMat[8]):
						s = 2.0*sqrt(1.0+OrientMat[0]-OrientMat[4]-OrientMat[8]);
						Quat[0] = (OrientMat[7]-OrientMat[5])/s;
						Quat[1] = 0.25*s;
						Quat[2] = (OrientMat[1]+OrientMat[3])/s;
						Quat[3] = (OrientMat[2]+OrientMat[6])/s;
					elif (OrientMat[4] > OrientMat[8]):
						s = 2.0*sqrt(1.0+OrientMat[4]-OrientMat[0]-OrientMat[8]);
						Quat[0] = (OrientMat[2]-OrientMat[6])/s;
						Quat[1] = (OrientMat[1]+OrientMat[3])/s;
						Quat[2] = 0.25*s;
						Quat[3] = (OrientMat[5]+OrientMat[7])/s;
					else:
						s = 2.0*sqrt(1.0+OrientMat[8]-OrientMat[0]-OrientMat[4]);
						Quat[0] = (OrientMat[3]-OrientMat[1])/s;
						Quat[1] = (OrientMat[2]+OrientMat[6])/s;
						Quat[2] = (OrientMat[5]+OrientMat[7])/s;
						Quat[3] = 0.25*s;
				if (Quat[0] < 0):
					Quat[0] = -Quat[0];
					Quat[1] = -Quat[1];
					Quat[2] = -Quat[2];
					Quat[3] = -Quat[3];
				QNorm = sqrt(Quat[0]*Quat[0] + Quat[1]*Quat[1] + Quat[2]*Quat[2] + Quat[3]*Quat[3]);
				Quat[0] /= QNorm;
				Quat[1] /= QNorm;
				Quat[2] /= QNorm;
				Quat[3] /= QNorm;
				quats[frameNr,xPos,yPos,0] = Quat[0]
				quats[frameNr,xPos,yPos,1] = Quat[1]
				quats[frameNr,xPos,yPos,2] = Quat[2]
				quats[frameNr,xPos,yPos,3] = Quat[3]

def MakeSymmetries(SGNr):
	Sym = np.zeros((24,4))
	if (SGNr <= 2): # Triclinic
		NrSymmetries = 1
		for i in range(NrSymmetries):
			for j in range(4):
				Sym[i][j] = TricSym[i][j]
	elif (SGNr > 2 and SGNr <= 15):  # Monoclinic
		NrSymmetries = 2
		for i in range(NrSymmetries):
			for j in range(4):
				Sym[i][j] = MonoSym[i][j]
	elif (SGNr >= 16 and SGNr <= 74): # Orthorhombic
		NrSymmetries = 4
		for i in range(NrSymmetries):
			for j in range(4):
				Sym[i][j] = OrtSym[i][j]
	elif (SGNr >= 75 and SGNr <= 142):  # Tetragonal
		NrSymmetries = 8
		for i in range(NrSymmetries):
			for j in range(4):
				Sym[i][j] = TetSym[i][j]
	elif (SGNr >= 143 and SGNr <= 167): # Trigonal
		NrSymmetries = 6
		for i in range(NrSymmetries):
			for j in range(4):
				Sym[i][j] = TrigSym[i][j]
	elif (SGNr >= 168 and SGNr <= 194): # Hexagonal
		NrSymmetries = 12
		for i in range(NrSymmetries):
			for j in range(4):
				Sym[i][j] = HexSym[i][j]
	elif (SGNr >= 195 and SGNr <= 230): # Cubic
		NrSymmetries = 24
		for i in range(NrSymmetries):
			for j in range(4):
				Sym[i][j] = CubSym[i][j]
	return NrSymmetries,Sym

@jit('void(float64[:],float64[:],float64[:])',nopython=True,nogil=True)
def QuaternionProduct(q, r, Q):
	Q[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
	Q[1] = r[1]*q[0] + r[0]*q[1] + r[3]*q[2] - r[2]*q[3]
	Q[2] = r[2]*q[0] + r[0]*q[2] + r[1]*q[3] - r[3]*q[1]
	Q[3] = r[3]*q[0] + r[0]*q[3] + r[2]*q[1] - r[1]*q[2]
	if (Q[0] < 0):
		Q[0] = -Q[0]
		Q[1] = -Q[1]
		Q[2] = -Q[2]
		Q[3] = -Q[3]

@jit('void(float64[:],float64[:],int64,float64[:,:])',nopython=True,nogil=True)
def BringDownToFundamentalRegionSym(QuatIn, QuatOut, NrSymmetries, Sym):
	qps = np.zeros((NrSymmetries,4))
	qt = np.zeros(4)
	q2 = np.zeros(4)
	maxCos=-10000.0
	for i in range(NrSymmetries):
		q2[0] = Sym[i][0]
		q2[1] = Sym[i][1]
		q2[2] = Sym[i][2]
		q2[3] = Sym[i][3]
		QuaternionProduct(QuatIn,q2,qt)
		qps[i][0] = qt[0]
		qps[i][1] = qt[1]
		qps[i][2] = qt[2]
		qps[i][3] = qt[3]
		if (maxCos < qt[0]):
			maxCos = qt[0]
			maxCosRowNr = i
	QuatOut[0] = qps[maxCosRowNr][0]
	QuatOut[1] = qps[maxCosRowNr][1]
	QuatOut[2] = qps[maxCosRowNr][2]
	QuatOut[3] = qps[maxCosRowNr][3]

@jit('double(float64[:],float64[:],int64,float64[:,:])',nopython=True,nogil=True)
def GetMisOrientationAngle(quat1, quat2, NrSymmetries, Sym):
	q1FR = np.zeros(4)
	q2FR = np.zeros(4)
	QP = np.zeros(4)
	MisV = np.zeros(4)
	BringDownToFundamentalRegionSym(quat1,q1FR,NrSymmetries,Sym)
	BringDownToFundamentalRegionSym(quat2,q2FR,NrSymmetries,Sym)
	q1FR[0] = -q1FR[0]
	QuaternionProduct(q1FR,q2FR,QP)
	BringDownToFundamentalRegionSym(QP,MisV,NrSymmetries,Sym)
	if (MisV[0] > 1):
		MisV[0] = 1
	return 2*(acos(MisV[0]))*rad2deg

@jit('void(float64[:,:,:,:],float64[:,:,:,:],int64[:],int64,float64[:,:])',nopython=True,nogil=True)
def MakeMisoArr(quats,misoarr,dims,NrSymmetries,Sym):
	quat1 = np.zeros(4)
	quat2 = np.zeros(4)
	for frameNr in range(dims[0]):
		for xpos in range(dims[1]):
			for ypos in range(dims[2]):
				quat1[0] = quats[frameNr][xpos][ypos][0]
				quat1[1] = quats[frameNr][xpos][ypos][1]
				quat1[2] = quats[frameNr][xpos][ypos][2]
				quat1[3] = quats[frameNr][xpos][ypos][3]
				if quat1[0] == fillVal: # fillVal quat, not to be used
					for i in range(13):
						misoarr[frameNr][xpos][ypos][i] = fillVal
				# -1,-1,-1
				f2 = frameNr - 1
				x2 = xpos - 1
				y2 = ypos - 1
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][0] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][0] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# -1,0,-1
				x2 = xpos
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][1] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][1] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# -1,1,-1
				x2 = xpos + 1
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][2] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][2] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# -1,-1,0
				x2 = xpos - 1
				y2 = ypos
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][3] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][3] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# -1,0,0
				x2 = xpos
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][4] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][4] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# -1,1,0
				x2 = xpos + 1
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][5] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][5] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# -1,-1,1
				x2 = xpos - 1
				y2 = ypos + 1
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][6] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][6] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# -1,0,1
				x2 = xpos
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][7] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][7] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# -1,1,1
				x2 = xpos + 1
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][8] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][8] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# 0,-1,-1
				f2 = frameNr
				x2 = xpos - 1
				y2 = ypos - 1
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][9] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][9] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# 0,0,-1
				x2 = xpos
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][10] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][10] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# 0,1,-1
				x2 = xpos + 1
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][11] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][11] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)
				# 0,-1,0
				x2 = xpos - 1
				y2 = ypos
				if f2 < 0 or x2 < 0 or y2 < 0:
					misoarr[frameNr][xpos][ypos][0] = fillVal
				else:
					quat2[0] = quats[f2][x2][y2][0]
					quat2[1] = quats[f2][x2][y2][1]
					quat2[2] = quats[f2][x2][y2][2]
					quat2[3] = quats[f2][x2][y2][3]
					if quat2[0] == fillVal:
						misoarr[frameNr][xpos][ypos][12] = fillVal
					else:
						misoarr[frameNr][xpos][ypos][12] = GetMisOrientationAngle(quat1,quat2,NrSymmetries,Sym)

def writeHDF5File(grID,eul1,eul2,eul3,conf,phNr,fileID):
	f = h5py.File(fileID,'w')
	grainIDs = f.create_group('GrainID')
	Euls = f.create_group('EulerAngles')
	Conf = f.create_group('Confidence')
	PhaseNr = f.create_group('PhaseNumber')
	GrainID = grainIDs.create_dataset('GrainNrs',data=grID,compression="gzip")
	Euler1 = Euls.create_dataset('EulerAngle1',data=eul1,compression="gzip")
	Euler2 = Euls.create_dataset('EulerAngle2',data=eul2,compression="gzip")
	Euler3 = Euls.create_dataset('EulerAngle3',data=eul3,compression="gzip")
	confidence = Conf.create_dataset('ConfidenceValue',data=conf,compression="gzip")
	PhaseNrs = PhaseNr.create_dataset('PhaseNr',data=phNr,compression="gzip")
	f.close()

def writeXMLXdmf(dims,deltas,fn,h5fn,sample_name):
	f = open(fn,'w')
	# Header
	f.write('<?xml version="1.0" ?>\n')
	f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
	f.write('<Xdmf xmlns:xi="http:#www.w3.org/2003/XInclude" Version="2.2">\n')
	f.write('<Information Name="%s" Value="0"/>\n'%sample_name)
	f.write('<Domain>\n')
	# Info about topology and Geometry
	f.write('<Grid Name="Structured Grid" GridType="Uniform">\n')
	f.write('<Topology TopologyType="3DCoRectMesh" Dimensions="%d %d %d">\n'%(dims[0]+1,dims[1]+1,dims[2]+1))
	f.write('</Topology>\n')
	f.write('<Geometry Type="ORIGIN_DXDYDZ">\n')
	f.write('<!-- Origin -->\n')
	f.write('<DataItem Dimensions="3" NumberType="Float" Format="XML">\n')
	f.write('%lf %lf %lf\n'%(-dims[0]/2*(abs(deltas[2])/deltas[2]),-dims[1]/2,-dims[2]/2))
	f.write('</DataItem>\n')
	f.write('<!-- DXDYDZ -->\n')
	f.write('<DataItem Dimensions="3" NumberType="Float" Format="XML">\n')
	f.write('%lf %lf %lf\n'%(deltas[2],deltas[0],deltas[1]))
	f.write('</DataItem>\n')
	f.write('</Geometry>\n')
	# Data: GrainID, EulerAngles, Confidence, PhaseNr
	f.write('<Attribute Name="GrainID" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Format="HDF" Dimensions="%d %d %d" NumberType="Int">\n'%(dims[0],dims[1],dims[2]))
	f.write('%s.h5:/GrainID/GrainNrs\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="EulerAngles" AttributeType="Vector" Center="Cell">\n')
	f.write('<DataItem ItemType="Function" Dimensions="%d %d %d 3"\n'%(dims[0],dims[1],dims[2]))
	f.write('	Function=" JOIN( $0,$1,$2 ) ">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/EulerAngles/EulerAngle1\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/EulerAngles/EulerAngle2\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/EulerAngles/EulerAngle3\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="Confidence" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/Confidence/ConfidenceValue\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="PhaseNumber" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/PhaseNumber/PhaseNr\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('</Grid>\n')
	f.write('</Domain>\n')
	f.write('</Xdmf>\n')
	# Close the file
	f.close()

@jit('void(float64[:,:],int64[:],float64[:,:,:])',nopython=True,nogil=True)
def mapData(data,dims,outArr):
	spacing = dims[2]
	nrRows,nrCols = data.shape
	outArr.fill(fillVal)
	gridSpacing = data[0,5]
	extent = int(math.ceil(gridSpacing/spacing))
	outArr[:,:,6] = 10000
	for i in range(nrRows):
		xPos = data[i,4]
		yPos = data[i,3]
		xBinNr = int(xPos/spacing + dims[0]/2)
		yBinNr = int(yPos/spacing + dims[1]/2)
		xT = spacing*(xBinNr - dims[0]/2)
		yT = spacing*(yBinNr - dims[1]/2)
		distt = math.sqrt((xT-xPos)*(xT-xPos)+(yT-yPos)*(yT-yPos))
		if (xBinNr < 0) or (xBinNr > dims[0]-1) or (yBinNr < 0) or (yBinNr > dims[1]-1):
			continue
		else:
			if (outArr[xBinNr,yBinNr,6] > distt):
				outArr[xBinNr,yBinNr,0] = data[i,0]
				outArr[xBinNr,yBinNr,1] = data[i,7]
				outArr[xBinNr,yBinNr,2] = data[i,8]
				outArr[xBinNr,yBinNr,3] = data[i,9]
				outArr[xBinNr,yBinNr,4] = data[i,10]
				outArr[xBinNr,yBinNr,5] = data[i,11]
				outArr[xBinNr,yBinNr,6] = distt
		for j in range(-extent,extent+1):
			for k in range(-extent,extent+1):
				xBinT = xBinNr + j
				yBinT = yBinNr + k
				if  (xBinT < 0) or (xBinT > dims[0]-1) or (yBinT < 0) or (yBinT > dims[1]-1):
					continue
				xT2 = spacing*(xBinT - dims[0]/2)
				yT2 = spacing*(yBinT - dims[1]/2)
				distt2 = math.sqrt(((xT2-xPos)*(xT2-xPos))+((yT2-yPos)*(yT2-yPos)))
				if (outArr[xBinT,yBinT,6] > distt2):
					outArr[xBinNr,yBinNr,0] = data[i,0]
					outArr[xBinNr,yBinNr,1] = data[i,7]
					outArr[xBinNr,yBinNr,2] = data[i,8]
					outArr[xBinNr,yBinNr,3] = data[i,9]
					outArr[xBinNr,yBinNr,4] = data[i,10]
					outArr[xBinNr,yBinNr,5] = data[i,11]
					# ~ outArr[xBinT,yBinT,0:6] = data[i,[0,7,8,9,10,11]]
					outArr[xBinT,yBinT,6] = distt2

@jit('void(float64[:,:,:,:],int64[:],float64[:,:,:]',nopython=True,nogil=True)
def calcKAM(misoarr,dims,kamarr):
	for frameNr in range(dims[0]):
		for xpos in range(dims[1]):
			for ypos in range(dims[2]):
				nEls = 0
				totmiso = 0
				for elNr in range(13):
					miso = misoarr[frameNr][xpos][ypos][elNr]

for fnr in range(startnr,endnr+1):
	print('LayerNr: '+ str(fnr))
	FileName = sampleName + 'Layer' + str(fnr) + '/' + filestem + str(fnr) + '.mic'
	t1 = time.time()
	micfiledata = np.genfromtxt(FileName,skip_header=4)
	data = micfiledata[micfiledata[:,10] > minConfidence,:]
	data = data.astype(float)
	mapData(data,dimarr,outarr)
	print(time.time() - t1)
	grainIDs[dataNr,:,:] = outarr[:,:,0]
	Euler1[dataNr,:,:] = outarr[:,:,1]
	Euler2[dataNr,:,:] = outarr[:,:,2]
	Euler3[dataNr,:,:] = outarr[:,:,3]
	Confidence[dataNr,:,:] = outarr[:,:,4]
	PhaseNr[dataNr,:,:] = outarr[:,:,5]
	dataNr += 1

# Make Symmetries
NrSymmetries,Sym = MakeSymmetries(spaceGroup)
Sym = Sym.astype(float)

# Get quaternions
quats = np.zeros((Dims[0],Dims[1],Dims[2],4))
Euler2Quat(Dims,Euler1,Euler2,Euler3,quats)

# Make MisoArray
MisoArr = np.zeros((Dims[0],Dims[1],Dims[2],13))
MakeMisoArr(quats,MisoArr,Dims,NrSymmetries,Sym)

# write files
writeHDF5File(grainIDs.astype(np.int32),Euler1.astype(np.float32),Euler2.astype(np.float32),Euler3.astype(np.float32),Confidence.astype(np.float32),PhaseNr.astype(np.float32),outfn+'.h5')
writeXMLXdmf(Dims,[xyspacing,xyspacing,zspacing],outfn+'.xmf',outfn,sampleName)

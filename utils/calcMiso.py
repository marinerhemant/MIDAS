from math import sin, cos, acos, sqrt
import numpy as np

TricSym=[[1.00000,0.00000,0.00000,0.00000],[1.00000,0.00000,0.00000,0.00000]];
MonoSym=[[1.00000,0.00000,0.00000,0.00000],[0.00000,1.00000,0.00000,0.00000]];
OrtSym=[[1.00000,0.00000,0.00000,0.00000],[1.00000,1.00000,0.00000,0.00000],[0.00000,0.00000,1.00000,0.00000],[0.00000,0.00000,0.00000,1.00000]];
TetSym=[[1.00000,0.00000,0.00000,0.00000],[0.70711,0.00000,0.00000,0.70711],[0.00000,0.00000,0.00000,1.00000],[0.70711,-0.00000,-0.00000,-0.70711],[0.00000,1.00000,0.00000,0.00000],[0.00000,0.00000,1.00000,0.00000],[0.00000,0.70711,0.70711,0.00000],[0.00000,-0.70711,0.70711,0.00000]];
TrigSym=[[1.00000,0.00000,0.00000,0.00000],[0.50000,0.00000,0.00000,0.86603],[0.50000,-0.00000,-0.00000,-0.86603],[0.00000,0.50000,-0.86603,0.00000],[0.00000,1.00000,0.00000,0.00000],[0.00000,0.50000,0.86603,0.00000]];
HexSym=[[1.00000,0.00000,0.00000,0.00000],[0.86603,0.00000,0.00000,0.50000],[0.50000,0.00000,0.00000,0.86603],[0.00000,0.00000,0.00000,1.00000],[0.50000,-0.00000,-0.00000,-0.86603],[0.86603,-0.00000,-0.00000,-0.50000],[0.00000,1.00000,0.00000,0.00000],[0.00000,0.86603,0.50000,0.00000],[0.00000,0.50000,0.86603,0.00000],[0.00000,0.00000,1.00000,0.00000],[0.00000,-0.50000,0.86603,0.00000],[0.00000,-0.86603,0.50000,0.00000]];
CubSym=[[1.00000,0.00000,0.00000,0.00000],[0.70711,0.70711,0.00000,0.00000],[0.00000,1.00000,0.00000,0.00000],[0.70711,-0.70711,0.00000,0.00000],[0.70711,0.00000,0.70711,0.00000],[0.00000,0.00000,1.00000,0.00000],[0.70711,0.00000,-0.70711,0.00000],[0.70711,0.00000,0.00000,0.70711],[0.00000,0.00000,0.00000,1.00000],[0.70711,0.00000,0.00000,-0.70711],[0.50000,0.50000,0.50000,0.50000],[0.50000,-0.50000,-0.50000,-0.50000],[0.50000,-0.50000,0.50000,0.50000],[0.50000,0.50000,-0.50000,-0.50000],[0.50000,0.50000,-0.50000,0.50000],[0.50000,-0.50000,0.50000,-0.50000],[0.50000,-0.50000,-0.50000,0.50000],[0.50000,0.50000,0.50000,-0.50000],[0.00000,0.70711,0.70711,0.00000],[0.00000,-0.70711,0.70711,0.00000],[0.00000,0.70711,0.00000,0.70711],[0.00000,0.70711,0.00000,-0.70711],[0.00000,0.00000,0.70711,0.70711],[0.00000,0.00000,0.70711,-0.70711]];

def QuaternionProduct(q,r):
	Q = [0,0,0,0]
	Q[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
	Q[1] = r[1]*q[0] + r[0]*q[1] + r[3]*q[2] - r[2]*q[3]
	Q[2] = r[2]*q[0] + r[0]*q[2] + r[1]*q[3] - r[3]*q[1]
	Q[3] = r[3]*q[0] + r[0]*q[3] + r[2]*q[1] - r[1]*q[2]
	if (Q[0] < 0):
		Q[0] = -Q[0]
		Q[1] = -Q[1]
		Q[2] = -Q[2]
		Q[3] = -Q[3]
	return Q

def MakeSymmetries(SGNr):
	if (SGNr <= 2):
		NrSymmetries = 1
		Sym = TricSym
	elif (SGNr <= 15):
		NrSymmetries = 2
		Sym = MonoSym
	elif (SGNr <= 74):
		NrSymmetries = 4
		Sym = OrtSym
	elif (SGNr <= 142):
		NrSymmetries = 8
		Sym = TetSym
	elif (SGNr <= 167):
		NrSymmetries = 6
		Sym = TrigSym
	elif (SGNr <= 194):
		NrSymmetries = 12
		Sym = HexSym
	elif (SGNr <= 230):
		NrSymmetries = 24
		Sym = CubSym
	return NrSymmetries,Sym

def BringDownToFundamentalRegionSym(QuatIn,NrSymmetries,Sym):
	maxCos = -10000.0
	q2 = [0,0,0,0]
	for i in range(NrSymmetries):
		q2[0] = Sym[i][0]
		q2[1] = Sym[i][1]
		q2[2] = Sym[i][2]
		q2[3] = Sym[i][3]
		qt = QuaternionProduct(QuatIn,q2)
		if (maxCos < qt[0]):
			maxCos = qt[0]
			QuatOut = qt
	return QuatOut

def OrientMat2Quat(OrientMat):
	Quat = [0,0,0,0]
	trace = OrientMat[0] + OrientMat[4] + OrientMat[8]
	if (trace > 0):
		s = 0.5/sqrt(trace+1.0)
		Quat[0] = 0.25/s
		Quat[1] = (OrientMat[7]-OrientMat[5])*s
		Quat[2] = (OrientMat[2]-OrientMat[6])*s
		Quat[3] = (OrientMat[3]-OrientMat[1])*s
	else:
		if (OrientMat[0]>OrientMat[4] and OrientMat[0]>OrientMat[8]):
			s = 2.0*sqrt(1.0+OrientMat[0]-OrientMat[4]-OrientMat[8])
			Quat[0] = (OrientMat[7]-OrientMat[5])/s
			Quat[1] = 0.25*s
			Quat[2] = (OrientMat[1]+OrientMat[3])/s
			Quat[3] = (OrientMat[2]+OrientMat[6])/s
		elif (OrientMat[4] > OrientMat[8]):
			s = 2.0*sqrt(1.0+OrientMat[4]-OrientMat[0]-OrientMat[8])
			Quat[0] = (OrientMat[2]-OrientMat[6])/s
			Quat[1] = (OrientMat[1]+OrientMat[3])/s
			Quat[2] = 0.25*s
			Quat[3] = (OrientMat[5]+OrientMat[7])/s
		else:
			s = 2.0*sqrt(1.0+OrientMat[8]-OrientMat[0]-OrientMat[4])
			Quat[0] = (OrientMat[3]-OrientMat[1])/s
			Quat[1] = (OrientMat[2]+OrientMat[6])/s
			Quat[2] = (OrientMat[5]+OrientMat[7])/s
			Quat[3] = 0.25*s
	if (Quat[0] < 0):
		Quat[0] = -Quat[0]
		Quat[1] = -Quat[1]
		Quat[2] = -Quat[2]
		Quat[3] = -Quat[3]
	QNorm = sqrt(Quat[0]*Quat[0] + Quat[1]*Quat[1] + Quat[2]*Quat[2] + Quat[3]*Quat[3])
	Quat[0] /= QNorm
	Quat[1] /= QNorm
	Quat[2] /= QNorm
	Quat[3] /= QNorm
	return Quat

def Euler2OrientMat(Euler):
	m_out = [0]*9
	psi = Euler[0]
	phi = Euler[1]
	theta = Euler[2]
	cps = cos(psi)
	cph = cos(phi)
	cth = cos(theta)
	sps = sin(psi)
	sph = sin(phi)
	sth = sin(theta)
	m_out[0] = cth * cps - sth * cph * sps
	m_out[1] = -cth * cph * sps - sth * cps
	m_out[2] = sph * sps
	m_out[3] = cth * sps + sth * cph * cps
	m_out[4] = cth * cph * cps - sth * sps
	m_out[5] = -sph * cps
	m_out[6] = sth * sph
	m_out[7] = cth * sph
	m_out[8] = cph
	return m_out

def eul2omMat(euler):
	m_out = np.zeros((euler.shape[0],9))
	cps = np.cos(euler[:,0])
	cph = np.cos(euler[:,1])
	cth = np.cos(euler[:,2])
	sps = np.sin(euler[:,0])
	sph = np.sin(euler[:,1])
	sth = np.sin(euler[:,2])
	m_out[:,0] = cth * cps - sth * cph * sps
	m_out[:,1] = -cth * cph * sps - sth * cps
	m_out[:,2] = sph * sps
	m_out[:,3] = cth * sps + sth * cph * cps
	m_out[:,4] = cth * cph * cps - sth * sps
	m_out[:,5] = -sph * cps
	m_out[:,6] = sth * sph
	m_out[:,7] = cth * sph
	m_out[:,8] = cph
	return m_out

# Euler angles must be in radians, answer in radians as well
def GetMisOrientationAngle(euler1,euler2,SGNum):
	quat1 = OrientMat2Quat(Euler2OrientMat(euler1))
	quat2 = OrientMat2Quat(Euler2OrientMat(euler2))
	NrSymmetries,Sym = MakeSymmetries(SGNum)
	q1FR = BringDownToFundamentalRegionSym(quat1,NrSymmetries,Sym)
	q2FR = BringDownToFundamentalRegionSym(quat2,NrSymmetries,Sym)
	q1FR = quat1
	q2FR = quat2
	q1FR[0] = -q1FR[0]
	QP = QuaternionProduct(q1FR,q2FR)
	MisV = BringDownToFundamentalRegionSym(QP,NrSymmetries,Sym)
	if (MisV[0] > 1):
		MisV[0] = 1
	return 2*acos(MisV[0])

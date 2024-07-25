from math import sin, cos, acos, sqrt, fabs, atan
import numpy as np
from scipy.linalg import expm
rad2deg = 57.2957795130823
deg2rad = 0.0174532925199433
EPS = 0.000000000001

def normalize(quat):
	return quat/np.linalg.norm(quat)

TricSym=[[1.00000,0.00000,0.00000,0.00000],[1.00000,0.00000,0.00000,0.00000]];
MonoSym=[[1.00000,0.00000,0.00000,0.00000],[0.00000,1.00000,0.00000,0.00000]];
OrtSym=[[1.00000,0.00000,0.00000,0.00000],[1.00000,1.00000,0.00000,0.00000],
		[0.00000,0.00000,1.00000,0.00000],[0.00000,0.00000,0.00000,1.00000]];
TetSym=[[1.00000,0.00000,0.00000,0.00000],[0.70711,0.00000,0.00000,0.70711],
		[0.00000,0.00000,0.00000,1.00000],[0.70711,-0.00000,-0.00000,-0.70711],
		[0.00000,1.00000,0.00000,0.00000],[0.00000,0.00000,1.00000,0.00000],
		[0.00000,0.70711,0.70711,0.00000],[0.00000,-0.70711,0.70711,0.00000]];
TrigSym=[[1.00000,0.00000,0.00000,0.00000],[0.50000,0.00000,0.00000,0.86603],
		[0.50000,-0.00000,-0.00000,-0.86603],[0.00000,0.50000,-0.86603,0.00000],
		[0.00000,1.00000,0.00000,0.00000],[0.00000,0.50000,0.86603,0.00000]];
HexSym=[[1.00000,0.00000,0.00000,0.00000],[0.86603,0.00000,0.00000,0.50000],
		[0.50000,0.00000,0.00000,0.86603],[0.00000,0.00000,0.00000,1.00000],
		[0.50000,-0.00000,-0.00000,-0.86603],[0.86603,-0.00000,-0.00000,-0.50000],
		[0.00000,1.00000,0.00000,0.00000],[0.00000,0.86603,0.50000,0.00000],
		[0.00000,0.50000,0.86603,0.00000],[0.00000,0.00000,1.00000,0.00000],
		[0.00000,-0.50000,0.86603,0.00000],[0.00000,-0.86603,0.50000,0.00000]];
CubSym=[[1.00000,0.00000,0.00000,0.00000],[0.70711,0.70711,0.00000,0.00000],
		[0.00000,1.00000,0.00000,0.00000],[0.70711,-0.70711,0.00000,0.00000],
		[0.70711,0.00000,0.70711,0.00000],[0.00000,0.00000,1.00000,0.00000],
		[0.70711,0.00000,-0.70711,0.00000],[0.70711,0.00000,0.00000,0.70711],
		[0.00000,0.00000,0.00000,1.00000],[0.70711,0.00000,0.00000,-0.70711],
		[0.50000,0.50000,0.50000,0.50000],[0.50000,-0.50000,-0.50000,-0.50000],
		[0.50000,-0.50000,0.50000,0.50000],[0.50000,0.50000,-0.50000,-0.50000],
		[0.50000,0.50000,-0.50000,0.50000],[0.50000,-0.50000,0.50000,-0.50000],
		[0.50000,-0.50000,-0.50000,0.50000],[0.50000,0.50000,0.50000,-0.50000],
		[0.00000,0.70711,0.70711,0.00000],[0.00000,-0.70711,0.70711,0.00000],
		[0.00000,0.70711,0.00000,0.70711],[0.00000,0.70711,0.00000,-0.70711],
		[0.00000,0.00000,0.70711,0.70711],[0.00000,0.00000,0.70711,-0.70711]];

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
	return normalize(Q)

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
	return normalize(QuatOut)

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
	return normalize(Quat)

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

def AxisAngleToOrientMat(axis,angle):
	R = np.zeros((3,3))
	lenInv = 1/np.linalg.norm(axis)
	u = axis[0]/lenInv
	v = axis[1]/lenInv
	w = axis[2]/lenInv
	angleRad = deg2rad*angle
	rcos = cos(angleRad)
	rsin = sin(angleRad)
	R[0][0] =      rcos + u*u*(1-rcos)
	R[1][0] =  w * rsin + v*u*(1-rcos)
	R[2][0] = -v * rsin + w*u*(1-rcos)
	R[0][1] = -w * rsin + u*v*(1-rcos)
	R[1][1] =      rcos + v*v*(1-rcos)
	R[2][1] =  u * rsin + w*v*(1-rcos)
	R[0][2] =  v * rsin + u*w*(1-rcos)
	R[1][2] = -u * rsin + v*w*(1-rcos)
	R[2][2] =      rcos + w*w*(1-rcos)
	return R

def MatrixMultF33(m,n):
	res = np.zeros((3,3))
	for r in range(3):
		res[r][0] = m[r][0]*n[0][0] + m[r][1]*n[1][0] + m[r][2]*n[2][0]
		res[r][1] = m[r][0]*n[0][1] + m[r][1]*n[1][1] + m[r][2]*n[2][1]
		res[r][2] = m[r][0]*n[0][2] + m[r][1]*n[1][2] + m[r][2]*n[2][2]
	return res

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
	q1FR[0] = -q1FR[0]
	QP = QuaternionProduct(q1FR,q2FR)
	MisV = BringDownToFundamentalRegionSym(QP,NrSymmetries,Sym)
	if (MisV[0] > 1):
		MisV[0] = 1
	return 2*acos(MisV[0]),MisV[1:]/sin(acos(MisV[0]))

# OM is 9 length vector
def GetMisOrientationAngleOM(OM1,OM2,SGNum):
	quat1 = OrientMat2Quat(OM1)
	quat2 = OrientMat2Quat(OM2)
	NrSymmetries,Sym = MakeSymmetries(SGNum)
	q1FR = BringDownToFundamentalRegionSym(quat1,NrSymmetries,Sym)
	q2FR = BringDownToFundamentalRegionSym(quat2,NrSymmetries,Sym)
	q1FR[0] = -q1FR[0]
	QP = QuaternionProduct(q1FR,q2FR)
	MisV = BringDownToFundamentalRegionSym(QP,NrSymmetries,Sym)
	if (MisV[0] > 1):
		MisV[0] = 1
	return 2*acos(MisV[0]),MisV[1:]/sin(acos(MisV[0]))

def CalcEtaAngleAll(y, z):
	alpha = rad2deg*np.arccos(z/np.linalg.norm(np.array([y,z]),axis=0))
	alpha[y>0] *= -1
	return alpha

def sin_cos_to_angle(s,c):
	if s>=0:
		return acos(c)
	else:
		return 2.0 * np.pi - acos(c)

def OrientMat2Euler(m):
	if len(m.shape)==1:
		m = m.reshape((3,3))
	determinant = np.linalg.det(m)
	if m[2][2]>1:
		m[2][2] = 1
	if (fabs(m[2][2] - 1.0) < EPS):
		phi = 0
	else:
		phi = acos(m[2][2])
	sph = sin(phi)
	if (fabs(sph) < EPS):
		psi = 0.0
		if fabs(m[2][2] - 1.0) < EPS:
			theta = sin_cos_to_angle(m[1][0], m[0][0])
		else: 
			theta = sin_cos_to_angle(-m[1][0], m[0][0])
	else:
		if fabs(-m[1][2] / sph) <= 1.0:
			psi = sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph)
		else:
			psi = sin_cos_to_angle(m[0][2] / sph,1)
		if fabs(m[2][1] / sph) <= 1.0:
			theta = sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph)
		else: 
			theta = sin_cos_to_angle(m[2][0] / sph,1)
	Euler = np.zeros(3)
	Euler[0] = psi
	Euler[1] = phi
	Euler[2] = theta
	return Euler

def rod2om(rod):
	cThOver2 = cos(atan(np.linalg.norm(rod)))
	th = 2*atan(np.linalg.norm(rod))
	quat = np.array([cThOver2, rod[0]/cThOver2, rod[1]/cThOver2, rod[2]/cThOver2])
	if th > EPS:
		w = quat[1:]*th/sin(th/2)
	else:
		w = np.array([0,0,0])
	wskew = np.array([[   0, -w[2],  w[1]],
				   	 [ w[2],     0, -w[0]],
					 [-w[1],  w[0],     0]])
	OM = expm(wskew)
	return OM
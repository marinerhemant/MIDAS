#define RealType double
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-5
#define MAX_N_SPOTS 5000
#define MAX_N_OMEGA_RANGES 20
#define MAX_POINTS_GRID_GOOD 300000

void MatrixMultF(RealType m[3][3],RealType v[3],RealType r[3]);

void MatrixMultF33(RealType m[3][3],RealType n[3][3],RealType res[3][3]);

int CalcDiffractionSpots(double Distance, 
	double ExcludePoleAngle, 
	double OmegaRanges[MAX_N_OMEGA_RANGES][2], 
	int NoOfOmegaRanges,
	double hkls[5000][4],
	int n_hkls,
	double Thetas[5000],
	double BoxSizes[MAX_N_OMEGA_RANGES][4], 
	int *nTspots, 
	double OrientMatr[3][3], 
	double *TheorSpots);

void RotationTilts(
    double tx,
    double ty,
    double tz,
    double RotMatOut[3][3]);

void Euler2OrientMat(
    double Euler[3],
    double m_out[3][3]);

void
SimulateAccOrient(
    const int NrOfFiles,
    const int nLayers,
    const double ExcludePoleAngle,
    const double Lsd[nLayers],
    const long long int SizeObsSpots,
    const double XGrain[3],
    const double YGrain[3],
    double RotMatTilts[3][3],
    const double OmegaStart,
    const double OmegaStep,
    const double px,
    const double ybc[nLayers],
    const double zbc[nLayers],
    const double gs,
    double hkls[5000][4],
    int n_hkls,
    double Thetas[5000],
    double OmegaRanges[MAX_N_OMEGA_RANGES][2],
    const int NoOfOmegaRanges,
    double BoxSizes[MAX_N_OMEGA_RANGES][4],
    double P0[nLayers][3],
    const int NrPixelsGrid,
    uint16_t *ObsSpotsInfo,
    double OrientMatIn[3][3],
    double *TheorSpots,
    int voxNr,
    FILE *spF);

void
RotateAroundZ(
    RealType v1[3],
    RealType alpha,
    RealType v2[3]);

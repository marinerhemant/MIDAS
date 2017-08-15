#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file ep) runIndexRefineScanning (string psfn, string grainsfn, int rownr)
{
	indexrefinescanning psfn rownr stderr=filename(ep);
}

string psfn = arg("ParamsFile","ps.txt");
string grainsfn = arg("GrainsFile","Grains.csv");
int nrelements = toInt(arg("nrelements","26569"));

foreach i in [0:nrelements]{
	file simx<simple_mapper;location="Output"prefix=strcat("IndexRefine_",i,"_"),suffix=".err">;
	simx = runIndexRefineScanning(psfn,grainsfn,i);
}

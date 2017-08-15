#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file ep) runIndexRefineScanning (string psfn, string grainsfn, int rownr, string fldr)
{
	indexrefinescanning psfn grainsfn rownr fldr stderr=filename(ep);
}

string fldr = arg("Folder","");
string psfn = arg("ParamsFile","ps.txt");
string grainsfn = arg("GrainsFile","Grains.csv");
int nrelements = toInt(arg("nrelements","26569"));

foreach i in [1:nrelements]{
	file simx<simple_mapper;location="Output",prefix=strcat("IndexRefine_",i,"_"),suffix=".err">;
	simx = runIndexRefineScanning(psfn,grainsfn,i,fldr);
}

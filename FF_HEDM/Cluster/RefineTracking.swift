#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app indexrefine (file param, int spotsinput, file hkl, file spotsfile, file simout)
{
   strainsrefine spotsinput stdout=filename(simout);
}

file params <"paramstest.txt">;
file hkl <"hkls.csv">;
file spotsfile <"SpotsToIndex.csv">;
string outfldr = arg("outfolder","/clhome/TOMO1/aboc");

int spots[] = readData("SpotsToIndex.csv");

foreach i in spots {
	file simout<simple_mapper;location=outfldr,prefix=strcat("Refine_",i),suffix=".out">;
    indexrefine(params, i, hkl, spotsfile,simout);
}

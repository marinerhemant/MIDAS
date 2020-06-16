#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

# app (file simout) indexrefine (file param, int spotsinput, file hkl, file spotsfile)
# {
#    strainsrefine spotsinput stdout=filename(simout);
# }

app indexrefine (file param, int spotsinput, file hkl, file spotsfile)
{
   strainsrefine spotsinput;
}

file params <"paramstest.txt">;
file hkl <"hkls.csv">;
file spotsfile <"SpotsToIndex.csv">;
string outfldr = arg("outfolder","/clhome/TOMO1/aboc");

int spots[] = readData("SpotsToIndex.csv");

foreach i in spots {
#	file simout<simple_mapper;location=outfldr,prefix=strcat("Refine_",i),suffix=".out">;
#    simout = indexrefine(params, i, hkl, spotsfile);
    indexrefine(params, i, hkl, spotsfile);
}

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app indexrefine (file param, int spotsinput, file hkl, file spotsfile)
{
   strainsrefine spotsinput;
}

file params <"paramstest.txt">;
file hkl <"hkls.csv">;
file spotsfile <"SpotsToIndex.csv">;

int spots[] = readData("SpotsToIndex.csv");

foreach i in spots {
    indexrefine(params, i, hkl, spotsfile);
}

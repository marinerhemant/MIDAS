#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app indexrefine (file param, int spotsinput, file hkl, file spotsfile)
{
   indexstrains spotsinput;
}

file params <"paramstest.txt">;
file hkl <"hkls.csv">;
file spotsfile <"SpotsToIndex.csv">;

int spots[] = readData("SpotsToIndex.csv");

foreach i in spots {
    indexrefine(params, i, hkl, spotsfile);
}

# This is the trial with 2000 jobs
#type file;
#app indexrefine (int spotsinput, string folder)
#{
#   indexstrains spotsinput folder;
#}
#string fldr = arg("folder","/data/tomo1/NFTest/");
#
#foreach i in [1:2000] {
#    indexrefine(i, fldr);
#}

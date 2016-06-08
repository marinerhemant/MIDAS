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

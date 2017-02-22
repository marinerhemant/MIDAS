type file;

app indexrefine (int spotsinput, string folder)
{
   indexstrains spotsinput folder;
}

string fldr = arg("folder","/data/tomo1/NFTest/");

foreach i in [1:2000] {
    indexrefine(i, fldr);
}

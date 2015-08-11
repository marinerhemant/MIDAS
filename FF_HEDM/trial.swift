type file;

app (file out) echo_app (string s)
{
   echo s stdout=filename(out);
}

file out <"out.txt">;
out = echo_app("Hello world!");


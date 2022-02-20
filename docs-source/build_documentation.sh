#!/usr/bin/env bash
doxygen ./source/doxygen/Doxyfile
if make html ; then
    echo "make Sphinx succeeded"
else
    echo "make Sphinx failed"
    exit 1
fi
rm ../docs -rf
mkdir ../docs
touch ../docs/.nojekyll
mv build/html/* ../docs
rm build -rf
for file in ../docs/*.html
do
        htmlmin $file $file
done
exit 0

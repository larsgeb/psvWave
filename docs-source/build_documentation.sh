#!/usr/bin/env bash
pip install breathe
pip install htmlmin
cd source/doxygen
doxygen Doxyfile
cd ../..
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
#!/usr/bin/env bash
pip install breathe
pip install htmlmin
cd source/doxygen
doxygen Doxyfile
cd ../..
make html
rm ../docs -rf
mkdir ../docs
touch ../docs/.nojekyll
mv build/html/* ../docs
rm build -rf
for file in ../docs/*.html
do
        htmlmin $file $file
done

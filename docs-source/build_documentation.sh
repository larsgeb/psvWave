conda activate forward-virieux
cd source/doxygen
doxygen Doxyfile
cd ../..
make html
rm ../docs -rf
mkdir ../docs
touch ../docs/.nojekyll
mv build/html/* ../docs
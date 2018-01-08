#!/bin/bash

set -e

for py_version in 3.6
do
  echo $py_version
  package_path=$(conda-build --python $py_version \
                             --channel matsci \
                             --output-folder ./build/$py_version\
                             --output \
                             '.')
  filename=$(basename $package_path)

  conda-build --no-anaconda-upload \
              --python ${py_version} \
              --channel matsci \
              --output-folder "./build/$py_version" \
              '.'
  for platform in osx-64 linux-{32,64} win-{32,64}
  do
    conda-convert --platform $platform \
                  --output-dir "./build/$py_version/" \
                  $package_path
    anaconda upload --force "./build/$py_version/$platform/$filename"
  done
done

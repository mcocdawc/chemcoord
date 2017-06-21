#!/bin/sh

# anaconda login

root_dir=$(conda-build --output .)
root_dir=${root_dir%/*}
system=${root_dir##*/}
root_dir=${root_dir%/*}


for py_version in '2.7' '3.5' '3.6'
do
  py_version_dir="${root_dir}/${py_version}"
  package_dir="${py_version_dir}/${system}"

  # The following is a workaround because of this bug:
  # https://github.com/conda/conda-build/issues/1957
  # intended would be:
  # package_path=$(conda-build --python $py_version --output-folder $package_dir --output .)
  package_name=$(conda-build --python $py_version --output .)
  package_name=${package_name##*/}
  package_path="$package_dir/$package_name"

  conda-build --no-anaconda-upload --python $py_version --output-folder $py_version_dir .
  for platform in 'osx-64' 'linux-32' 'linux-64' 'win-32' 'win-64'
  do
    conda-convert -p $platform -o $py_version_dir $package_path
    anaconda upload $py_version_dir/$platform/$package_name
  done
done

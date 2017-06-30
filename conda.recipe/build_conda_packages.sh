#!/bin/bash

# =============================================================================
# Please note the bug, that for conda-build the option '--output' does
# not respect the directories given by '--output-folder':
# https://github.com/conda/conda-build/issues/1957
# =============================================================================

tmp=$(dirname "$(conda-build --output .)")
system=$(basename "$tmp")
root_dir=$(dirname "$tmp")


for py_version in 2.7 3.5 3.6
do
  package_name=$(basename "$(conda-build --python $py_version --output .)")
  package_path=$root_dir/$py_version/$system/$package_name

  conda-build --no-anaconda-upload \
              --python ${py_version} \
              --output-folder "${root_dir}/${py_version}" \
              "."
  for platform in osx-64 linux-{32,64} win-{32,64}
  do
    conda-convert --platform $platform \
                  --output-dir "$root_dir/$py_version" \
                  "$package_path"
    anaconda upload --force "$root_dir/$py_version/$platform/$package_name"
  done
done

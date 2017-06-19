#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup file for the chemcoord package.
"""

from __future__ import with_statement
from __future__ import absolute_import
from setuptools import setup, find_packages
from io import open
import os
import subprocess
import version

MAIN_PACKAGE = 'chemcoord'
DESCRIPTION = "Python module for dealing with chemical coordinates."
LICENSE = 'LGPLv3'
AUTHOR = 'Oskar Weser'
EMAIL = 'oskar.weser@gmail.com'
URL = 'https://github.com/mcocdawc/chemcoord'
# numba is not in the INSTALL_REQUIRES because it breaks the
# automatic generation of the documentation at readthedocs.org.
# Read the section 'I get import errors on libraries that depend on C modules'
# at http://docs.readthedocs.io/en/latest/faq.html
INSTALL_REQUIRES = ['numpy', 'pandas>=0.20', 'sortedcontainers', 'sympy',
                    'numba']
KEYWORDS = ['chemcoord', 'transformation', 'cartesian', 'internal',
            'chemistry', 'zmatrix', 'xyz', 'zmat', 'coordinates',
            'coordinate system']

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Physics']


def readme():
    '''Return the contents of the README.md file.'''
    with open('README.md') as freadme:
        return freadme.read()


def setup_package():
    setup(
        name=MAIN_PACKAGE,
        version=version.get_version(pep440=True),
        url=URL,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        include_package_data=True,
        keywords=KEYWORDS,
        license=LICENSE,
        long_description=readme(),
        classifiers=CLASSIFIERS,
        packages=find_packages('src'),
        package_dir={'': 'src'},
        install_requires=INSTALL_REQUIRES
    )


if __name__ == "__main__":
    setup_package()

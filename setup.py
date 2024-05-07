#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup file for the chemcoord package.
"""

from setuptools import setup, find_packages



MAIN_PACKAGE = 'chemcoord'
DESCRIPTION = "Python module for dealing with chemical coordinates."
LICENSE = 'LGPLv3'
AUTHOR = 'Oskar Weser'
EMAIL = 'oskar.weser@gmail.com'
URL = 'https://github.com/mcocdawc/chemcoord'
INSTALL_REQUIRES = ['numpy>=1.0', 'scipy', 'pandas>=1.0', 'numba>=0.35',
                    'sortedcontainers', 'sympy', 'six', 'pymatgen']
KEYWORDS = ['chemcoord', 'transformation', 'cartesian', 'internal',
            'chemistry', 'zmatrix', 'xyz', 'zmat', 'coordinates',
            'coordinate system']
CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    'Operating System :: Unix',
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Physics']
VERSION = '2.1.1'


def readme():
    '''Return the contents of the README.md file.'''
    with open('README.rst') as freadme:
        return freadme.read()


def setup_package():
    setup(
        name=MAIN_PACKAGE,
        version=VERSION,
        url=URL,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        include_package_data=True,
        keywords=KEYWORDS,
        license=LICENSE,
        long_description_content_type="text/x-rst",
        long_description=readme(),
        classifiers=CLASSIFIERS,
        packages=find_packages('src'),
        package_dir={'': 'src'},
        install_requires=INSTALL_REQUIRES,
    )


if __name__ == "__main__":
    setup_package()


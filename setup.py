#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup file for the chemcoord package.
"""

from __future__ import with_statement
from __future__ import absolute_import
from setuptools import setup, find_packages
from setuptools_scm import get_version
import setuptools_scm
from setuptools_scm import get_version
from io import open

MAIN_PACKAGE = 'chemcoord'
DESCRIPTION = "Python module for dealing with chemical coordinates."
<<<<<<< HEAD
VERSION = '1.3.0'
=======
>>>>>>> experimental
LICENSE = 'LGPLv3'
AUTHOR = 'Oskar Weser'
EMAIL = 'oskar.weser@gmail.com'
URL = 'https://github.com/mcocdawc/chemcoord'
<<<<<<< HEAD
INSTALL_REQUIRES = ['numpy', 'pandas>=0.20']
KEYWORDS = [
    'chemcoord', 'transformation', 'cartesian', 'internal', 'chemistry',
    'zmatrix', 'xyz', 'zmat', 'coordinates', 'coordinate system']
=======
INSTALL_REQUIRES = ['numpy', 'pandas', 'setuptools_scm']
SETUP_REQUIRES = ['setuptools_scm']
KEYWORDS = ['chemcoord', 'transformation', 'cartesian', 'internal',
            'chemistry', 'zmatrix', 'xyz', 'zmat', 'coordinates',
            'coordinate system']
>>>>>>> experimental

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Physics']


def give_version():
    release = setuptools_scm.get_version(root='.', relative_to=__file__)
    return '.'.join(release.split('.')[:3])


def readme():
    '''Return the contents of the README.md file.'''
    with open('README.md') as freadme:
        return freadme.read()


def setup_package():
    setup(
        name=MAIN_PACKAGE,
        version=give_version(),
        setup_requires=SETUP_REQUIRES,
        url=URL,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        include_package_data=True,
        keywords=KEYWORDS,
        license=LICENSE,
        long_description=readme(),
        classifiers=CLASSIFIERS,
        packages=find_packages(),
<<<<<<< HEAD
        # requires=INSTALL_REQUIRES
        install_requires=INSTALL_REQUIRES
    )
=======
        install_requires=INSTALL_REQUIRES)
>>>>>>> experimental


if __name__ == "__main__":
    setup_package()

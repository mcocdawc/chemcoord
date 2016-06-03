#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""Setup file for the chemcoord package.
"""

from __future__ import with_statement
from __future__ import absolute_import
from setuptools import setup, find_packages
from io import open


MAIN_PACKAGE = u'chemcoord'
DESCRIPTION = u"Python module for dealing with chemical coordinates."
VERSION = u'1.1.0'
LICENSE = u'LGPLv3'
AUTHOR = u'Oskar Weser'
EMAIL = u'oskar.weser@gmail.com'
URL = u'https://github.com/mcocdawc/chemcoord'
REQUIRES = [
    u'numpy', u'pandas', u'copy', u'math', u'collections', u'os', u'sys',
    u'io']
KEYWORDS = [
    u'chemcoord', u'transformation', u'cartesian', u'internal', u'chemistry',
    u'zmatrix', u'xyz', u'zmat', u'coordinates', u'coordinate system']

CLASSIFIERS = [
    u'Development Status :: 5 - Production/Stable',
    u'Environment :: Console',
    u'Intended Audience :: Science/Research',
    (
        u'License :: OSI Approved :: '
        u'GNU Lesser General Public License v3 (LGPLv3)'),
    u'Natural Language :: English',
    u'Programming Language :: Python :: 3',
    u'Topic :: Scientific/Engineering :: Chemistry',
    u'Topic :: Scientific/Engineering :: Physics']


def readme():
    u'''Return the contents of the README.md file.'''
    with open(u'README.md') as freadme:
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
        long_description=readme(),
        classifiers=CLASSIFIERS,
        packages=find_packages(),
        requires=REQUIRES
    )


if __name__ == u"__main__":
    setup_package()

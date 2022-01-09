#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup file for the chemcoord package.
"""

# The git get_version samples are from Git user aebrahim

from setuptools import setup, find_packages

from subprocess import check_output, CalledProcessError
from os import path, name, devnull, environ, listdir
import re
import shutil
import tempfile



MAIN_PACKAGE = 'chemcoord'
DESCRIPTION = "Python module for dealing with chemical coordinates."
LICENSE = 'LGPLv3'
AUTHOR = 'Oskar Weser'
EMAIL = 'oskar.weser@gmail.com'
URL = 'https://github.com/mcocdawc/chemcoord'
INSTALL_REQUIRES = ['numpy>=1.21', 'scipy', 'pandas>=1.0', 'numba>=0.35',
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
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Physics']


def readme():
    '''Return the contents of the README.md file.'''
    with open('README.rst') as freadme:
        return freadme.read()


def format_git_describe(git_str, pep440=False):
    """format the result of calling 'git describe' as a python version"""
    if git_str is None:
        return None
    if "-" not in git_str:  # currently at a tag
        return git_str
    else:
        # formatted as version-N-githash
        # want to convert to version.postN-githash
        git_str = git_str.replace("-", ".post", 1)
        if pep440:  # does not allow git hash afterwards
            return git_str.split("-")[0]
        else:
            return git_str.replace("-g", "+git")

def call_git_describe(abbrev=7):
    """return the string output of git desribe"""
    GIT_COMMAND = "git"
    CURRENT_DIRECTORY = path.dirname(path.abspath(__file__))
    try:
        with open(devnull, "w") as fnull:
            arguments = [GIT_COMMAND, "describe", "--tags",
                         "--abbrev=%d" % abbrev]
            return check_output(arguments, cwd=CURRENT_DIRECTORY,
                                stderr=fnull).decode("ascii").strip()
    except (OSError, CalledProcessError):
        return None


def get_version(pep440=False):
    """Tracks the version number.

    pep440: bool
        When True, this function returns a version string suitable for
        a release as defined by PEP 440. When False, the githash (if
        available) will be appended to the version string.

    The file VERSION holds the version information. If this is not a git
    repository, then it is reasonable to assume that the version is not
    being incremented and the version returned will be the release version as
    read from the file.

    However, if the script is located within an active git repository,
    git-describe is used to get the version information.

    The file VERSION will need to be changed by manually. This should be done
    before running git tag (set to the same as the version in the tag).

    """

    git_version = format_git_describe(call_git_describe(), pep440=pep440)
    if git_version is None:  # not a git repository
        raise ValueError("git_version is None")
    return git_version


def setup_package():
    setup(
        name=MAIN_PACKAGE,
        version=get_version(pep440=True),
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
        install_requires=INSTALL_REQUIRES,
    )


if __name__ == "__main__":
    setup_package()

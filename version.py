#!/usr/bin/env python
"""Get version identification from git

See the documentation of get_version for more information

Written by Git user aebrahim
"""
from __future__ import print_function

from subprocess import check_output, CalledProcessError
from os import path, name, devnull, environ, listdir
import re
import shutil
import tempfile


def sed_inplace(filename, pattern, repl):
    '''
    Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
    `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.
    '''
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)

    # For portability, NamedTemporaryFile() defaults to mode "w+b"
    # (i.e., binary writing with updating).
    # This is usually a good thing. In this case, # however, binary writing
    # imposes non-trivial encoding constraints trivially
    # resolved by switching to text writing. Let's do that.
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                tmp_file.write(pattern_compiled.sub(repl, line))

    # Overwrite the original file with the munged temporary file in a
    # manner preserving file attributes (e.g., permissions).
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)


__all__ = ("get_version",)

CURRENT_DIRECTORY = path.dirname(path.abspath(__file__))
VERSION_FILE = path.join(CURRENT_DIRECTORY, "VERSION")
CC_INIT = path.join(CURRENT_DIRECTORY, 'src', 'chemcoord', '__init__.py')
CC_TUTORIAL = path.join(CURRENT_DIRECTORY, 'docs', 'source', 'tutorial.rst')

GIT_COMMAND = "git"

if name == "nt":
    def find_git_on_windows():
        """find the path to the git executable on windows"""
        # first see if git is in the path
        try:
            check_output(["where", "/Q", "git"])
            # if this command succeeded, git is in the path
            return "git"
        # catch the exception thrown if git was not found
        except CalledProcessError:
            pass
        # There are several locations git.exe may be hiding
        possible_locations = []
        # look in program files for msysgit
        if "PROGRAMFILES(X86)" in environ:
            possible_locations.append("%s/Git/cmd/git.exe" %
                                      environ["PROGRAMFILES(X86)"])
        if "PROGRAMFILES" in environ:
            possible_locations.append("%s/Git/cmd/git.exe" %
                                      environ["PROGRAMFILES"])
        # look for the github version of git
        if "LOCALAPPDATA" in environ:
            github_dir = "%s/GitHub" % environ["LOCALAPPDATA"]
            if path.isdir(github_dir):
                for subdir in listdir(github_dir):
                    if not subdir.startswith("PortableGit"):
                        continue
                    possible_locations.append("%s/%s/bin/git.exe" %
                                              (github_dir, subdir))
        for possible_location in possible_locations:
            if path.isfile(possible_location):
                return possible_location
        # git was not found
        return "git"

    GIT_COMMAND = find_git_on_windows()


def call_git_describe(abbrev=7):
    """return the string output of git desribe"""
    try:
        with open(devnull, "w") as fnull:
            arguments = [GIT_COMMAND, "describe", "--tags",
                         "--abbrev=%d" % abbrev]
            return check_output(arguments, cwd=CURRENT_DIRECTORY,
                                stderr=fnull).decode("ascii").strip()
    except (OSError, CalledProcessError):
        return None


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


def read_release_version():
    """Read version information from VERSION file"""
    try:
        with open(VERSION_FILE, "r") as infile:
            version = str(infile.read().strip())
        if len(version) == 0:
            version = None
        return version
    except IOError:
        return None


def update_release_version():
    """Update VERSION file"""
    version = get_version(pep440=True)
    with open(VERSION_FILE, "w") as outfile:
        outfile.write(version)
        outfile.write("\n")


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
        return read_release_version()
    return git_version


# def call_git_hash():
#     """return the string output of git rev-parse HEAD"""
#     try:
#         with open(devnull, "w") as fnull:
#             arguments = [GIT_COMMAND, 'rev-parse', 'HEAD']
#             return check_output(arguments, cwd=CURRENT_DIRECTORY,
#                                 stderr=fnull).decode("ascii").strip()
#     except (OSError, CalledProcessError):
#         return None
#
#
# def read_git_hash():
#     """Read version information from VERSION file"""
#     try:
#         with open(CC_INIT, "r") as f:
#             found = False
#             while not found:
#                 line = f.readline().strip().split()
#                 try:
#                     found = True if line[0] == '_git_hash' else False
#                 except IndexError:
#                     pass
#         git_hash = line[2].replace('"', '')
#         if len(git_hash) == 0:
#             git_hash = None
#         return git_hash
#     except IOError:
#         return None
#
#
# def update_git_hash():
#     """Update cc.__init__.py"""
#     git_hash = get_git_hash()
#     sed_inplace(CC_INIT, '_git_hash = .*', '_git_hash = "{}"'.format(git_hash))
#
#
# def get_git_hash():
#     """Get git hash
#     """
#     git_hash = call_git_hash()
#     if git_hash is None:  # not a git repository
#         git_hash = read_git_hash()
#     return git_hash


def call_git_branch():
    """return the string output of git desribe"""
    try:
        with open(devnull, "w") as fnull:
            arguments = [GIT_COMMAND, 'rev-parse', '--abbrev-ref', 'HEAD']
            return check_output(arguments, cwd=CURRENT_DIRECTORY,
                                stderr=fnull).decode("ascii").strip()
    except (OSError, CalledProcessError):
        return None


def read_git_branch():
    """Read version information from VERSION file"""
    try:
        with open(CC_INIT, "r") as f:
            found = False
            while not found:
                line = f.readline().strip().split()
                try:
                    found = True if line[0] == '_git_branch' else False
                except IndexError:
                    pass
        branch = line[2].replace('"', '')
        if len(branch) == 0:
            branch = None
        return branch
    except IOError:
        return None


def update_git_branch():
    """Update cc.__init__.py"""
    branch = get_git_branch()
    sed_inplace(CC_INIT, '_git_branch = .*',
                '_git_branch = "{}"'.format(branch))


def get_git_branch():
    """Get git branch name
    """
    git_branch = call_git_branch()
    if git_branch is None:  # not a git repository
        git_branch = read_git_branch()
    return git_branch


def update_doc_tutorial_url():
    branch = get_git_branch()
    fixed_beginning = '<http://nbviewer.jupyter.org/github/mcocdawc/chemcoord/'
    pattern = fixed_beginning + 'blob/.*/Tutorial/Tutorial.ipynb>'
    replacement = (fixed_beginning
                   + 'blob/{}/Tutorial/Tutorial.ipynb>'.format(branch))
    sed_inplace(CC_TUTORIAL, pattern, replacement)


if __name__ == "__main__":
    update_git_branch()
    update_release_version()
    update_doc_tutorial_url()

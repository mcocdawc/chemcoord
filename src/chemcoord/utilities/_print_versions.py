# The following code was taken from the pandas project and modified.
# http://pandas.pydata.org/
import codecs
import importlib
import locale
import os
import platform
import struct
import sys


def get_sys_info():
    "Returns system information as a dict"

    blob = []

    # commit = cc._git_hash
    # blob.append(('commit', commit))

    try:
        (sysname, nodename, release, version,
         machine, processor) = platform.uname()
        blob.extend([
            ("python", "%d.%d.%d.%s.%s" % sys.version_info[:]),
            ("python-bits", struct.calcsize("P") * 8),
            ("OS", "%s" % (sysname)),
            ("OS-release", "%s" % (release)),
            # ("Version", "%s" % (version)),
            ("machine", "%s" % (machine)),
            ("processor", "%s" % (processor)),
            # ("byteorder", "%s" % sys.byteorder),
            ("LC_ALL", "%s" % os.environ.get('LC_ALL', "None")),
            ("LANG", "%s" % os.environ.get('LANG', "None")),
            ("LOCALE", "%s.%s" % locale.getlocale()),
        ])
    except Exception:
        pass

    return blob


def show_versions(as_json=False):
    sys_info = get_sys_info()

    deps = [
        # (MODULE_NAME, f(mod) -> mod version)
        ("chemcoord", lambda mod: mod.__version__),
        ("numpy", lambda mod: mod.version.version),
        ("scipy", lambda mod: mod.version.version),
        ("pandas", lambda mod: mod.__version__),
        ("numba", lambda mod: mod.__version__),
        ("sortedcontainers", lambda mod: mod.__version__),
        ("sympy", lambda mod: mod.__version__),
        ("pytest", lambda mod: mod.__version__),
        ("pip", lambda mod: mod.__version__),
        ("setuptools", lambda mod: mod.__version__),
        ("IPython", lambda mod: mod.__version__),
        ("sphinx", lambda mod: mod.__version__),
        # ("tables", lambda mod: mod.__version__),
        # ("matplotlib", lambda mod: mod.__version__),
        # ("Cython", lambda mod: mod.__version__),
        # ("xarray", lambda mod: mod.__version__),
        # ("patsy", lambda mod: mod.__version__),
        # ("dateutil", lambda mod: mod.__version__),
        # ("pytz", lambda mod: mod.VERSION),
        # ("blosc", lambda mod: mod.__version__),
        # ("bottleneck", lambda mod: mod.__version__),
        # ("numexpr", lambda mod: mod.__version__),
        # ("feather", lambda mod: mod.__version__),
        # ("openpyxl", lambda mod: mod.__version__),
        # ("xlrd", lambda mod: mod.__VERSION__),
        # ("xlwt", lambda mod: mod.__VERSION__),
        # ("xlsxwriter", lambda mod: mod.__version__),
        # ("lxml", lambda mod: mod.etree.__version__),
        # ("bs4", lambda mod: mod.__version__),
        # ("html5lib", lambda mod: mod.__version__),
        # ("sqlalchemy", lambda mod: mod.__version__),
        # ("pymysql", lambda mod: mod.__version__),
        # ("psycopg2", lambda mod: mod.__version__),
        # ("jinja2", lambda mod: mod.__version__),
        # ("s3fs", lambda mod: mod.__version__),
        # ("pandas_gbq", lambda mod: mod.__version__),
        # ("pandas_datareader", lambda mod: mod.__version__)
    ]

    deps_blob = list()
    for (modname, ver_f) in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = ver_f(mod)
            deps_blob.append((modname, ver))
        except Exception:
            deps_blob.append((modname, None))

    if (as_json):
        try:
            import json
        except Exception:
            import simplejson as json

        j = dict(system=dict(sys_info), dependencies=dict(deps_blob))

        if as_json is True:
            print(j)
        else:
            with codecs.open(as_json, "wb", encoding='utf8') as f:
                json.dump(j, f, indent=2)

    else:

        print("\nINSTALLED VERSIONS")
        print("------------------")

        for k, stat in sys_info:
            print("%s: %s" % (k, stat))

        print("")
        for k, stat in deps_blob:
            print("%s: %s" % (k, stat))


def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-j", "--json", metavar="FILE", nargs=1,
                      help="Save output as JSON into file, pass in "
                      "'-' to output to stdout")

    options = parser.parse_args()[0]

    if options.json == "-":
        options.json = True

    show_versions(as_json=options.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())

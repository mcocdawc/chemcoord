Installation Guide
==================

You need a working python 3.x installation together with the modules:

- numpy
- pandas
- math
- copy

All of them come shipped by default if you use e.g. the `anaconda3 installer <https://www.continuum.io/downloads/>`_.

Unix
++++

If you already have a directory for custom modules and a correctly set PYTHONPATH, 
just put the two files chemcoord.py and constants.py in this directory.

Otherwise create a directory e.g. `~/Python_scripts`, copy the files there and execute in the terminal the two commands::

    echo export PYTHONPATH="$PYTHONPATH:~/Python_scripts/" >> ~/.bash_profile
    source ~/.bash_profile




Windows
+++++++

To be written.

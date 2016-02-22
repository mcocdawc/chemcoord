Installation Guide
==================

You need a working python 3.x installation together with the modules:

- numpy
- pandas
- math
- copy
- collections
- os
- sys
- distutils.core

All of them come shipped by default with nearly every python installation.
You can use for example the `anaconda3 installer <https://www.continuum.io/downloads/>`_.

The advantage of the anaconda3 installer is that you get a lot of additional modules and programs,
that make it really easy to work with python. 
For example `Ipython <http://ipython.org/>`_ and the `jupyter notebooks <http://jupyter.org/>`_.
I highly recommend to use those.

Unix
++++

Create a directory where you want to put the source code and execute there::

   git clone -b stable https://github.com/mcocdawc/chemcoord.git

If you want to install it for the user then execute::

    python setup.py install --user

For a systemwide installation execute (you need probably ``sudo`` rights)::

    python setup.py install 

To read the documentation go to docs and execute for a browser based documentation::
    
    make html

Afterwards go to ``build/html`` and open the ``index.html`` file in a browser.

If you want to have a PDF version of the documentation execute::

    make latex
    make latexpdf
    make latexpdf

Afterwards go to ``build/latex`` and open the ``ChemCoord.pdf`` file with a PDF reader.
 



Windows
+++++++

I tested neither installation nor running the program on windows.
Since I use mainly the IO functionality provided by pandas for Dataframes and 
do all operations on numpy arrays and pandas Dataframes kept in memory, 
it could work on Windows.

If you get it installed and running, please report it on the Github page.



from distutils.core import setup

setup(name='chemcoord',
      version='1.0',
      description='A package to work with chemical coordinates',
      license='MIT',
      author='Oskar Weser',
      author_email='oskar.weser@gmail.com',
      url='https://github.com/mcocdawc/chemcoord',
      packages=['chemcoord'],
      requires=['numpy', 'pandas', 'copy', 'math', 'collections', 'os', 'sys'],
     )

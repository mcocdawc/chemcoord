# Changelog for v2.1.1 -> v2.1.2

- Removed scipy from the dependencies in the `setup.py` because it is indeed unused.
    This is necessary for the conda packaging.

- Made URLs in the doc that pointing to images more stable.

- Improved Logo.

- Removed `conda.recipe`. We now do this from the outside with `conda-forge`.

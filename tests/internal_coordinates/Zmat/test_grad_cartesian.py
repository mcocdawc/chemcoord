import itertools
import os
import sys

import chemcoord as cc
import numpy as np
import pandas as pd
import pytest
from chemcoord.exceptions import UndefinedCoordinateSystem
from chemcoord.xyz_functions import allclose


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_structure_path(script_path):
    test_path = os.path.join(script_path)
    while True:
        structure_path = os.path.join(test_path, 'structures')
        if os.path.exists(structure_path):
            return structure_path
        else:
            test_path = os.path.join(test_path, '..')


STRUCTURE_PATH = get_structure_path(get_script_path())


def test_grad_cartesian_properties():
    path = os.path.join(STRUCTURE_PATH, 'MIL53_beta.xyz')
    molecule = cc.Cartesian.read_xyz(path, start_index=1)
    fragment = molecule.get_fragment([(12, 17), (55, 60)])
    connection = np.array([[3, 99, 1, 12], [17, 3, 99, 12], [60, 3, 17, 12]])
    connection = pd.DataFrame(connection[:, 1:], index=connection[:, 0],
                              columns=['b', 'a', 'd'])
    c_table = molecule.get_construction_table([(fragment, connection)])
    zmolecule = molecule.get_zmat(c_table)

    r = 0.3
    zmolecule2 = zmolecule.copy()
    zmolecule2.safe_loc[3, 'bond'] += r

    dist_zmol = zmolecule.copy()
    dist_zmol.unsafe_loc[:, ['bond', 'angle', 'dihedral']] = 0
    dist_zmol.unsafe_loc[3, 'bond'] = r

    new = zmolecule.get_grad_cartesian(chain=False)(
            dist_zmol).loc[:, ['x', 'y', 'z']]
    index = new.index[~np.isclose(new, 0.).all(axis=1)]
    assert (index == [3]).all()

    new = zmolecule.get_grad_cartesian()(dist_zmol).loc[:, ['x', 'y', 'z']]
    index = new.index[~np.isclose(new, 0.).all(axis=1)]
    assert (index
            == [3, 17, 60, 6, 19, 62, 38, 37, 81, 80, 7, 39, 82, 10]).all()


def test_grad_cartesian():
    path = os.path.join(STRUCTURE_PATH, 'cis_platin.xyz')
    m = cc.Cartesian.read_xyz(path, start_index=1)


    zm1 = m.get_zmat()
    c_table_1 = zm1.loc[:, ['b', 'a', 'd']]
    c_table_1.loc[[10, 11], 'd'] = 9
    c_table_1.loc[[6, 7], 'd'] = 5
    zm1 = m.get_zmat(c_table_1)

    expected1 = np.array(
[[[[0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [1.2246467991473532e-16, -2.05, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [7.498798913309288e-33, -1.2552629691260368e-16, 0.0],
   [-1.0, -1.2552629691260368e-16, 2.5105259382520737e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [-1.3859417434252973e-16, 2.32, 0.0],
   [0.0, 0.0, 0.0],
   [-1.2246467991473532e-16, 2.32, 3.479442695775509e-32],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[-2.062449987798341e-16, 0.0, 0.0],
   [8.486445599452463e-33, -1.4205902870109295e-16, -1.0937538767153572e-63],
   [-2.1668615184306324e-16, -2.719981023310188e-32, -5.682361148043718e-16],
   [0.0, 0.0, 0.0],
   [1.0, 1.4205902870109295e-16, -5.682361148043718e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[-5.144503788329211e-33, 0.0, 0.0],
   [1.419046634655322e-16, -2.3754160000000004, 1.2932958484069384e-48],
   [1.7053917694543972e-32, 1.971758490477608e-48, 0.9450769999999998],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [5.789631688971223e-17, -1.992598313956677e-17, -0.9450769999999998],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[2.4353838601294378e-33, 0.0, 0.0],
   [1.432824807228753e-16, -2.3984799999999997, 5.046900411988019e-32],
   [-1.069588635110061e-32, -1.2626317257114155e-48, -0.4473950000000001],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-3.0871488776485294e-33, 1.0624937783613618e-33, 0.4473950000000001],
   [0.8234366374014372, -0.30616442225848084, 0.4473950000000001],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[2.5190065355265865e-33, 0.0, 0.0],
   [1.4595418212766882e-16, -2.443203, -4.8682876010936025e-32],
   [-1.1063146168768674e-32, -1.3059861408711251e-48, -0.4627569999999997],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [2.977892840541758e-33, -1.024891490854326e-33, 0.4627569999999997],
   [0.0, 0.0, 0.0],
   [-0.7942709783949574, 0.33970163463607644, 0.4627569999999997],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [1.2146199658561512e-32, -2.033215562020596e-16, -1.2932958484069382e-48],
   [-1.0, -2.033215562020596e-16, -0.9450769999999996],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.32556805805179784, -0.9450769999999998, 5.786927614988916e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.5274139090352864e-17, 0.0, 0.0],
   [-4.916431196536014e-17, 0.8229869999999999, -6.341604937868123e-48],
   [-1.0, 0.8229869999999999, 0.4473950000000002],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-7.852544157871546e-33, 2.7025840484898886e-33, -1.281818822642148e-16],
   [-0.3486703913933672, -0.9367336271288653, -1.281818822642148e-16],
   [0.0, 0.0, 0.0]],
  [[-1.4733578212543596e-17, 0.0, 0.0],
   [4.74243576886789e-17, -0.7938610000000003, 2.445666777717557e-48],
   [-1.0, -0.7938610000000003, 0.46275700000000003],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [4.219914089947816e-33, -1.4523538201384581e-33, 6.888423932014015e-17],
   [0.0, 0.0, 0.0],
   [-0.3934060641823097, -0.9188902700377233, 6.888423932014015e-17]]],
 [[[0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [0.0, 0.0, 2.5105259382520737e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [0.0, 0.0, 1.5372537772284038e-32],
   [1.2246467991473532e-16, 1.5372537772284038e-32, 2.05],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [0.0, 0.0, -2.841180574021859e-16],
   [0.0, 0.0, 0.0],
   [1.4997597826618573e-32, -2.8411805740218586e-16, 2.841180574021859e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [7.29953995049337e-32, 1.9846861501216097e-47, 1.7397213478877544e-32],
   [2.719981023310188e-32, 3.414291455286823e-48, -2.32],
   [0.0, 0.0, 0.0],
   [-2.449293598294706e-16, -3.479442695775509e-32, -2.32],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [-9.047617490669021e-17, -2.3775422497016e-32, 2.9090456010434093e-16],
   [-9.047617490669022e-17, -1.135713919485386e-32, -5.786927614988912e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.9455186087943385, -0.32541600000000037, 5.786927614988912e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[4.47990982665507e-33, 0.0, 0.0],
   [-2.028933247519264e-32, -1.1029138271604832e-32, 2.9372908548189433e-16],
   [-1.4851939771117047e-32, -1.71717906084161e-48, -0.822987],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-5.0405089822405296e-17, 1.7347752394385322e-17, 0.822987],
   [-0.4476394334178014, 0.1664381474996969, 0.822987],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[-4.3213631502055576e-33, 0.0, 0.0],
   [2.9212669696889314e-32, -1.1407840809917484e-32, 2.9920607336172104e-16],
   [2.3967709558454382e-32, 2.8666549019640674e-48, 0.7938610000000001],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-5.21358266206524e-17, 1.794341433133709e-17, -0.7938610000000001],
   [0.0, 0.0, 0.0],
   [-0.4629959843714644, 0.1980186825392439, -0.7938610000000001],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [9.047617490669021e-17, 2.3775422497015992e-32, 2.48997093000511e-32],
   [2.1294085482142554e-16, 3.6256848494904953e-32, 2.3754160000000004],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.9455186087943385, 0.3254160000000005, 5.786927614988914e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.8273982327343675e-16, 0.0, 0.0],
   [-4.82323634938773e-33, -1.1029138271604832e-32, -1.0078683952898831e-16],
   [7.172396504342276e-17, 6.956580945989229e-33, 2.3984799999999997],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [5.0405089822405265e-17, -1.7347752394385316e-17, -0.822987],
   [0.4476394334178013, -0.16643814749969674, -0.822987],
   [0.0, 0.0, 0.0]],
  [[-1.7627255211039031e-16, 0.0, 0.0],
   [-4.988849636973181e-33, -1.1407840809917486e-32, 9.721993326179176e-17],
   [6.998170403949316e-17, 6.913247191476506e-33, 2.4432029999999996],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [5.213582662065243e-17, -1.794341433133711e-17, 0.7938610000000001],
   [0.0, 0.0, 0.0],
   [0.46299598437146455, -0.19801868253924385, 0.7938610000000001]]],
 [[[1.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.0, -2.5105259382520737e-16, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.2246467991473532e-16, 2.05, 3.0745075544568075e-32],
   [-6.123233995736766e-17, 2.05, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [1.0, 2.841180574021859e-16, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [1.385941743425297e-16, -2.3199999999999994, -6.958885391551018e-32],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-6.123233995736766e-17, 2.32, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.0, -3.089218699750965e-16, 1.157385522997783e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.3255680580517977, -0.9450769999999998, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.0, -0.8229870000000002, -5.479008547045301e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.3486703913933672, -0.9367336271288653, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.0, 0.7938609999999999, -5.667138788330313e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.39340606418230983, -0.9188902700377233, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.4190466346553217e-16, 2.3754160000000004, -1.1573855229977827e-16],
   [-6.123233995736766e-17, 2.3754160000000004, 2.3417340879905174e-48],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-7.783161089959995e-17, -3.794329301032237e-17, -0.9450769999999998],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.9999999999999999, 0.0, 0.0],
   [-1.432824807228753e-16, 2.39848, 5.4790085470453066e-17],
   [-6.123233995736765e-17, 2.39848, -4.1211069309958323e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [8.630708535680646e-33, 2.4584311876306815e-32, 0.447395],
   [0.8234366374014372, -0.3061644222584809, 0.447395],
   [0.0, 0.0, 0.0]],
  [[0.9999999999999999, 0.0, 0.0],
   [-1.4595418212766882e-16, 2.443203, 5.667138788330319e-17],
   [-6.123233995736764e-17, 2.443203, 3.975258502682646e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [2.698063034032188e-33, -1.269093278436207e-32, 0.4627569999999998],
   [0.0, 0.0, 0.0],
   [-0.7942709783949574, 0.3397016346360764, 0.4627569999999998]]]])

    assert np.allclose(zm1.get_grad_cartesian(as_function=False), expected1)

    c_table_2 = zm1.loc[:, ['b', 'a', 'd']]
    c_table_2.loc[2, ['a', 'd']] = [4, 8]
    zm2 = m.get_zmat(c_table_2)

    expected2 = np.array(
[[[[0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [1.2246467991473532e-16, -2.05, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [7.498798913309288e-33, -1.2552629691260368e-16, 0.0],
   [-1.0, -1.2552629691260368e-16, 2.5105259382520737e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[-2.575717417130363e-16, 0.0, 0.0],
   [-1.385941743425297e-16, 2.3199999999999994, -6.349633666471827e-48],
   [3.5020254367488095e-48, 2.3199999999999994, 2.841180574021859e-16],
   [-1.224646799147353e-16, 2.32, 2.841180574021859e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[-2.062449987798341e-16, 0.0, 0.0],
   [8.486445599452463e-33, -1.4205902870109295e-16, -1.0937538767153572e-63],
   [-2.1668615184306324e-16, -2.719981023310188e-32, -5.682361148043718e-16],
   [0.0, 0.0, 0.0],
   [1.0, 1.4205902870109295e-16, -5.682361148043718e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[-5.144503788329211e-33, 0.0, 0.0],
   [1.419046634655322e-16, -2.3754160000000004, 1.2932958484069384e-48],
   [1.7053917694543972e-32, 1.971758490477608e-48, 0.9450769999999998],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [5.789631688971223e-17, -1.992598313956677e-17, -0.9450769999999998],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[2.4353838601294378e-33, 0.0, 0.0],
   [1.432824807228753e-16, -2.3984799999999997, 5.046900411988019e-32],
   [-1.069588635110061e-32, -1.2626317257114155e-48, -0.4473950000000001],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-3.0871488776485294e-33, 1.0624937783613618e-33, 0.4473950000000001],
   [0.8234366374014372, -0.30616442225848084, 0.4473950000000001],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[2.5190065355265865e-33, 0.0, 0.0],
   [1.4595418212766882e-16, -2.443203, -4.8682876010936025e-32],
   [-1.1063146168768674e-32, -1.3059861408711251e-48, -0.4627569999999997],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [2.977892840541758e-33, -1.024891490854326e-33, 0.4627569999999997],
   [0.0, 0.0, 0.0],
   [-0.7942709783949574, 0.33970163463607644, 0.4627569999999997],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [1.2146199658561512e-32, -2.033215562020596e-16, -1.2932958484069382e-48],
   [-1.0, -2.033215562020596e-16, -0.9450769999999996],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.32556805805179784, -0.9450769999999998, 5.786927614988916e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.5274139090352864e-17, 0.0, 0.0],
   [-4.916431196536014e-17, 0.8229869999999999, -6.341604937868123e-48],
   [-1.0, 0.8229869999999999, 0.4473950000000002],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-7.852544157871546e-33, 2.7025840484898886e-33, -1.281818822642148e-16],
   [-0.3486703913933672, -0.9367336271288653, -1.281818822642148e-16],
   [0.0, 0.0, 0.0]],
  [[-1.4733578212543596e-17, 0.0, 0.0],
   [4.74243576886789e-17, -0.7938610000000003, 2.445666777717557e-48],
   [-1.0, -0.7938610000000003, 0.46275700000000003],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [4.219914089947816e-33, -1.4523538201384581e-33, 6.888423932014015e-17],
   [0.0, 0.0, 0.0],
   [-0.3934060641823097, -0.9188902700377233, 6.888423932014015e-17]]],
 [[[0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [0.0, 0.0, 2.5105259382520737e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [0.0, 0.0, 1.5372537772284038e-32],
   [1.2246467991473532e-16, 1.5372537772284038e-32, 2.05],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [-2.719981023310188e-32, 5.719241562852334e-32, -2.841180574021859e-16],
   [-2.7199810233101876e-32, -2.8411805740218586e-16, 2.8411805740218586e-16],
   [1.2246467991473532e-16, -2.841180574021859e-16, 2.32],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [7.29953995049337e-32, 1.9846861501216097e-47, 1.7397213478877544e-32],
   [2.719981023310188e-32, 3.414291455286823e-48, -2.32],
   [0.0, 0.0, 0.0],
   [-2.449293598294706e-16, -3.479442695775509e-32, -2.32],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [-9.047617490669021e-17, -2.3775422497016e-32, 2.9090456010434093e-16],
   [-9.047617490669022e-17, -1.135713919485386e-32, -5.786927614988912e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.9455186087943385, -0.32541600000000037, 5.786927614988912e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[4.47990982665507e-33, 0.0, 0.0],
   [-2.028933247519264e-32, -1.1029138271604832e-32, 2.9372908548189433e-16],
   [-1.4851939771117047e-32, -1.71717906084161e-48, -0.822987],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-5.0405089822405296e-17, 1.7347752394385322e-17, 0.822987],
   [-0.4476394334178014, 0.1664381474996969, 0.822987],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[-4.3213631502055576e-33, 0.0, 0.0],
   [2.9212669696889314e-32, -1.1407840809917484e-32, 2.9920607336172104e-16],
   [2.3967709558454382e-32, 2.8666549019640674e-48, 0.7938610000000001],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-5.21358266206524e-17, 1.794341433133709e-17, -0.7938610000000001],
   [0.0, 0.0, 0.0],
   [-0.4629959843714644, 0.1980186825392439, -0.7938610000000001],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0],
   [9.047617490669021e-17, 2.3775422497015992e-32, 2.48997093000511e-32],
   [2.1294085482142554e-16, 3.6256848494904953e-32, 2.3754160000000004],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.9455186087943385, 0.3254160000000005, 5.786927614988914e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.8273982327343675e-16, 0.0, 0.0],
   [-4.82323634938773e-33, -1.1029138271604832e-32, -1.0078683952898831e-16],
   [7.172396504342276e-17, 6.956580945989229e-33, 2.3984799999999997],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [5.0405089822405265e-17, -1.7347752394385316e-17, -0.822987],
   [0.4476394334178013, -0.16643814749969674, -0.822987],
   [0.0, 0.0, 0.0]],
  [[-1.7627255211039031e-16, 0.0, 0.0],
   [-4.988849636973181e-33, -1.1407840809917486e-32, 9.721993326179176e-17],
   [6.998170403949316e-17, 6.913247191476506e-33, 2.4432029999999996],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [5.213582662065243e-17, -1.794341433133711e-17, 0.7938610000000001],
   [0.0, 0.0, 0.0],
   [0.46299598437146455, -0.19801868253924385, 0.7938610000000001]]],
 [[[1.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.0, -2.5105259382520737e-16, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.2246467991473532e-16, 2.05, 3.0745075544568075e-32],
   [-6.123233995736766e-17, 2.05, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-2.1668615184306322e-16, 2.841180574021858e-16, 3.4794426957755093e-32],
   [-4.2432227997262375e-33, 2.841180574021859e-16, -9.388095800725523e-32],
   [1.0, 2.841180574021859e-16, -2.841180574021859e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [1.385941743425297e-16, -2.3199999999999994, -6.958885391551018e-32],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-6.123233995736766e-17, 2.32, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.0, -3.089218699750965e-16, 1.157385522997783e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.3255680580517977, -0.9450769999999998, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.0, -0.8229870000000002, -5.479008547045301e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.3486703913933672, -0.9367336271288653, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.0, 0.7938609999999999, -5.667138788330313e-17],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-0.39340606418230983, -0.9188902700377233, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0],
   [-1.4190466346553217e-16, 2.3754160000000004, -1.1573855229977827e-16],
   [-6.123233995736766e-17, 2.3754160000000004, 2.3417340879905174e-48],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [-7.783161089959995e-17, -3.794329301032237e-17, -0.9450769999999998],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0]],
  [[0.9999999999999999, 0.0, 0.0],
   [-1.432824807228753e-16, 2.39848, 5.4790085470453066e-17],
   [-6.123233995736765e-17, 2.39848, -4.1211069309958323e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [8.630708535680646e-33, 2.4584311876306815e-32, 0.447395],
   [0.8234366374014372, -0.3061644222584809, 0.447395],
   [0.0, 0.0, 0.0]],
  [[0.9999999999999999, 0.0, 0.0],
   [-1.4595418212766882e-16, 2.443203, 5.667138788330319e-17],
   [-6.123233995736764e-17, 2.443203, 3.975258502682646e-16],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0],
   [2.698063034032188e-33, -1.269093278436207e-32, 0.4627569999999998],
   [0.0, 0.0, 0.0],
   [-0.7942709783949574, 0.3397016346360764, 0.4627569999999998]]]]
)

    assert np.allclose(zm2.get_grad_cartesian(as_function=False), expected2)

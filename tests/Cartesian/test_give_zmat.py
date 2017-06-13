import chemcoord as cc
import pytest
import os
import sys


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


OWN_DIR = get_script_path()


def test_back_and_forth1():
    molecule1 = cc.Cartesian.read_xyz(os.path.join(OWN_DIR, 'MIL53_small.xyz'))
    zmolecule = molecule1.give_zmat()
    molecule2 = zmolecule.give_cartesian()
    assert cc.xyz_functions.isclose(molecule1, molecule2, align=False)


# def test_back_and_forth2():
#     molecule1 = cc.Cartesian.read_xyz(os.path.join(OWN_DIR, 'ruthenium.xyz'))
#     zmolecule = molecule1.give_zmat()
#     molecule2 = zmolecule.give_cartesian()
#     assert cc.xyz_functions.isclose(molecule1, molecule2)


def test_back_and_forth3():
    molecule1 = cc.Cartesian.read_xyz(os.path.join(OWN_DIR, 'Cd_lattice.xyz'))
    zmolecule = molecule1.give_zmat()
    molecule2 = zmolecule.give_cartesian()
    assert cc.xyz_functions.isclose(molecule1, molecule2)


def test_back_and_forth4():
    molecule1 = cc.Cartesian.read_xyz(os.path.join(OWN_DIR, 'nasty_cube.xyz'))
    zmolecule = molecule1.give_zmat()
    molecule2 = zmolecule.give_cartesian()
    assert cc.xyz_functions.isclose(molecule1, molecule2, align=False)


def test_zmat_writing():
    molecule1 = cc.Cartesian.read_xyz(os.path.join(OWN_DIR, 'MIL53_small.xyz'))
    zmolecule = molecule1.give_zmat()
    molecule2 = zmolecule.give_cartesian()
    output = """ O  nan      nan  nan        nan  nan        nan
Cr    1 1.952026  nan        nan  nan        nan
 H    1 0.890000    2 119.132614  nan        nan
Cr    1 1.952026    2 121.734771    3 180.000000
 O    4 1.936862    1  86.222347    2 140.422999
 O    4 1.951342    1 180.000000    2 180.000000
 O    4 1.949800    1  90.270495    2 309.305333
 O    4 1.936862    1  86.222347    2 219.577001
 O    4 1.949800    1  90.270495    2  50.694667
 C    9 1.303680    4 138.980553    1 341.184666
 O   10 1.303680    9 117.045765    4 351.809748
 C   10 1.460790    9 121.477117    4 171.809674
 H   12 1.092764   10 110.882645    9  29.990887
 H   12 1.092501   10 110.887878    9 270.000039
 H   12 1.092764   10 110.882645    9 150.009191
 O    2 1.949800   11 101.387774    1 269.669588
 C   16 1.303680    2 138.980553   11  71.515078
 C   17 1.460790   16 121.477117    2 171.809674
 H   18 1.092764   17 110.882645   16  29.990887
 H   18 1.092764   17 110.882645   16 150.009191
 H   18 1.092501   17 110.887878   16 270.000039
 O    2 1.936862   11  89.746747   16 176.551481
 C   22 1.304149    2 139.863676   11  95.232681
 C   23 1.460789   22 123.219166    2 203.962617
 O   23 1.304149   22 113.561691    2  23.962716
 H   25 0.970000   23 106.000000   22 180.000000
 O    2 1.936862   22  78.948348   11 177.170556
 C   27 1.304149    2 139.863676   22  87.596763
 C   28 1.460789   27 123.219166    2 156.037383
 O   28 1.304149   27 113.561691    2 336.037284
 H   30 0.970000   28 106.000000   27 180.000000
 O    2 1.951342   22  93.764988   27  93.106761
 H   32 0.890000    2 127.750000   22 320.423689
 H   32 0.890000    2 127.750000   22 140.423689
 C    5 1.304149    4 139.863676    8  87.596763
 O   35 1.304149    5 113.561691    4 336.037284
 C   35 1.460789    5 123.219166    4 156.037383
 H   37 1.139673   35 108.068992    5  48.729034
 H   37 1.013970   35 116.103706    5 173.100296
 H   37 1.132098   35 108.508067    5 298.639448
 C    8 1.304149    4 139.863676    5 272.403237
 O   41 1.304149    8 113.561691    4  23.962716
 C   41 1.460789    8 123.219166    4 203.962617
 H   43 1.139673   41 108.068992    8 311.270966
 H   43 1.013970   41 116.103706    8 186.899704
 H   43 1.132098   41 108.508067    8  61.360552
 H    6 0.890000    4 127.750000    9 309.305392
 H    6 0.890000   47 104.499999    4 180.000000
 H   29 1.139673   28 108.068992   31 228.729092
 H   24 1.139673   23 108.068992   26 131.270908
 H   29 1.132098   49 102.192841   28 114.337231
 H   24 1.132098   50 102.192841   23 245.662769
 H   36 0.970000   35 106.000000    5 180.000000
 H   42 0.970000   41 106.000000    8 180.000000
 H   29 1.013970   51 110.811919   49 242.595217
 H   24 1.013970   52 110.811919   50 117.404783"""
    assert zmolecule.to_zmat() == output

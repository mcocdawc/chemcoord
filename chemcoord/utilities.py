import numpy as np
import pandas as pd
import math as m
import copy
from chemcoord import xyz_functions

def distance_frame(xyz_frame, origin):
    origin = np.array(origin, dtype=float)
    frame_distance = xyz_frame.copy()
    frame_distance['distance'] = np.linalg.norm(np.array(frame_distance.loc[:,'x':'z']) - origin, axis =1)
    return frame_distance

def normalize(vector):
    normed_vector = vector / np.linalg.norm(vector)
    return normed_vector

def rotation_matrix(axis, angle):
    '''
    Euler-Rodrigues formula for rotation matrix. 
    Input angle in radians.
    Follows the mathematical convention of "left hand rule for rotation"
    in a "right hand rule" coordinate system.
    '''
    # Normalize the axis
    axis = normalize(np.array(axis))
    a = np.cos( angle/2 )
    b,c,d = - axis * np.sin(angle/2)
    rot_matrix = np.array( [[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]]
        )
    return rot_matrix

def give_angle(Vector1, Vector2):
    '''
    Calculate the angle in degrees between two vectors.
    The vectors do not have to be normalized.
    '''
    vector1 = normalize(Vector1)
    vector2 = normalize(Vector2)

    # Is this ok
    scalar_product = np.dot(vector1, vector2)
    if  -1.00000000000001 < scalar_product < -1.:
        scalar_product = -1.

    elif 1.00000000000001 > scalar_product > 1.:
        scalar_product = 1.


    angle = m.acos(scalar_product)
    angle = np.degrees(angle)

    return angle


def add_dummy(zmat_frame, xyz_frame, index):
    zmat = zmat_frame.copy()
    p_list = []
    xyz = xyz_frame.copy()

    atom, bond, angle, dihedral = zmat.loc[index, ['atom', 'bond', 'angle', 'dihedral']]
    angle, dihedral = map(m.radians, (angle, dihedral))
    bond_with, angle_with, dihedral_with = zmat.loc[index, ['bond_with', 'angle_with', 'dihedral_with']]
    bond_with, angle_with, dihedral_with = map(int, (bond_with, angle_with, dihedral_with))
    
    
    vb = np.array(xyz.loc[bond_with, ['x', 'y', 'z']], dtype=float)
    va = np.array(xyz.loc[angle_with, ['x', 'y', 'z']], dtype=float)
    vd = np.array(xyz.loc[dihedral_with, ['x', 'y', 'z']], dtype=float)
    
    AB = vb - va
    DA = vd - va
    
    n1 = normalize(np.cross(DA, AB))
    ab = normalize(AB)
    
    # Vector of length distance pointing along the x-axis
    d = bond * -ab
    
    # Rotate d by the angle around the n1 axis
    d = np.dot(rotation_matrix(-n1, angle), d)
    d = np.dot(rotation_matrix(-ab, dihedral), d)
    
    # Add d to the position of q to get the new coordinates of the atom
    p = vb + d
    
    # Change of nonlocal variables
    p_list.append(list(p))
    xyz.loc[index] = [atom] + list(p)
    return xyz

def basistransform(xyz_frame, old_basis, new_basis):
    frame = xyz_frame.copy()
    v1, v2, v3 = old_basis
    ex, ey, ez = new_basis

    # Map v3 on ez
    axis = np.cross(ez, v3)
    angle = give_angle(ez, v3)
    rotationmatrix = rotation_matrix(axis, m.radians(angle))
    new_axes = np.dot(rotationmatrix, old_basis)
    frame = xyz_functions.move(frame, matrix = rotationmatrix)
    v1, v2, v3 = np.transpose(new_axes)

    # Map v1 on ex
    axis = ez
    angle = give_angle(ex, v1)
    if (angle != 0):
        angle = angle if 0 < np.dot(ez, np.cross(ex, v1)) else 360 - angle
    rotationmatrix = rotation_matrix(axis, m.radians(angle))
    new_axes = np.dot(rotationmatrix, new_axes)
    frame = xyz_functions.move(frame, matrix = rotationmatrix)
    v1, v2, v3 = np.transpose(new_axes)

    # Assert that new axes is right handed.
    if new_axes[1, 1] < 0:
        mirrormatrix = np.array([[1, 0, 0], [0,-1, 0],[0, 0, 1]])
        new_axes = np.dot(mirrormatrix, new_axes)
        frame = xyz_functions.move(frame, matrix = mirrormatrix)
        v1, v2, v3 = np.transpose(new_axes)

    return frame

def location(xyz_frame, index):
    vector = xyz_frame.ix[index, ['x', 'y', 'z']].get_values().astype(float)
    return vector

def orthormalization(basislist):
    v1, v2 =  basislist[0], basislist[1]

    e1 = normalize(v1)
    e3 = normalize(np.cross(e1, v2))
    e2 = normalize(np.cross(e3, e1))
    return [e1, e2, e3]





#    @staticmethod
#    def _introduce_pseudoatoms(zmat_frame, xyz_frame, first_three_atoms=None):
#        xyz = xyz_frame.copy()
#        for index in zmat_frame.index:
#            assert zmat_frame.at[index, 'atom'] == xyz_frame.at[index, 'atom']
#        if first_three_atoms is None:
#            zmat = zmat_frame.iloc[3:, :].copy()
#        else:
#            zmat = zmat_frame.loc[set(zmat_frame) - set(first_three_atoms), :].copy()
#        
#        buildlist = zmat_functions.build_list(zmat_frame)
#        problem_buildlist = zmat_functions.build_list(zmat[zmat['dihedral'].isnull()], complete=True)
#        problem_index = [element[0] for element in problem_buildlist] 
#        first_three = first_three_atoms if not (first_three_atoms is None) else [element[0] for element in buildlist[:3]]
#
#        max_index = zmat.index.max()
#        start = max_index + 1
#        dummy_index = list(range(start, start + 3))
#        dummy_buildlist = [
#                [dummy_index[0]] + first_three,
#                [dummy_index[1]] + [dummy_index[0]] + first_three[0:-1],
#                [dummy_index[2]] + [dummy_index[1]] + [dummy_index[0]] + first_three[0:-2],
#                ]
#
#
#        bond, angle, dihedral = 10., 85., 20.
#        for dummy in dummy_buildlist:
#            zmat.loc[dummy[0], ['bond_with', 'angle_with', 'dihedral_with']] = dummy[1:]
#            zmat.loc[dummy[0], ['bond', 'angle', 'dihedral']] = [bond, angle, dihedral]
#            zmat.loc[dummy[0], 'atom'] = 'U'
#            xyz = _utilities.add_dummy(zmat, xyz, dummy[0])
#
#
#        new_buildlist = []
#        for element in problem_buildlist:
#            new_buildlist.append([element[0]] + dummy_index)
#
#        bond_with_list = [element[1] for element in new_buildlist]
#        angle_with_list = [element[2] for element in new_buildlist]
#        dihedral_with_list = [element[3] for element in new_buildlist]
#
#
#        location_array = xyz.loc[dummy_index, ['x', 'y', 'z']].get_values().astype(float)
#
#        bond_list = xyz_functions._distance_optimized(
#                location_array,
#                buildlist = new_buildlist,
#                indexlist = dummy_index,
#                exclude_first = False
#                )
#        angle_list = xyz_functions._angle_degrees_optimized(
#                location_array,
#                new_buildlist,
#                indexlist = dummy_index,
#                exclude_first = False
#                )
#        dihedral_list = xyz_functions._dihedral_degrees_optimized(
#                location_array,
#                new_buildlist,
#                indexlist = dummy_index,
#                exclude_first = False
#               )
#
#        zmat.loc[problem_index, ['dihedral_with']] = dihedral_with_list
#        zmat.loc[problem_index, ['dihedral']] = dihedral_list
#        zmat.loc[problem_index, ['angle_with']] = angle_with_list
#        zmat.loc[problem_index, ['angle']] = angle_list
#        zmat.loc[problem_index, ['bond_with']] = bond_with_list
#        zmat.loc[problem_index, ['bond']] = bond_list
#        return new_buildlist, problem_index

#        vB, vA, vD = ht_xyz.ix[first_three, 'x':'z'].get_values().astype(float)
#        vI = ht_xyz.ix[dummy_index[0], 'x':'z'].get_values().astype(float)
#
#            xyz.loc[index, ['x', 'y', 'z']] = dummy1
#
#
#
#
#
#
#        
#        # deal with angles
#        for position, dummy in enumerate(dummy_buildlist):
#            # insert dummy into zmatrix
#            zmat.loc[dummy[0], ['bond_with', 'angle_with', 'dihedral_with']] = dummy[1:]
#            zmat.loc[dummy[0], ['bond', 'angle', 'dihedral']] = [bond, angle, dihedral]
#            zmat.loc[dummy[0], 'atom'] = 'X'
#
#            # update xyz frame
#            xyz = _utilities.add_dummy(zmat, xyz, dummy[0])
#
#            atom = problem_buildlist[position] 
#
#            # calculate new angles and dihedrals
##            print(dummy)
#            zmat.loc[atom[0], 'angle_with'] = dummy[0]
#
#            bond_with, angle_with, dihedral_with = zmat.loc[atom[0], ['bond_with', 'angle_with', 'dihedral_with']]
#            bond_with, angle_with, dihedral_with = map(int, (bond_with, angle_with, dihedral_with))
#
#            zmat.loc[atom[0], 'angle'] = xyz_functions.angle_degrees(xyz, atom[0], bond_with, angle_with)
#            zmat.loc[atom[0], 'dihedral'] = xyz_functions.dihedral_degrees(xyz, atom[0], bond_with, angle_with, dihedral_with)
#
##        # deal with dihedrals
#        problem_buildlist = zmat_functions.build_list(zmat[zmat['dihedral'].isnull()], complete=True)
#        problem_index = [element[0] for element in problem_buildlist] 
##
##        max_index = zmat.index.max()
##        start = max_index + 1
##        end = start + len(problem_index)
##        dummy_index = list(range(start, end))
##
##        dummy_buildlist = copy.deepcopy(problem_buildlist)
##        for position, element in enumerate(dummy_buildlist):
##            element[0] = dummy_index[position]
#
#
#
##        return zmat.ix[problem_index + dummy_index]
##        return xyz.ix[problem_index + dummy_index]
##        return xyz
#        return zmat, xyz
##        return zmat.ix[problem_index + dummy_index], xyz.ix[problem_index + dummy_index], problem_buildlist

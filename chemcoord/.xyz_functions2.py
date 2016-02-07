
# TODO rewrite
def make_similar(xyz_frame1, xyz_frame2, prealign = True):
    """Similarizes two xyz_frames.

    Returns a reindexed copy of xyz_frame2 that minimizes the distance 
    for each atom from xyz_frame1 to itself on xyz_frame2.

    .. warning:: The algorithm is still very basic, so it is important, to have a good
        prealignment and quite similar frames.

    Args:
        xyz_frame1 (pd.DataFrame): 
        xyz_frame2 (pd.DataFrame): 
        prealign (bool): If True both frames are moved to their barycenters and aligned along 
            their principal axes of inertia before reindexing.

    Returns:
        pd.DataFrame: Reindexed copy of xyz_frame2.
    """
    if prealign:
        frame1 = inertia(xyz_frame1)['transformed_frame']
        frame2 = inertia(xyz_frame2)['transformed_frame']
    else:
        frame1 = xyz_frame1.copy()
        frame2 = xyz_frame2.copy()

    assert set(frame1['atom']) == set(frame2['atom'])
    atomset = set(frame1['atom'])
    framedic = {}


    for atom in atomset:
        framedic[atom] = (frame1[frame1['atom'] == atom], frame2[frame2['atom'] == atom])
        assert framedic[atom][0].shape[0] == framedic[atom][1].shape[0]

    list_of_new_indexed_frames = []

    def _distance(vector1, vector2):
        length = np.linalg.norm(vector1 - vector2)
        return length


    for atom in atomset:
        index_dic = {}
        distance_frame_dic = {}

        frame1_indexlist = list(framedic[atom][0].index)
        for index_on_frame1 in frame1_indexlist:
            location_atom_frame1 = framedic[atom][0].loc[index_on_frame1, ['x', 'y', 'z']].get_values().astype(float)
            distances_to_atom_on_frame1 = distance_frame(framedic[atom][1], location_atom_frame1)
            
            
            distances_to_atom_on_frame1 = distances_to_atom_on_frame1.sort_values(by='distance')
            index_on_frame2 = distances_to_atom_on_frame1.iloc[0].name
            distance_new = distances_to_atom_on_frame1.at[index_on_frame2, 'distance']
            location_of_atom2 = distances_to_atom_on_frame1.loc[index_on_frame2, ['x', 'y', 'z']].get_values().astype(float)


            i = 1
            while True:
                if index_on_frame2 in index_dic.keys():
                    location_of_old_atom1 = framedic[atom][0].loc[index_dic[index_on_frame2], ['x', 'y', 'z']].get_values().astype(float)  
                    distance_old = utilities.distance(location_of_old_atom1, location_of_atom2)
                    if distance_new < distance_old:
                        frame1_indexlist.append(index_dic[index_on_frame2])
                        index_dic[index_on_frame2] = index_on_frame1
                        break
                    else:
                        index_on_frame2 = distances_to_atom_on_frame1.iloc[i].name


                        distance_new = distances_to_atom_on_frame1.at[index_on_frame2, 'distance']
                        location_of_atom2 = distances_to_atom_on_frame1.loc[index_on_frame2, ['x', 'y', 'z']].get_values().astype(float)
                        i = i + 1

                else:
                    index_dic[index_on_frame2] = index_on_frame1
                    break


        new_index = [index_dic[old_index2] for old_index2 in framedic[atom][1].index]
        framedic[atom][1].index = new_index
        list_of_new_indexed_frames.append(framedic[atom][1])
    
    xyz_frame3 = pd.concat(list_of_new_indexed_frames)
   
   
    return xyz_frame3.sort_index()


def from_to(xyz_frame1, xyz_frame2, step=5, extrapolate=(0,0)):
    """Returns list of xyz_frames for the movement from xyz_frame1 to xyz_frame2.
    
    Args:
        xyz_frame1 (pd.DataFrame): 
        xyz_frame2 (pd.DataFrame): 
        step (int): 
        extrapolate (tuple):

    Returns:
        list: The list contains xyz_frame1 as first and xyz_frame2 as last element.
        The number of intermediate frames is defined by step.
        Please note, that for this reason: len(list) = (step + 1).
        The numbers in extrapolate define how many frames are appended to the left and right of the list
        continuing the movement.
    """
    xyzframe1 = xyz_frame1.copy()
    xyzframe2 = xyz_frame2.copy()

    difference = xyzframe2.copy()
    difference.loc[:, ['x', 'y', 'z']] = xyzframe2.loc[:, ['x', 'y', 'z']] - (
            xyzframe1.loc[:, ['x', 'y', 'z']]
            )

    step_frame = difference.copy()
    step_frame.loc[:, ['x', 'y', 'z']]  = step_frame.loc[:, ['x', 'y', 'z']] / step

    list_of_xyzframes = []
    temp_xyz = xyzframe1.copy()

    for t in range(-extrapolate[0], step + 1 + extrapolate[1]):
        temp_xyz.loc[:, ['x', 'y', 'z']] = xyzframe1.loc[:, ['x', 'y', 'z']] + (
                    step_frame.loc[:, ['x', 'y', 'z']] * t
                    )
        appendframe = temp_xyz.copy()
        list_of_xyzframes.append(appendframe)
    return list_of_xyzframes


def modify(buildlist, entries):
    """Modify a buildlist for constructing zmatrices.
   
    In order to know about the meaning of the buildlist, go to :func:`to_zmat`.

    .. warning:: The user has to make sure himself that the modified buildlist is still valid. 
        This means that atoms use only other atoms as reference, that are in previous rows of the buildlist.

    Here is an example::

        In: buildlist_test = [[1], [2, 1], [3, 2, 1], [4, 3, 2, 1]]
        In: modify(buildlist_test, [[3, 1, 2], [4, 1, 2, 3]])
        Out: [[1], [2, 1], [3, 1, 2], [4, 1, 2, 3]]

    Args:
        buildlist (list): 
        entries (list): Entries you want to change.

    Returns:
        list: Modified buildlist. 
    """
    buildlist_modified = copy.deepcopy(buildlist)
    if type(entries[0]) == list:
        indexlist = [element[0] for element in buildlist]
        for entry in entries:
            position = indexlist.index(entry[0])
            buildlist_modified[position] = entry
    elif type(entries[0]) == int:
        position = indexlist.index(entries[0])
        buildlist_modified[position] = entries
    return buildlist_modified


def test_conversion(xyz_frame, steps=10, **kwargs):
    temp_xyz = xyz_frame.copy()
    framelist = [temp_xyz]
    for _ in range(steps):
        temp_zmat = to_zmat(temp_xyz)
        temp_xyz = zmat_functions.to_xyz(temp_zmat, **kwargs)
        framelist.append(temp_xyz)
    return framelist






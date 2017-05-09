import pandas as pd


class _generic_Indexer(object):
    def __init__(self, molecule):
        self.molecule = molecule

    def return_appropiate_type(self, selected):
        if isinstance(selected, pd.Series):
            series_like_frame = pd.DataFrame(selected).T
            if self.molecule._is_physical(series_like_frame) is not True:
                return selected
            else:
                selected = series_like_frame

        if self.molecule._is_physical(selected):
            molecule = self.molecule.__class__(selected)
            # NOTE here is the difference to the _pandas_wrapper definition
            # TODO make clear in documentation that metadata is an
            # alias/pointer
            # TODO persistent attributes have to be inserted here
            molecule.metadata = self.molecule.metadata.copy()
            molecule._metadata = self.molecule.metadata.copy()
            keys_not_to_keep = [
                'bond_dict'   # You could end up with loose ends
                ]
            for key in keys_not_to_keep:
                try:
                    molecule._metadata.pop(key)
                except KeyError:
                    pass
            return molecule
        else:
            return selected


class _Loc(_generic_Indexer):

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule.frame.loc[key[0], key[1]]
        else:
            selected = self.molecule.frame.loc[key]
        return self.return_appropiate_type(selected)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule.frame.loc[key[0], key[1]] = value
        else:
            self.molecule.frame.loc[key] = value


class _ILoc(_generic_Indexer):

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule.frame.iloc[key[0], key[1]]
        else:
            selected = self.molecule.frame.iloc[key]
        return self.return_appropiate_type(selected)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule.frame.iloc[key[0], key[1]] = value
        else:
            self.molecule.frame.iloc[key] = value

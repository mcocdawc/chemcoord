import pandas as pd


class _generic_Indexer(object):
    def __init__(self, molecule):
        self.molecule = molecule

    def return_appropiate_type(self, selected):
        if isinstance(selected, pd.Series):
            frame = pd.DataFrame(selected).T
            if self.molecule._required_cols <= set(frame.columns):
                selected = frame
            else:
                return selected

        if (isinstance(selected, pd.DataFrame)
                and self.molecule._required_cols <= set(selected.columns)):
            molecule = self.molecule.__class__(selected)
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
        try:
            return self.molecule._return_appropiate_type(selected)
        except AttributeError:
            return selected
        # return self.return_appropiate_type(selected)

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

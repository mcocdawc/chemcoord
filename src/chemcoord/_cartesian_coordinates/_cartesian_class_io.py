import json
import os
import re
import subprocess
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Hashable
from functools import partial
from threading import Thread
from types import ModuleType
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from pandas._typing import ReadCsvBuffer
from pymatgen.core.structure import Molecule as PyMatGenMolecule
from typing_extensions import Self

from chemcoord import constants
from chemcoord._cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord._cartesian_coordinates._cartesian_class_pandas_wrapper import (
    COORDS,
)
from chemcoord._generic_classes.generic_IO import GenericIO
from chemcoord.configuration import settings
from chemcoord.typing import (
    Axes,
    FloatFormatType,
    FormattersType,
    PathLike,
    SequenceNotStr,
    WriteBuffer,
)

pyscf: ModuleType | None = None
try:
    import pyscf  # type: ignore[no-redef]
    from pyscf.gto.mole import Mole
except ImportError:
    pass
ase: ModuleType | None = None
try:
    import ase  # type: ignore[no-redef]
    from ase.atoms import Atoms as AseAtoms
except ImportError:
    pass


class CartesianIO(CartesianCore, GenericIO):
    """This class provides IO-methods.

    Contains ``write_filetype`` and ``read_filetype`` methods
    like ``write_xyz()`` and ``read_xyz()``.

    The generic functions ``read`` and ``write``
    figure out themselves what the filetype is and use the
    appropiate IO-method.

    The ``view`` method uses external viewers to display a temporarily
    written xyz-file.
    """

    def __repr__(self) -> str:
        return self._frame.__repr__()

    def _repr_html_(self) -> str:  # noqa: PLW3201
        new = self._sympy_formatter()

        def insert_before_substring(insert_txt: str, substr: str, txt: str) -> str:
            "Under the assumption that substr only appears once."
            return (insert_txt + substr).join(txt.split(substr))

        html_txt = new._frame._repr_html_()  # type: ignore[operator]
        insert_txt = f"<caption>{self.__class__.__name__}</caption>\n"
        return insert_before_substring(insert_txt, "<thead>", html_txt)

    @overload
    def to_string(
        self,
        buf: None = None,
        columns: Axes | None = ...,
        col_space: int | list[int] | dict[Hashable, int] | None = ...,
        header: bool | SequenceNotStr[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        justify: str | None = ...,
        line_width: int | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: bool = ...,
    ) -> str: ...

    @overload
    def to_string(
        self,
        buf: WriteBuffer[str] | PathLike = ...,
        columns: Axes | None = ...,
        col_space: int | list[int] | dict[Hashable, int] | None = ...,
        header: bool | SequenceNotStr[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        justify: str | None = ...,
        line_width: int | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: bool = ...,
    ) -> None: ...

    def to_string(
        self,
        buf: WriteBuffer[str] | PathLike | None = None,
        columns: Axes | None = None,
        col_space: int | list[int] | dict[Hashable, int] | None = None,
        header: bool | SequenceNotStr[str] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: FormattersType | None = None,
        float_format: FloatFormatType | None = None,
        sparsify: bool | None = None,
        index_names: bool = True,
        justify: str | None = None,
        line_width: int | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: bool = False,
    ) -> str | None:
        """Render a DataFrame to a console-friendly tabular output.

        Wrapper around the :meth:`pandas.DataFrame.to_string` method.
        """
        return self._frame.to_string(  # type: ignore[misc]
            buf=buf,  # type: ignore[arg-type]
            columns=columns,  # type: ignore[arg-type]
            col_space=col_space,
            header=header,  # type: ignore[arg-type]
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,  # type: ignore[arg-type]
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            line_width=line_width,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
        )

    @overload
    def to_latex(
        self,
        buf: None = None,
        columns: Axes | None = ...,
        col_space: int | list[int] | dict[Hashable, int] | None = ...,
        header: bool | SequenceNotStr[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        bold_rows: bool = ...,
        column_format: str | None = ...,
        longtable: bool | None = ...,
        escape: bool | None = ...,
        encoding: str | None = ...,
        decimal: str = ...,
        multicolumn: bool | None = ...,
        multicolumn_format: str | None = ...,
        multirow: bool | None = ...,
    ) -> str: ...

    @overload
    def to_latex(
        self,
        buf: WriteBuffer[str] | PathLike = ...,
        columns: Axes | None = ...,
        col_space: int | list[int] | dict[Hashable, int] | None = ...,
        header: bool | SequenceNotStr[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        bold_rows: bool = ...,
        column_format: str | None = ...,
        longtable: bool | None = ...,
        escape: bool | None = ...,
        encoding: str | None = ...,
        decimal: str = ...,
        multicolumn: bool | None = ...,
        multicolumn_format: str | None = ...,
        multirow: bool | None = ...,
    ) -> None: ...

    def to_latex(
        self,
        buf: WriteBuffer[str] | PathLike | None = None,
        columns: Axes | None = None,
        col_space: int | list[int] | dict[Hashable, int] | None = None,
        header: bool | SequenceNotStr[str] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: FormattersType | None = None,
        float_format: FloatFormatType | None = None,
        sparsify: bool | None = None,
        index_names: bool = True,
        bold_rows: bool = True,
        column_format: str | None = None,
        longtable: bool | None = None,
        escape: bool | None = None,
        encoding: str | None = None,
        decimal: str = ".",
        multicolumn: bool | None = None,
        multicolumn_format: str | None = None,
        multirow: bool | None = None,
    ) -> str | None:
        """Render a DataFrame to a tabular environment table.

        You can splice this into a LaTeX document.
        Requires ``\\usepackage{booktabs}``.
        Wrapper around the :meth:`pandas.DataFrame.to_latex` method.
        """
        return self._frame.to_latex(  # type: ignore[misc,call-overload]
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            bold_rows=bold_rows,
            column_format=column_format,
            longtable=longtable,
            escape=escape,
            encoding=encoding,
            decimal=decimal,
            multicolumn=multicolumn,
            multicolumn_format=multicolumn_format,
            multirow=multirow,
        )

    @overload
    def to_xyz(
        self,
        buf: None = None,
        sort_index: bool = ...,
        index: bool = ...,
        header: bool | SequenceNotStr[str] = ...,
        float_format: FloatFormatType = ...,
        overwrite: bool = ...,
    ) -> str: ...

    @overload
    def to_xyz(
        self,
        buf: PathLike = ...,
        sort_index: bool = ...,
        index: bool = ...,
        header: bool | SequenceNotStr[str] = ...,
        float_format: FloatFormatType = ...,
        overwrite: bool = ...,
    ) -> None: ...

    def to_xyz(
        self,
        buf: PathLike | None = None,
        sort_index: bool = True,
        index: bool = False,
        header: bool | SequenceNotStr[str] = False,
        float_format: FloatFormatType = "{:.6f}".format,
        overwrite: bool = True,
    ) -> str | None:
        """Write xyz-file

        Args:
            buf (str, path object or file-like object):
                File path or object, if None is provided the result is returned as
                a string.
            sort_index (bool): If sort_index is true, the
                :class:`~chemcoord.Cartesian`
                is sorted by the index before writing.
            index (bool): Whether to print index (row) labels.
            float_format (one-parameter function): Formatter function
                to apply to column’s elements if they are floats.
                The result of this function must be a unicode string.
            overwrite (bool): May overwrite existing files.

        Returns:
            formatted : string (or unicode, depending on data and options)
        """
        if sort_index:
            molecule_string = (
                self.loc[:, ["atom", "x", "y", "z"]]
                .sort_index()
                .to_string(header=header, index=index, float_format=float_format)  # type: ignore[arg-type]
            )
        else:
            molecule_string = self.loc[:, ["atom", "x", "y", "z"]].to_string(
                header=header,  # type: ignore[arg-type]
                index=index,
                float_format=float_format,  # type: ignore[arg-type]
            )

        # NOTE the following might be removed in the future
        # introduced because of formatting bug in pandas
        # See https://github.com/pandas-dev/pandas/issues/13032
        space = " " * (self.loc[:, "atom"].str.len().max() - len(self.iloc[0, 0]))

        output = "{n}\n{message}\n{alignment}{frame_string}".format(
            n=len(self),
            alignment=space,
            frame_string=molecule_string,
            message="Created by chemcoord http://chemcoord.readthedocs.io/",
        )

        if buf is not None:
            with open(buf, mode="w" if overwrite else "x") as f:
                f.write(output)
            return None
        else:
            return output

    def write_xyz(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Deprecated, use :meth:`~chemcoord.Cartesian.to_xyz`"""
        message = "Will be removed in the future. Please use to_xyz()."
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.to_xyz(*args, **kwargs)

    @classmethod
    def read_xyz(
        cls,
        buf: ReadCsvBuffer[str] | PathLike,
        start_index: int = 0,
        nrows: int | None = None,
        engine: Literal["c", "python", "pyarrow", "python-fwf"] | None = None,
    ) -> Self:
        """Read a file of coordinate information.

        Reads xyz-files.

        Args:
            buf (str, path object or file-like object):
                This is passed on to :func:`pandas.read_table` and has the same
                constraints.
                Any valid string path is acceptable. The string could be a URL.
                Valid URL schemes include http, ftp, s3, and file.
                For file URLs, a host is expected. A local file could be: file://localhost/path/to/table.csv.
                If you want to pass in a path object, pandas accepts any os.PathLike.
                By file-like object, we refer to objects with a read() method,
                such as a file handler (e.g. via builtin open function) or StringIO.
            start_index (int):
            nrows (int): Number of rows of file to read.
                Note that the first two rows are implicitly excluded.
            engine (str): Wrapper for the same argument of :func:`pandas.read_csv`.

        Returns:
            Cartesian:
        """
        frame = pd.read_csv(
            buf,
            skiprows=2,
            comment="#",
            nrows=nrows,
            sep=r"\s+",
            names=["atom", "x", "y", "z"],
            dtype={"x": float, "y": float, "z": float},
            engine=engine,
        )

        remove_digits = partial(re.sub, r"[0-9]+", "")
        frame["atom"] = frame["atom"].apply(lambda x: remove_digits(x).capitalize())

        molecule = cls(frame)
        molecule.index = range(start_index, start_index + len(molecule))  # type: ignore[assignment]

        return molecule

    @overload
    def to_cjson(self, buf: None = None, **kwargs: Any) -> dict[str, Any]: ...
    @overload
    def to_cjson(self, buf: PathLike = ..., **kwargs: Any) -> None: ...

    def to_cjson(
        self, buf: PathLike | None = None, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Write a cjson file or return dictionary.

        The cjson format is specified
        `here <https://github.com/OpenChemistry/chemicaljson>`_.

        Args:
            buf (str): If it is a filepath, the data is written to
                filepath. If it is None, a dictionary with the cjson
                information is returned.
            kwargs: The keyword arguments are passed into the
                ``dump`` function of the
                `json library <https://docs.python.org/3/library/json.html>`_.

        Returns:
            dict:
        """
        cjson_dict: dict[str, Any] = {"chemical json": 0}

        cjson_dict["atoms"] = {}

        atomic_number = constants.elements["atomic_number"].to_dict()
        cjson_dict["atoms"] = {"elements": {}}
        cjson_dict["atoms"]["elements"]["number"] = [
            int(atomic_number[x]) for x in self["atom"]
        ]

        cjson_dict["atoms"]["coords"] = {}
        coords = self.loc[:, COORDS].values.reshape(len(self) * 3)
        cjson_dict["atoms"]["coords"]["3d"] = [float(x) for x in coords]

        bonds = []
        bond_dict = self.get_bonds()
        for i in bond_dict:
            for b in bond_dict[i]:
                bonds += [int(i), int(b)]
                bond_dict[b].remove(i)

        cjson_dict["bonds"] = {"connections": {}}
        cjson_dict["bonds"]["connections"]["index"] = bonds

        if buf is None:
            return cjson_dict
        else:
            with open(buf, mode="w") as f:
                f.write(json.dumps(cjson_dict, **kwargs))
            return None

    @classmethod
    def read_cjson(cls, buf: dict[str, Any] | PathLike) -> Self:
        """Read a cjson file or a dictionary.

        The cjson format is specified
        `here <https://github.com/OpenChemistry/chemicaljson>`_.

        Args:
            buf (str, dict): If it is a filepath, the data is read from
                filepath. If it is a dictionary, the dictionary is interpreted
                as cjson.

        Returns:
            Cartesian:
        """
        if isinstance(buf, dict):
            data = buf.copy()
        else:
            with open(buf) as f:
                data = json.load(f)
            assert data["chemical json"] == 0

        n_atoms = len(data["atoms"]["coords"]["3d"])
        metadata = {}
        _metadata = {}

        coords = np.array(data["atoms"]["coords"]["3d"]).reshape((n_atoms // 3, 3))

        atomic_number = constants.elements["atomic_number"]
        elements = [
            dict(zip(atomic_number, atomic_number.index))[x]
            for x in data["atoms"]["elements"]["number"]
        ]

        try:
            connections = data["bonds"]["connections"]["index"]
        except KeyError:
            pass
        else:
            bond_dict = defaultdict(set)
            for i, b in zip(connections[::2], connections[1::2]):
                bond_dict[i].add(b)
                bond_dict[b].add(i)
            _metadata["bond_dict"] = dict(bond_dict)

        try:
            metadata.update(data["properties"])
        except KeyError:
            pass

        # This is a mixin, so some attributes are not defined.
        return cls(
            atoms=elements,  # type: ignore[call-arg]
            coords=coords,
            _metadata=_metadata,
            metadata=metadata,
        )

    def view(self, viewer: PathLike | None = None, use_curr_dir: bool = False) -> None:
        """View your molecule.

        .. note:: This function writes a temporary file and opens it with
            an external viewer.
            If you modify your molecule afterwards you have to recall view
            in order to see the changes.

        Args:
            viewer (str): The external viewer to use. If it is None,
                the default as specified in cc.settings.defaults.viewer
                is used.
            use_curr_dir (bool): If True, the temporary file is written to
                the current diretory. Otherwise it gets written to the
                OS dependendent temporary directory.

        Returns:
            None:
        """
        if viewer is None:
            viewer = settings.defaults.viewer
        if use_curr_dir:
            TEMP_DIR = os.path.curdir
        else:
            TEMP_DIR = tempfile.gettempdir()

        def give_filename(i: int) -> str:
            return os.path.join(TEMP_DIR, f"ChemCoord_{i}.xyz")

        i = 1
        while os.path.exists(give_filename(i)):
            i = i + 1
        self.to_xyz(give_filename(i))

        def open_file(i: int) -> None:
            """Open file and close after being finished."""
            try:
                subprocess.check_call([viewer, give_filename(i)])
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise
            finally:
                if use_curr_dir:
                    pass
                else:
                    os.remove(give_filename(i))

        Thread(target=open_file, args=(i,)).start()

    def get_pymatgen_molecule(self) -> PyMatGenMolecule:
        """Create a Molecule instance of the pymatgen library

        .. note:: This method is only available, if the
            `pymatgen library <http://pymatgen.org>`_ is installed.

        Args:
            None

        Returns:
            :class:`pymatgen.core.structure.Molecule`:
        """
        return PyMatGenMolecule(
            self["atom"].values,
            self.loc[:, COORDS].values,  # type: ignore[arg-type]
        )

    @classmethod
    def from_pymatgen_molecule(cls, molecule: PyMatGenMolecule) -> Self:
        """Create an instance of the own class from a pymatgen molecule

        Args:
            molecule (:class:`pymatgen.core.structure.Molecule`):

        Returns:
            Cartesian:
        """
        return cls.set_atom_coords(
            atoms=[el.value for el in molecule.species], coords=molecule.cart_coords
        )

    if ase is not None:

        def get_ase_atoms(self) -> AseAtoms:
            """Create an Atoms instance of the ase library

            .. note:: This method is only available,
                if the `ase library <https://wiki.fysik.dtu.dk/ase/>`_
                is installed.

            Args:
                None

            Returns:
                :class:`ase.atoms.Atoms`:
            """
            return AseAtoms("".join(self["atom"]), self.loc[:, COORDS].values)

        @classmethod
        def from_ase_atoms(cls, atoms: AseAtoms) -> Self:
            """Create an instance of the own class from an ase molecule

            .. note:: This method is only available,
                if the `ase library <https://wiki.fysik.dtu.dk/ase/>`_
                is installed.

            Args:
                molecule (:class:`ase.atoms.Atoms`):

            Returns:
                Cartesian:
            """
            return cls.set_atom_coords(
                atoms=atoms.get_chemical_symbols(), coords=atoms.positions
            )  # type: ignore[call-arg]

    if pyscf is not None:

        def to_pyscf(self, **kwargs: Any) -> Mole:
            """Convert to a PySCF molecule.

            .. note:: This method is only available,
                if the `pyscf library <https://sunqm.github.io/pyscf/>`_ is installed.

            The kwargs are passed to the constructor of :class:`pyscf.gto.mole.Mole`.

            Returns:
                pyscf.gto.mole.Mole:
            """
            assert pyscf is not None, "pyscf is not installed"

            mol = Mole()
            mol.atom = [
                [row[1].iloc[0], tuple(row[1].iloc[1:4])]
                for row in self._frame.iterrows()
            ]
            mol.build(**kwargs)
            return mol

        @classmethod
        def from_pyscf(cls, mol: Mole) -> Self:
            """Create an instance of the own class from a PySCF molecule

            .. note:: This method is only available,
                if the `pyscf library <https://sunqm.github.io/pyscf/>`_ is installed.

            .. warning:: This method may lose information during the transformation.
                The :class:`pyscf.gto.mole.Mole` class contains more information
                than the :class:`Cartesian` class, such as charge, spin multipicity,
                or basis set.

            Args:
                mol (:class:`pyscf.gto.mole.Mole`):

            Returns:
                Cartesian:
            """
            return cls.set_atom_coords(
                atoms=mol.elements,
                coords=mol.atom_coords(unit="Angstrom"),
            )


# comment in order to test tests

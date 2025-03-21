import warnings
from typing import Literal, overload

import pandas as pd
from pandas._typing import ReadCsvBuffer
from typing_extensions import Self

import chemcoord.constants as constants
from chemcoord._generic_classes.generic_IO import GenericIO
from chemcoord._internal_coordinates._zmat_class_core import ZmatCore
from chemcoord.exceptions import InvalidReference, UndefinedCoordinateSystem
from chemcoord.typing import (
    FloatFormatType,
    PathLike,
    WriteBuffer,
)


class ZmatIO(ZmatCore, GenericIO):
    def __repr__(self) -> str:
        return self._frame.__repr__()

    #  def _abs_ref_formatter(
    #      self, format_as: Literal["raw", "string", "latex"] = "string"
    #  ) -> Self:

    def _abs_ref_formatter(self, format_as="string"):
        out = self.copy()
        if format_as == "raw":
            pass
        elif format_as == "string":
            rename = constants.string_repr
        elif format_as == "latex":
            rename = constants.latex_repr
        else:
            message = "Give either 'latex', 'string' or 'raw' as format"
            raise ValueError(message)
        if format_as != "raw":
            out._frame.replace(
                to_replace={col: rename for col in ["b", "a", "d"]}, inplace=True
            )
        return out

    def _repr_html_(self):  # noqa: PLW3201
        out = self._sympy_formatter()._abs_ref_formatter(format_as="string")

        def insert_before_substring(insert_txt, substr, txt):
            """Under the assumption that substr only appears once."""
            return (insert_txt + substr).join(txt.split(substr))

        html_txt = out._frame._repr_html_()
        insert_txt = f"<caption>{self.__class__.__name__}</caption>\n"
        return insert_before_substring(insert_txt, "<thead>", html_txt)

    def _remove_upper_triangle(self) -> Self:
        out = self.copy()
        out._frame = out._frame.astype(
            {k: str for k in ["b", "bond", "a", "angle", "d", "dihedral"]}
        )
        for i in range(min(len(self), 3)):
            out.unsafe_iloc[i, (2 * i + 1) :] = ""
        return out

    def to_string(
        self,
        format_abs_ref_as: Literal["string", "latex", "raw"] = "string",
        upper_triangle: bool = True,
        header: bool = True,
        index: bool = True,
        **kwargs,
    ) -> str:
        """Render a DataFrame to a console-friendly tabular output.

        Wrapper around the :meth:`pandas.DataFrame.to_string` method.
        """
        out = self._sympy_formatter()
        out = out._abs_ref_formatter(format_as=format_abs_ref_as)
        if not upper_triangle:
            out = out._remove_upper_triangle()

        content = out._frame.to_string(buf=None, header=header, index=index, **kwargs)
        if not index and not header:
            # NOTE(the following might be removed in the future
            # introduced because of formatting bug in pandas
            # See https://github.com/pandas-dev/pandas/issues/13032)
            space = " " * (out.loc[:, "atom"].str.len().max() - len(out.iloc[0, 0]))
            content = space + content
        return content

    @overload
    def to_latex(
        self,
        buf: None = None,
        upper_triangle: bool = ...,
        **kwargs,
    ) -> str: ...

    @overload
    def to_latex(
        self,
        buf: WriteBuffer[str] | PathLike = ...,
        upper_triangle: bool = ...,
        **kwargs,
    ) -> None: ...

    def to_latex(
        self,
        buf: WriteBuffer[str] | PathLike | None = None,
        upper_triangle: bool = True,
        **kwargs,
    ) -> str | None:
        """Render a DataFrame to a tabular environment table.

        You can splice this into a LaTeX document.
        Requires ``\\usepackage{booktabs}``.
        Wrapper around the :meth:`pandas.DataFrame.to_latex` method.
        """
        out = self._sympy_formatter()
        out = out._abs_ref_formatter(format_as="latex")
        if not upper_triangle:
            out = out._remove_upper_triangle()
        return out._frame.to_latex(buf=buf, **kwargs)

    @classmethod
    def read_zmat(
        cls, inputfile: ReadCsvBuffer[str] | PathLike, implicit_index: bool = True
    ) -> Self:
        """Reads a zmat file.

        Lines beginning with ``#`` are ignored.

        Args:
            inputfile (str):
            implicit_index (bool): If this option is true the first column
            has to be the element symbols for the atoms.
                The row number is used to determine the index.

        Returns:
            Zmat:
        """
        cols = ["atom", "b", "bond", "a", "angle", "d", "dihedral"]
        if implicit_index:
            zmat_frame = pd.read_csv(inputfile, comment="#", sep=r"\s+", names=cols)
            zmat_frame.index = range(1, len(zmat_frame) + 1)  # type: ignore[assignment]
        else:
            zmat_frame = pd.read_csv(
                inputfile, comment="#", sep=r"\s+", names=["temp_index"] + cols
            )
            zmat_frame.set_index("temp_index", drop=True, inplace=True)
            zmat_frame.index.name = None
        if pd.isnull(zmat_frame.iloc[0, 1]):
            zmat_values = [1.27, 127.0, 127.0]
            zmat_refs = [constants.int_label[x] for x in ["origin", "e_z", "e_x"]]
            for row, i in enumerate(zmat_frame.index[:3]):
                cols = ["b", "a", "d"]
                zmat_frame = zmat_frame.astype({k: "O" for k in cols})
                if row < 2:
                    zmat_frame.loc[i, cols[row:]] = zmat_refs[row:]
                    zmat_frame.loc[i, ["bond", "angle", "dihedral"][row:]] = (
                        zmat_values[row:]
                    )
                else:
                    zmat_frame.loc[i, "d"] = zmat_refs[2]
                    zmat_frame.loc[i, "dihedral"] = zmat_values[2]

        elif zmat_frame.iloc[0, 1] in constants.int_label.keys():
            zmat_frame = zmat_frame.replace(
                {col: constants.int_label for col in ["b", "a", "d"]}  # type: ignore[misc]
            )

        zmat_frame = cls._cast_correct_types(zmat_frame).replace(
            {col: constants.string_repr for col in ["b", "a", "d"]}  # type: ignore[misc]
        )
        try:
            Zmat = cls(zmat_frame)
        except InvalidReference:
            raise UndefinedCoordinateSystem(
                "Your zmatrix cannot be transformed to cartesian coordinates"
            )
        return Zmat

    def to_zmat(
        self,
        buf: PathLike | None = None,
        upper_triangle: bool = True,
        implicit_index: bool = True,
        float_format: FloatFormatType = "{:.6f}".format,
        overwrite: bool = True,
        header: bool = False,
    ) -> str | None:
        """Write zmat-file

        Args:
            buf (str): StringIO-like, optional buffer to write to
            implicit_index (bool): If implicit_index is set, the zmat indexing
                is changed to ``range(1, len(self) + 1)``.
                Using :meth:`~chemcoord.Zmat.change_numbering`
                Besides the index is omitted while writing which means,
                that the index is given
                implicitly by the row number.
            float_format (one-parameter function): Formatter function
                to apply to column’s elements if they are floats.
                The result of this function must be a unicode string.
            overwrite (bool): May overwrite existing files.

        Returns:
            formatted : string (or unicode, depending on data and options)
        """
        out = self.copy()
        if implicit_index:
            out = out.change_numbering(new_index=range(1, len(self) + 1))
        if not upper_triangle:
            out = out._remove_upper_triangle()

        output = out.to_string(
            index=(not implicit_index), float_format=float_format, header=header
        )

        if buf is not None:
            with open(buf, mode="w" if overwrite else "x") as f:
                f.write(output)
            return None
        else:
            return output

    def write(self, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Zmat.to_zmat`"""
        message = "Will be removed in the future. Please use to_zmat()."
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.to_zmat(*args, **kwargs)

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Self, TypeAlias

import numpy as np
from attrs import define
from joblib import Parallel, delayed
from numpy import float64
from numpy.linalg import lstsq

# had to put these here to avoid circular import
from sortedcontainers import SortedSet

from chemcoord.configuration import settings
from chemcoord.typing import ArithmeticOther, Vector

if TYPE_CHECKING:
    from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian

#: Unfortunately SortedSet is not a generic type, if it was, the primitives
#: would be declared as
#: ``SortedSet[tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]``
Primitives: TypeAlias = SortedSet


# the key prioritizes length, then sorts lexicographically
SetOfPrimitives = partial(SortedSet, key=lambda x: (len(x), x))


@define
class RedundantInternalCoordinates:
    q: Vector[float64]
    primitives_idx: Primitives

    #: The reference is an example cartesian for which the redundant
    #: internal coordinates could be defined.
    #: This is relevant as
    #: 1. starting guess and
    #: 2. to keep track of the index of the molecule.
    reference: Cartesian

    def copy(self) -> Self:
        return self.__class__(
            self.q.copy(), self.primitives_idx.copy(), self.reference.copy()
        )

    def __sub__(self, other: Self) -> DeltaRedundantInternalCoordinates:
        if self.primitives_idx != other.primitives_idx:
            raise ValueError("Can only add q with the same primitive indices")
        return DeltaRedundantInternalCoordinates(
            self.q - other.q,  # type: ignore[arg-type]
            self.primitives_idx,
            self.reference,
        )

    def __add__(self, other: DeltaRedundantInternalCoordinates) -> Self:
        if self.primitives_idx != other.primitives_idx:
            raise ValueError("Can only add q with the same primitive indices")
        new = self.copy()
        new.q = self.q + other.delta_q  # type: ignore[assignment]
        return new

    def get_cartesian(
        self,
        *,
        start_guess: Cartesian | None = None,
        rtol: float = 0,
        atol: float = 1e-8,
        max_iter: int = 100,
    ) -> Cartesian:
        from chemcoord._cartesian_coordinates.xyz_functions import allclose

        if start_guess is None:
            start_guess = self.reference
        elif set(start_guess.index) != set(self.reference.index):
            raise ValueError(
                "The start guess has to be indexed in the same way as self.reference"
            )
        start_guess = start_guess.loc[self.reference.index, :]

        previous = start_guess
        converged = False
        i = 0
        while not converged:
            if (i := i + 1) > max_iter:
                raise ValueError(f"Not converged after {max_iter} iterations.")

            B = previous.get_Wilson_B(self.primitives_idx)
            q_current = previous.get_ric(self.primitives_idx)

            delta_q = (self - q_current).minimize_dihedral()
            delta_x = lstsq(B, delta_q.delta_q)[0]

            new = previous + delta_x.reshape(len(previous), 3)
            converged = allclose(new, previous, rtol=rtol, atol=atol, align=True)
            previous = new
        return start_guess.align(new)[1] + start_guess.get_centroid()


@define
class DeltaRedundantInternalCoordinates:
    delta_q: Vector[float64]
    primitives_idx: Primitives

    #: The reference is an example cartesian for which the redundant
    #: internal coordinates could be defined.
    #: This is relevant as
    #: 1. starting guess and
    #: 2. to keep track of the index of the molecule.
    reference: Cartesian

    def copy(self) -> Self:
        return self.__class__(
            self.delta_q.copy(), self.primitives_idx.copy(), self.reference.copy()
        )

    def __mul__(self, other: ArithmeticOther) -> Self:
        new = self.copy()
        new.delta_q = new.delta_q * other
        return new

    def __rmul__(self, other: ArithmeticOther) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: ArithmeticOther) -> Self:
        new = self.copy()
        new.delta_q = new.delta_q / other
        return new

    def minimize_dihedral(self) -> Self:
        cleaned = np.array(
            [
                coord_val
                if len(idx) != 4
                else np.mod(coord_val + np.pi, 2 * np.pi) - np.pi
                for idx, coord_val in zip(self.primitives_idx, self.delta_q)
            ]
        )
        return self.__class__(cleaned, self.primitives_idx, self.reference)  # type: ignore[arg-type]


def get_primitives_idx(start: Cartesian, end: Cartesian) -> Primitives:
    aug_start_idx, bad_start_idx = start._fix_undef_dihedrals(
        start.get_primitives_idx()
    )
    aug_end_idx, bad_end_idx = end._fix_undef_dihedrals(end.get_primitives_idx())

    return (aug_start_idx | aug_end_idx) - (bad_start_idx | bad_end_idx)


def RIC_interpolate(
    start: Cartesian,
    end: Cartesian,
    N: int,
    coord_idx: Primitives | None = None,
) -> list[Cartesian]:
    if coord_idx is None:
        coord_idx = get_primitives_idx(start, end)
    q1, q2 = start.get_ric(coord_idx), end.get_ric(coord_idx)
    Delta_q = (q2 - q1).minimize_dihedral()
    Qs = [q1 + (Delta_q * i / (N - 1)) for i in range(N)]

    return Parallel(n_jobs=settings.defaults.n_worker)(
        delayed(lambda q: q.get_cartesian())(q) for q in Qs
    )
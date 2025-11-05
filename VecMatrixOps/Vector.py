#VecMatrixOps/Vector.py
from __future__ import annotations
import math
import numpy as np
from typing import TypeVar, Generic

T = TypeVar("T", int, float)

EPS = 1e-12

class Vector(Generic[T]):
    values: np.ndarray

    def __init__(self, values: list[float] | np.ndarray) -> None:
        if isinstance(values, list):
            arr = np.array(values, dtype=float, copy=True)
        elif isinstance(values, np.ndarray):
            assert values.ndim==1
            arr = np.array(values, dtype=float, copy=True)
        else:
            raise TypeError("All elements must be int or float")
        self.values = arr



    def __repr__(self) -> str:
        return f"Vector({self.values.tolist()})"

    def raw_repr(self) -> str:
        return repr(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key: int | slice) -> float | Vector:
        if isinstance(key, slice):
            return Vector(self.values[key])
        elif isinstance(key, int):
            return self.values[key]
        return NotImplemented

    def __setitem__(self, key: int | slice, val: float | list) -> None:
        if isinstance(key, int):
            self.values[key] = float(val)
        elif isinstance(key, slice):
            self.values[key] = np.asarray(val, dtype=float)
        else:
            raise TypeError("Indices must be integers or slices.")


    @classmethod
    def from_list(cls, lst: list[float]) -> Vector:
        if not isinstance(lst, list):
            raise TypeError("Honestly you should know better.")
        if not lst:
            raise ValueError("Vector cannot be empty.")
        if not isinstance(lst[0], (float, int)):
            raise TypeError("Vectors can only be instantiated from a list of floats.")
        return cls(lst)

    def tolist(self) -> list[float]:
        return self.values.tolist()

    @property
    def shape(self) -> tuple[int]:
        return (len(self.values), )

    def __eq__(self, other: Vector) -> bool:
        if not isinstance(other, (Vector)) or len(self) != len(other):
            return False
        return bool(np.allclose(self.values, other.values, atol=EPS))

    def __add__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of equal length to add.")
            return Vector(self.values + other.values)
        elif isinstance(other, (float, int)):
            return Vector(self.values + float(other))
        return NotImplemented

    def __radd__(self, other: int | float) -> Vector:
        if isinstance(other, (float, int)):
            return self.__add__(other)
        return NotImplemented

    def __iadd__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '+=' operation. In place addition can only be conducted with Vectors of equal length.")
            self.values += other.values
        elif isinstance(other, (float, int)):
            self.values += float(other)
        else:
            return NotImplemented
        return self

    def __sub__(self, other: Vector | float | int) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of equal length to subtract.")
            return Vector(self.values - other.values)
        elif isinstance(other, (float, int)):
            return Vector(self.values - float(other))
        return NotImplemented


    def __isub__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of equal length for in place subtraction.")
            self.values -= other.values
        elif isinstance(other, (float, int)):
            self.values -= float(other)
        else:
            return NotImplemented
        return self


    def __mul__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '*' operation. Vectors must be of equal length for element-wise multiplication.")
            return Vector(self.values * other.values)
        elif isinstance(other, (float, int)):
            return Vector(self.values * float(other))
        else:
            return NotImplemented

    def __rmul__(self, other: float | int) -> Vector:
        if isinstance(other, (float, int)):
            return self.__mul__(other)
        return NotImplemented

    def __imul__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of equal length for in place multiplication.")
            self.values *= other.values
        elif isinstance(other, (float, int)):
            self.values *= float(other)
        else:
            return NotImplemented
        return self

    def __matmul__(self, other: Vector) -> float:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of equal length for dot product.")
            return float(self.values @ other.values)
        return NotImplemented

    def __truediv__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vector division can only be conducted on vectors of equal length.")
            if np.any(np.abs(other.values) < EPS):
                raise ZeroDivisionError("Division by Zero: zero values within vector.")
            return Vector(self.values / other.values)
        elif isinstance(other, (float, int)):
            if abs(other) < EPS:
                raise ZeroDivisionError("Division by zero.")
            return Vector(self.values / other)
        return NotImplemented


    def __itruediv__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vector division can only be conducted on vectors of equal length.")
            if np.any(np.abs(other.values) < EPS):
                raise ZeroDivisionError("Division by Zero: zero values within vector.")
            self.values /= other.values
        elif isinstance(other, (float, int)):
            if abs(other) < EPS:
                raise ZeroDivisionError("Division by zero.")
            self.values /= other
        else:
            return NotImplemented
        return self


    def dot(self, other: Vector) -> float:
        return float(self.values @ other.values)

    def norm(self, p: float = 2) -> float:
        """Compute the L^p norm (magnitude) of the vector."""
        if isinstance(p, (int, float)):
            if p == np.inf:
                return float(np.max(np.abs(self.values)))
            elif p == 0:
                return float(np.count_nonzero(~np.isclose(self.values, 0, atol=EPS)))
            elif p > 0:
                return float(np.sum(np.abs(self.values)**p) ** (1/p))
            elif p < 0:
                raise ValueError("norm: p must be > 0 (or 0/inf handled specially)")
        raise TypeError("p must be an int or float.")

    def normalized(self) -> Vector:
        """Returns the unit vector in the direction of the input vector"""
        n = self.norm(2)
        if n < EPS:
            raise ValueError("Cannot normalize a zero or near-zero vector.")
        return self / n

    def distance(self, other: Vector, p: float = 2) -> float:
        """Returns the distance between the tips of 2 vectors, Manhattan Distance, Euclidean Distance
        or Chebyshev distance"""
        if not isinstance(other, Vector):
            return NotImplemented
        return (self - other).norm(p)

    def angle(self, other: Vector, degrees: bool = False) -> float:
        """Returns the angle between 2 vectors"""
        if not isinstance(other, Vector):
            return NotImplemented
        n1, n2 = self.norm(), other.norm()
        if n1 < EPS or n2 < EPS:
            raise ValueError("Angle is undefined for zero or near-zero vectors")
        cos = np.clip((self @ other) / (n1 * n2), -1.0, 1.0)
        angle_rad = np.arccos(cos)
        return float(np.degrees(angle_rad) if degrees else angle_rad)

    def project_onto(self, other: Vector) -> Vector:
        """Returns a projection of one vector in the direction of the other"""
        if not isinstance(other, Vector):
            return NotImplemented
        denom = other @ other
        if abs(denom) < EPS:
            raise ValueError("Cannot project onto zero vector.")
        scale = (self @ other) / denom
        return other * scale

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Vector:
        return cls(arr)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    def sum(self) -> float:
        return float(np.sum(self.values))

    def mean(self) -> float:
        if len(self) == 0:
            raise ValueError("mean is undefined for empty vector.")
        return self.sum() / len(self)


    def max(self) -> float:
        if len(self) == 0:
            raise ValueError("max is undefined for empty vector.")
        return np.max(self.values)

    def min(self) -> float:
        if len(self) == 0:
            raise ValueError("min is undefined for empty vector.")
        return np.min(self.values)

    def append(self, val: float | int) -> None:
        if not isinstance(val, (float, int)):
            raise TypeError("Only float values may be appended to Vectors.")
        self.values = np.append(self.values, float(val))




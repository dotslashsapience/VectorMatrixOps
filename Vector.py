from __future__ import annotations
import math
import numpy as np

EPS = 1e-12

class Vector:
    values: list[float]

    def __init__(self, values: list[float]) -> None:
        if not all(isinstance(v, (int, float)) for v in values):
            raise TypeError("All elements must be int or float")
        self.values = [float(v) for v in values]

    def __repr__(self) -> str:
        return f"Vector({self.values})"

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
            self.values[key] = [float(v) for v in val]
        else:
            raise TypeError("Indices must be integers or slices.")


    @classmethod
    def from_list(cls, lst) -> Vector:
        return Vector(lst)

    def tolist(self) -> list[float]:
        return self.values

    @property
    def shape(self) -> tuple[int]:
        return (len(self.values), )

    def __eq__(self, other: Vector) -> bool:
        if not isinstance(other, (Vector)) or len(self) != len(other):
            return False
        return all(math.isclose(a, b, abs_tol=EPS) for a,b in zip(self, other))


    def __add__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '+' operation. Vectors must be of equal length to add.")
            return Vector([a + b for a,b in zip(self, other)])
        elif isinstance(other, (float, int)):
            return Vector([val + other for val in self])
        return NotImplemented

    def __radd__(self, other: int | float) -> Vector:
        if isinstance(other, (float, int)):
            return self.__add__(other)
        return NotImplemented

    def __iadd__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '+=' operation. In place addition can only be conducted with Vectors of equal length.")
            self[:] = [a + b for a,b in zip(self, other)]
            return self
        elif isinstance(other, (float, int)):
            self[:] = [val + other for val in self]
            return self
        return NotImplemented

    def __sub__(self, other: Vector | float | int) -> Vector:
        if not isinstance(other, (float, int, Vector)):
            return NotImplemented
        elif isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vector subtraction can only be conducted on vectors of equal length.")
            return Vector([a - b for a,b in zip(self, other)])
        else:
            return Vector([val - other for val in self])


    def __isub__(self, other: float | int | Vector) -> Vector:
        if not isinstance(other, (float, int, Vector)):
            return NotImplemented
        elif isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid -= operation. In place subtraction can only be conducted with Vectors of equal length.")
            self[:] = [a - b for a,b in zip(self, other)]
            return self
        else:
            self[:] = [val - other for val in self]
            return self


    def __mul__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '*' operation. Vectors must be of equal length for element-wise multiplication.")
            return Vector([a * b for a,b in zip(self, other)])
        elif isinstance(other, (float, int)):
            return Vector([val * other for val in self])
        return NotImplemented

    def __rmul__(self, other: float | int) -> Vector:
        if isinstance(other, (float, int)):
            return self.__mul__(other)
        return NotImplemented

    def __imul__(self, other: float | int | Vector) -> Vector:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '*' operation. Vectors must be of equal length for element-wise multiplication.")
            self[:] = [a * b for a,b in zip(self, other)]
            return self
        elif isinstance(other, (float, int)):
            self[:] = [val * other for val in self]
            return self
        return NotImplemented

    def __matmul__(self, other: Vector) -> float:
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of equal length for dot product.")
            return sum(self[i] * other[i] for i in range(len(self)))
        return NotImplemented

    def __truediv__(self, other: float | int | Vector) -> Vector:
        if not isinstance(other, (float, int, Vector)):
            return NotImplemented
        elif isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vector division can only be conducted on vectors of equal length.")
            if all(abs(val) > EPS for val in other):
                return Vector([a / b for a,b in zip(self, other)])
            else:
                raise ZeroDivisionError("Division by Zero: zero values within vector.")
        elif abs(other) < EPS:
            raise ZeroDivisionError("Division by zero.")
        return Vector([val / other for val in self])


    def __itruediv__(self, other: float | int | Vector) -> Vector:
        if not isinstance(other, (float, int, Vector)):
            return NotImplemented
        elif isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vector division can only be conducted on vectors of equal length.")
            if all(abs(val) > EPS for val in other):
                self[:] = [a / b for a,b in zip(self, other)]
                return self
            else:
                raise ZeroDivisionError("Division by Zero: zero values within vector.")
        elif abs(other) < EPS:
            raise ZeroDivisionError("Division by zero.")
        self[:] = [val / other for val in self]
        return self


    def dot(self, other: Vector) -> float:
        return self @ other

    def norm(self, p: float = 2) -> float:
        """For most cases this is used to return the absolute value or the magnitude of the vector.
        There are additional uses in ML"""
        if not isinstance(p, (float, int)):
            return NotImplemented
        if p == float('inf'):
            return max(abs(val) for val in self) if len(self) else 0.0
        if p == 0:
            return float(sum(1 for val in self if abs(val) > EPS))
        if p <= 0:
            raise ValueError("norm: p must be > 0 (or 0/inf handled specially)")
        return sum(abs(val) ** p for val in self) ** (1/p)

    def normalized(self) -> Vector:
        """Returns the unit vector in the direction of the input vector"""
        n = self.norm(2)
        if n < EPS:
            raise ValueError("Cannot normalize a zero or near-zero vector.")
        return type(self)([val / n for val in self])

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
            raise ValueError("Angle is undefined for zero or near zero vectors")
        cos = (self @ other) / (n1 * n2)
        cos = max(-1.0, min(1.0, cos))
        if degrees:
            return math.acos(cos) * (180/math.pi)
        else:
            return math.acos(cos)

    def project_onto(self, other: Vector) -> Vector:
        """Returns a projection of one vector in the direction of the other"""
        if not isinstance(other, Vector):
            return NotImplemented
        denom = other @ other
        if denom < EPS:
            raise ValueError("Cannot project onto zero vector.")
        scale = (self @ other) / denom
        return type(self)([scale * val for val in other])

    def to_numpy(self) -> np.ndarray:
        return np.array(self.values, dtype=float)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Vector:
        return cls(arr.tolist())

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array(self.values, dtype=dtype)

    def sum(self) -> float:
        return math.fsum(val for val in self)

    def mean(self) -> float:
        if len(self) == 0:
            raise ValueError("mean is undefined for empty vector.")
        return self.sum() / len(self)


    def max(self) -> float:
        if len(self) == 0:
            raise ValueError("max is undefined for empty vector.")
        return max(self)

    def min(self) -> float:
        if len(self) == 0:
            raise ValueError("max is undefined for empty vector.")
        return min(self)






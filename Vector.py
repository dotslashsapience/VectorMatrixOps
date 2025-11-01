from __future__ import annotations
from dataclasses import dataclass
from typing import List

EPS = 1e-12

@dataclass(init=False, repr=False, eq=True, order=True, unsafe_hash=False,
           frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)
class Vector:
    values: list[float]

    def __init__(self, values: list[float]):
        if not all(isinstance(v, (int, float)) for v in values):
            raise TypeError("All elements must be int or float")
        for v in values:
            float(v)
        self.values = values

    def __repr__(self):
        return f"Vector({self.values})"

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key: int | slice):
        if isinstance(key, slice):
            return Vector(self.values[key])
        elif isinstance(key, int):
            return self.values[key]
        return NotImplemented

    def __setitem__(self, key: int | slice, val: float | list):
        if isinstance(key, int):
            self.values[key] = float(val)
        elif isinstance(key, slice):
            self.values[key] = [float(v) for v in val]
        else:
            raise TypeError("Indices must be integers or slices.")


    @classmethod
    def from_list(cls, lst):
        return Vector(lst)

    def tolist(self):
        return self.values

    @property
    def shape(self):
        return (len(self.values), )

    def __add__(self, other: float | int | Vector):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '+' operation. Vectors must be of equal length to add.")
            return Vector([a + b for a,b in zip(self, other)])
        elif isinstance(other, (float, int)):
            return Vector([val + other for val in self])
        return NotImplemented

    def __radd__(self, other: int | float):
        if isinstance(other, (float, int)):
            return self.__add__(other)
        return NotImplemented

    def __iadd__(self, other: float | int | Vector):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '+=' operation. In place addition can only be conducted with Vectors of equal length.")
            self[:] = [a + b for a,b in zip(self, other)]
            return self
        elif isinstance(other, (float, int)):
            self[:] = [val + other for val in self]
            return self
        return NotImplemented

    def __mul__(self, other: float | int | Vector):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '*' operation. Vectors must be of equal length for element-wise multiplication.")
            return Vector([a * b for a,b in zip(self, other)])
        elif isinstance(other, (float, int)):
            return Vector([val * other for val in self])
        return NotImplemented

    def __rmul__(self, other: float | int):
        if isinstance(other, (float, int)):
            return self.__mul__(other)
        return NotImplemented

    def __imul__(self, other: float | int | Vector):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Invalid '*' operation. Vectors must be of equal length for element-wise multiplication.")
            self[:] = [a * b for a,b in zip(self, other)]
            return self
        elif isinstance(other, (float, int)):
            self[:] = [val * other for val in self]
            return self
        return NotImplemented

    def __matmul__(self, other: Vector):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of equal length for dot product.")
            return sum(self[i] * other[i] for i in range(len(self)))
        return NotImplemented

    def __truediv__(self, other: float | int):
        if isinstance(other, (float, int)):
            if other == 0:
                raise ZeroDivisionError("Division by zero.")
            return Vector([val / other for val in self])
        return NotImplemented

    def __itruediv__(self, other: float | int):
        if isinstance(other, (float, int)):
            if other == 0:
                raise ZeroDivisionError("Division by zero.")
            self[:] = [val / other for val in self]
            return self
        return NotImplemented

    def dot(self, other: Vector):
        return self @ other

    def norm(self, p: float = 2):
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

vec = Vector([0.4, 0.5, 0.2, 0.3])
vec2 = Vector([5.4, 4.5, 3.3, 2.3])
#vec3 = Vector([0.4, 0.5, 0.1, 0.3])
#vec4 = Vector([0.4, 0.5, 0.2, 0.3])



vec3 = vec * vec2
print(vec3)


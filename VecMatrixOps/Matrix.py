#VecMatrixOps/Matrix.py
from __future__ import annotations
from typing import Callable
from .Vector import Vector
import random
import math
import numpy as np

EPS = 1e-12

class Matrix:
    rows = list[Vector]

    def __init__(self, rows: list[Vector[float]] | list[list[float]]) -> Matrix:
        if not rows:
            raise ValueError("Matrix cannot be empty.")
        if not all(len(rows[0]) == len(row) for row in rows):
            raise ValueError("All rows must be of same length.")
        if isinstance(rows[0], Vector):
            self.rows = [row for row in rows]
        elif isinstance(rows[0], list):
            self.rows = [Vector(row) for row in rows]
        else:
            raise TypeError("Rows must be list of lists or list of Vectors.")

    def __repr__(self) -> str:
        inner = ",\n ".join(row.raw_repr() for row in self.rows)
        return f"Matrix([\n {inner}\n])"

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key: int | slice | tuple[int, int]) -> Vector | slice | float:
        if isinstance(key, tuple):
            i, j = key
            return self.rows[i][j]
        elif isinstance(key, (int, slice)):
            return self.rows[key]
        else:
            return NotImplemented

    def __setitem__(self, key: int | slice | tuple[int, int], val: Vector[float] | list[float] | float) -> None:
        if isinstance(key, tuple):
            i, j = key
            self.rows[i][j] = val
        elif isinstance(key, (int, slice)):
            self.rows[key] = val
        else:
            raise TypeError(f"Invalid key type: {type(key).__name__}")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0])

    @classmethod
    def from_rows(cls, rows: list[Vector[float]] | list[list[float]]) -> Matrix:
        if not rows:
            raise ValueError("Rows cannot be empty.")
        if not all(len(rows[0]) == len(row) for row in rows):
            raise ValueError("All rows must be of equal length.")
        if len(rows[0]) == 0:
            raise ValueError("Matrices must have at least one column.")
        if isinstance(rows[0], list):
            rows = [Vector(row) for row in rows]
        elif not isinstance(rows[0], Vector):
                raise TypeError("Rows must be list of lists or list of Vectors.")
        return cls(rows)

    @classmethod
    def from_cols(cls, cols: list[list[float]] | list[Vector[float]]) -> Matrix:
        if not cols:
            raise ValueError("Columns cannot be empty.")
        if not all(len(cols[0]) == len(col) for col in cols):
            raise ValueError("All columns must be same length.")
        if len(cols[0]) == 0:
            raise ValueError("Matrices must have at least one row.")
        elif not isinstance(cols[0], (list, Vector)):
            raise TypeError("Columns must be list of lists or list of Vectors.")
        rows = [Vector([col[i] for col in cols]) for i in range(0, len(cols[0]))]
        return cls(rows)

    @classmethod
    def zeros(cls, r: int, c: int) -> Matrix:
        if not (isinstance(r, int) and isinstance(c, int)):
            raise TypeError("Rows and columns must be specified as integers.")
        if r < 1 or c < 1:
            raise ValueError("Matrix must have 1 or more rows and columns.")
        return cls([Vector([0 for _ in range(0, c)]) for _ in range(0, r)])

    @classmethod
    def ones(cls, r: int, c: int) -> Matrix:
        if not (isinstance(r, int) and isinstance(c, int)):
            raise TypeError("Rows and columns must be specified as integers.")
        if r < 1 or c < 1:
            raise ValueError("Matrix must have 1 or more rows and columns.")
        return cls([Vector([1 for _ in range(0, c)]) for _ in range(0, r)])

    @classmethod
    def eye(cls, n: int) -> Matrix:
        if not isinstance(n, int):
            raise TypeError("Matrix dimensions must be specified as an integer.")
        if n < 1:
            raise ValueError("Matrix must have 1 or more rows and columns.")
        mtrx = Matrix.zeros(n, n)
        for i in range(0,n):
            mtrx[i][i] = 1
        return mtrx

    @classmethod
    def random(cls, r: int, c: int, lo: float=0.0, hi: float=1.0) -> Matrix:
        if not (isinstance(r, int) and isinstance(c, int)):
            raise TypeError("Matrix dimensions must be specified as integers.")
        if r < 1 or c < 1:
            raise ValueError("Matrix must have 1 or more rows and columns.")
        if not (isinstance(lo, (float, int)) and isinstance(hi, (float, int))):
            raise TypeError("Max(hi) and min(lo) must be specified as floats.")
        if lo >= hi:
            raise ValueError("Min(lo) must be less than max(hi).")
        random.seed()
        return cls([Vector([random.uniform(lo, hi) for _ in range(0, c)]) for _ in range(0, r)])

    def __eq__(self, other: Matrix) -> bool:
        if not isinstance(other, Matrix):
            raise TypeError("Matrices can only be compared to other matrices.")
        if self.shape != other.shape:
            return False
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                if not math.isclose(self[i][j], other[i][j], abs_tol=EPS):
                    return False
        return True

    def sum(self) -> float:
        return math.fsum(val for row in self
                         for val in row)

    def row_sum(self, i: int) -> float:
        return math.fsum(val for val in self[i])

    def col_sum(self, j: int) -> float:
        return math.fsum(row[j] for row in self.rows)

    def trace(self) -> float:
        if len(self) != len(self.rows[0]):
            raise ValueError("Trace can only be conducted on square matrices.")
        return math.fsum(self[i][i] for i in range(0, len(self)))

    def frobenius(self) -> float:
        return math.sqrt(math.fsum((val ** 2) for row in self
                                   for val in row))

    @property
    def nrows(self) -> int:
        return len(self)

    @property
    def ncols(self) -> int:
        return len(self[0])

    def row(self, i: int) -> Vector:
        if not isinstance(i, int):
            raise TypeError("Row must be specified by an integer.")
        return self[i]

    def col(self, j: int) -> Vector:
        if not isinstance(j, int):
            raise TypeError("Column must be specified by an integer")
        return Vector([row[j] for row in self])

    def set_row(self, i: int, v: Vector[float] | list[float]) -> None:
        if not isinstance(i, int):
            raise TypeError("Row must be specified by an integer")
        if not isinstance(v, (Vector, list)):
            raise TypeError("Rows must be a list or Vector of floats")
        if len(v) != self.ncols:
            raise ValueError("Incorrect row length")
        if isinstance(v, list):
            v = Vector(v)
        self[i] = v

    def set_col(self, j: int, v: Vector[float] | list[float]) -> None:
        if not isinstance(j, int):
            raise TypeError("Column must be specified by an integer.")
        if not isinstance(v, (Vector, list)):
            raise TypeError("Columns must be Vector or list.")
        if not isinstance(v[0], (int, float)):
            raise TypeError("Column values must be floats.")
        if len(v) != self.nrows:
            raise ValueError("Incorrect column length.")
        for i, val in enumerate(v): #type: tuple[int, float]
            self[i][j] = val

    def transpose(self) -> Matrix:
        return Matrix.from_cols(self.rows)

    @property
    def T(self) -> Matrix:
        return self.transpose()

    def apply(self, fn: Callable[[float], float]) -> Matrix:
        if not callable(fn):
            raise TypeError("Argument must be a callable function.")
        return Matrix([[fn(val) for val in row]
                       for row in self])

    def __add__(self, other: Matrix | float | int) -> Matrix:
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same dimensions for element-wise addition.")
            return Matrix([
                Vector([a + b for a,b in zip(row_a, row_b)])
                for row_a,row_b in zip(self, other)])
        elif isinstance(other, (float, int)):
            return Matrix([Vector([(val + float(other)) for val in row])
                           for row in self])
        raise TypeError("Addition supported only between Matrix and scalar.")

    def __iadd__(self, other: Matrix | float | int) -> Matrix:
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same dimensions for in-place element-wise addition.")
            for row_a, row_b in zip(self, other):
                row_a[:] = [a + b for a,b in zip(row_a, row_b)]
            return self
        elif isinstance(other, (float, int)):
            for row in self:
                row[:] = [val + float(other) for val in row]
            return self
        raise TypeError("In place addition supported only between Matrix and scalars.")

    def __sub__(self, other: Matrix | float | int) -> Matrix:
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same dimensions for element-wise subtraction")
            return Matrix([
                Vector([a - b for a,b in zip(row_a, row_b)])
                for row_a, row_b in zip(self,other)])
        elif isinstance(other, (float, int)):
            return Matrix([
                Vector([val - float(other) for val in row])
                for row in self])
        raise TypeError("Subtraction only supported between Matrices and scalars.")

    def __isub__(self, other: Matrix | float | int) -> Matrix:
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have same dimensions for in place subtraction.")
            for row_a,row_b in zip(self, other):
                row_a[:] = [a-b for a,b in zip(row_a, row_b)]
            return self
        elif isinstance(other, (float, int)):
            for row in self:
                row[:] = [val - float(other) for val in row]
            return self
        raise TypeError("In-place subtraction supported only between Matrices and scalars.")

    def __mul__(self, other: Matrix | float | int) -> Matrix:
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have same dimensions for element-wise multiplication.")
            return Matrix([
                [a * b for a,b in zip(row_a, row_b)]
                for row_a,row_b in zip(self, other)])
        if isinstance(other, (float, int)):
            return Matrix([
                Vector([val * other for val in row])
                for row in self])
        raise TypeError("Multiplication only supported between matrices and scalars.")

    def __rmul__(self, other: float | int) -> Matrix:
        if isinstance(other, (float, int)):
            return self * other
        raise TypeError("Multiplication supported only between matrices and scalars.")

    def __truediv__(self, other: float | int) -> Matrix:
        if isinstance(other, (float, int)):
            return Matrix([
                Vector([val / other for val in row])
                for row in self])
        raise TypeError("True division only supported between matrix and scalar.")

    def __itruediv__(self, other: float | int) -> Matrix:
        if isinstance(other, (float, int)):
            for row in self:
                row[:] = [val / other for val in row]
            return self
        raise TypeError("In place division only supported between matrix and scalar.")

    def __matmul__(self, other: Matrix | Vector) -> Matrix | Vector:
        if isinstance(other, Matrix):
            if self.ncols != other.nrows:
                raise ValueError("Invalid shapes for Matrix multiplication.")
            result = []
            for i in range(0, self.nrows):
                row = []
                for j in range(0, other.ncols):
                    s = 0.0
                    for k in range(self.ncols):
                        s += self[i][k] * other[k][j]
                    row.append(s)
                result.append(Vector(row))
            return Matrix(result)
        elif isinstance(other, Vector):
            if self.ncols != len(other):
                raise ValueError("Matrix Columns must match Vector Length.")
            result = []
            for i in range(0, self.nrows):
                s = 0.0
                for j in range(0, self.ncols):
                    s += self[i][j] * other[j]
                result.append(s)
            return Vector(result)
        return NotImplemented

    def matvec(self, v: Vector) -> Vector:
        return self @ v

    def vecmat(self, v:) -> Vector:
        return (self.transpose @ v).transpose()

    def to_numpy(self) -> np.ndarray:
        """Return a NumPy ndarray copy of this Matrix."""
        return np.array([[val for val in row] for row in self], dtype=float)

    def from_numpy(self, arr: np.ndarray) -> Matrix:
        """Create an array from a NumPy ndarray"""
        if arr.ndim != 2:
            raise ValueError("Input array must be a 2-dimensional array to convert to Matrix.")
        return Matrix([Vector(row.tolist()) for row in arr])

    def __array__(self, dtype=None) -> np.ndarray:
        """Allow implicit conversion when passing Matrix to NumPy functions."""
        return np.array([[val for val in row] for row in self], dtype=dtype)




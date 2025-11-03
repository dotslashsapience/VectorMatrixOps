#VecMatrixOps/Matrix.py
from __future__ import annotations
from .Vector import Vector
import random
import math
import numpy as np

EPS = 1e-12

class Matrix:
    rows = list[Vector]

    def __init__(self, rows: list[Vector[float]] | list[list[float]]):
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

    def __setitem__(self, key: int | slice | tuple[int, int], val: Vector | list[float] | float) -> None:
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
        return math.fsum(val for row in self for val in row)

    def row_sum(self, i: int) -> float:
        return math.fsum(val for val in self[i])

    def col_sum(self, j: int) -> float:
        return math.fsum(row[j] for row in self.rows)

rows = [[2, 3, 5], [2 ,5 , 9], [9, 2, 5], [9, 2, 3]]
mtrx = Matrix(rows)
mtrx1 = Matrix.from_rows(rows)
mtrx2 = Matrix.from_rows(rows)
print(mtrx1.col_sum(0))

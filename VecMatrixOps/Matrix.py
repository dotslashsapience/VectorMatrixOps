#VecMatrixOps/Matrix.py
from __future__ import annotations
from .Vector import Vector
import math
import numpy as np

EPS = 1e-12

class Matrix:
    rows = list[Vector]

    def __init__(self, rows: list[Vector]):
        if not rows:
            raise ValueError("Matrix cannot be empty.")
        elif not all(len(rows[0]) == len(row) for row in rows):
            raise ValueError("All rows must be of same length.")
        else:
            self.rows = [Vector(row) for row in rows]

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
        # need to add a rectangularity check
        if not rows:
            raise ValueError("Rows cannot be empty.")
        if isinstance(rows[0], list):
            rows = [Vector(r) for r in rows]
        elif not isinstance(rows[0], Vector):
                raise TypeError("Rows must be list of lists or list of Vectors.")
        return cls(rows)

    @classmethod
    def from_cols(cls, cols: list[list[float]] | list[Vector[float]]) -> Matrix:
        #add a rectangularity check
        if not cols:
            raise ValueError("Cols can not be empty.")
        elif not isinstance(cols[0], (list, Vector)):
            raise TypeError("Columns must be list of lists or list of Vectors.")
        cols = [Vector([col[i] for col in cols]) for i in range(0, len(cols[0]))]
        return cls(cols)


rows = [[2, 3, 5], [2 ,5 , 9], [9, 2, 5], [9, 2, 3]]
mtrx1 = Matrix.from_rows(rows)
mtrx2 = Matrix.from_cols(rows)
print(mtrx1)
print(mtrx2)
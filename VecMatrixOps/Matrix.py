#VecMatrixOps/Matrix.py
from __future__ import annotations
from typing import Callable, Any
from .Vector import Vector
import numpy as np
from numpy.typing import NDArray

EPS = 1e-12

class Matrix:
    __array_priority__ = 1000
    _a: NDArray[np.float64]

    def __init__(self, data: list[list[float]] | np.ndarray | Matrix, *, copy: bool=True, dtype=float):
        if isinstance(data, Matrix):
            arr = data._a
        else:
            arr = np.asarray(data, dtype=dtype)
            if arr.ndim != 2:
                raise ValueError("Matrix must be 2D")
        if arr.dtype == dtype:
            self._a = arr.copy() if copy else arr
        else:
            self._a = arr.astype(dtype)

    def dtype(self):
        return self._a.dtype

    def __repr__(self) -> str:
        body = np.array2string(self._a, separator=", ")
        return f"Matrix(shape={self._a.shape}, dtype={self._a.dtype})\n{body}"

    def __len__(self) -> int:
        return len(self._a)

    def __iter__(self):
        for row in self._a:
            yield row

    def __getitem__(self, key: int | slice | tuple[int, int]) -> Vector | float | Matrix:
        out = self._a[key]
        if isinstance(out, np.ndarray):
            if out.ndim == 2:
                return Matrix(out, copy=False)
            elif out.ndim == 1:
                return Vector(out)
        return float(out)

    def __setitem__(self, key: int | slice | tuple[int, int], val: Vector[float] | list[float] | float | Matrix) -> None:
        if isinstance(val, Matrix):
            self._a[key] = val._a
        elif isinstance(val, Vector):
            self._a[key] = val.values
        else:
            self._a[key] = val

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(map(int, self._a.shape))

    @classmethod
    def from_rows(cls, rows: list[Vector[float]] | list[list[float]]) -> Matrix:
        return cls(np.asarray(rows, dtype=np.float64), copy=True)

    @classmethod
    def from_cols(cls, cols: list[list[float]] | list[Vector[float]]) -> Matrix:
        return cls(np.asarray(cols, dtype=np.float64).T, copy=True)

    @classmethod
    def zeros(cls, r: int, c: int) -> Matrix:
        if r < 1 or c < 1: raise ValueError("r and c must be >= 1.")
        return cls(np.zeros((r, c), dtype=np.float64), copy=False)

    @classmethod
    def ones(cls, r: int, c: int) -> Matrix:
        if r < 1 or c < 1: raise ValueError("r and c must be >= 1.")
        return cls(np.ones((r, c), dtype=np.float64), copy=False)

    @classmethod
    def eye(cls, n: int) -> Matrix:
        if n < 1: raise ValueError("n must be >= 1.")
        return cls(np.eye(n, dtype=np.float64), copy=False)

    @classmethod
    def random(cls, r: int, c: int, lo: float=0.0, hi: float=1.0) -> Matrix:
        if not (hi > lo): raise ValueError("hi must be > lo")
        rng = np.random.default_rng()
        return cls(rng.uniform(lo, hi, size=(r, c)).astype(float), copy=False)

    def __eq__(self, other: Matrix) -> bool:
        if not isinstance(other, Matrix):
            return False
        if self.shape != other.shape:
            return False
        return bool(np.allclose(self._a, other._a, rtol=0.0, atol=EPS))

    def sum(self) -> float:
        return float(np.sum(self._a))

    def row_sum(self, i: int) -> float:
        return float(np.sum(self._a[i, :]))

    def col_sum(self, j: int) -> float:
        return float(np.sum(self._a[:, j]))

    def trace(self) -> float:
        if self.nrows != self.ncols:
            raise ValueError("Trace requires square Matrix.")
        return float(np.trace(self._a))

    def frobenius(self) -> float:
        return float(np.linalg.norm(self._a, ord="fro"))

    @property
    def nrows(self) -> int:
        return int(self._a.shape[0])

    @property
    def ncols(self) -> int:
        return int(self._a.shape[1])

    def row(self, i: int) -> Vector:
        if not isinstance(i, int):
            raise TypeError("Row must be specified by an integer.")
        return self[i]

    def col(self, j: int) -> Vector:
        if not isinstance(j, int):
            raise TypeError("Column must be specified by an integer")
        return Vector(self._a[:, j])

    def set_row(self, i: int, v: Vector[float] | list[float]) -> None:
        self._a[i, :] = v.values if isinstance(v, Vector) else v

    def set_col(self, j: int, v: Vector[float] | list[float]) -> None:
        self._a[:, j] = v.values if isinstance(v, Vector) else v

    def transpose(self) -> Matrix:
        return self.T

    @property
    def T(self) -> Matrix:
        return Matrix(self._a.T, copy=False)

    def apply(self, fn: Callable[[float], float]) -> Matrix:
        vfn = np.vectorize(fn, otypes=[float])
        return Matrix(vfn(self._a), copy=False)

    def _wrap_bin(self, other, op):
        b = other._a if isinstance(other, Matrix) else other
        return Matrix(op(self._a, b), copy=False)

    def __add__(self, other: Matrix | float | int) -> Matrix:
        return self._wrap_bin(other, np.add)

    def __iadd__(self, other: Matrix | float | int) -> Matrix:
        np.add(self._a, other._a if isinstance(other, Matrix) else other, out=self._a)
        return self

    def __sub__(self, other: Matrix | float | int) -> Matrix:
        return self._wrap_bin(other, np.subtract)

    def __isub__(self, other: Matrix | float | int) -> Matrix:
        np.subtract(self._a, other._a if isinstance(other, Matrix) else other, out=self._a)
        return self

    def __mul__(self, other: Matrix | float | int) -> Matrix:
        return self._wrap_bin(other, np.multiply)

    def __rmul__(self, other: float | int) -> Matrix:
        return Matrix(self._a * other)

    def __imul__(self, other: float | int | Matrix) -> Matrix:
        np.multiply(self._a, other._a if isinstance(other, Matrix) else other, out=self._a)
        return self

    def __truediv__(self, other: float | int | Matrix) -> Matrix:
        return self._wrap_bin(other, np.divide)

    def __itruediv__(self, other: float | int | Matrix) -> Matrix:
        np.divide(self._a, other._a if isinstance(other, Matrix) else other, out=self._a)
        return self

    def __matmul__(self, other: Matrix | Vector) -> Matrix | Vector:
        if isinstance(other, Matrix):
            return Matrix(self._a @ other._a, copy=False)
        if isinstance(other, Vector):
            return Vector(self._a @ other.values)
        return NotImplemented

    def matvec(self, v: Vector) -> Vector:
        return Vector(self._a @ v.values)

    def vecmat(self, v: Vector) -> Vector:
        return Vector(v.values @ self._a)

    def to_numpy(self, *, copy: bool=True) -> np.ndarray:
        """Return a NumPy ndarray copy of this Matrix."""
        return self._a.copy() if copy else self._a

    @classmethod
    def from_numpy(cls, arr: np.ndarray, *, copy: bool=False) -> Matrix:
        """Create an array from a NumPy ndarray"""
        if arr.ndim != 2:
            raise ValueError("Input array must be a 2-dimensional array to convert to Matrix.")
        return cls(arr, copy=copy)

    def __array__(self, dtype=None) -> np.ndarray:
        """Allow implicit conversion when passing Matrix to NumPy functions."""
        return np.asarray(self._a, dtype=dtype)

    def copy(self) -> Matrix:
        return Matrix(self._a.copy(), copy=False)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__":
            return NotImplemented

        processed_inputs = []
        for x in inputs:
            if isinstance(x, Matrix):
                processed_inputs.append(x._a)
            elif isinstance(x, Vector):
                processed_inputs.append(x.values)
            else:
                processed_inputs.append(x)

        out = kwargs.get('out', None)
        if out:
            processed_out = []
            for o in out:
                if isinstance(o, Matrix):
                    processed_out.append(o._a)
                elif isinstance(o, Vector):
                    processed_out.append(o.values)
                else:
                    processed_out.append(o)
            kwargs['out'] = tuple(processed_out)

        results = ufunc(*processed_inputs, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            return self._wrap_result(results)
        else:
            return tuple(self._wrap_result(res) for res in results)

    @staticmethod
    def _wrap_result(res: Any) -> Any:
        """Helper for __array_ufunc__ to wrap results."""
        if isinstance(res, np.ndarray):
            if res.ndim == 2:
                return Matrix(res, copy=False)
            elif res.ndim == 1:
                return Vector(res, copy=False)
        return res




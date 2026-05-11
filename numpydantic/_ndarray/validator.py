"""Internal validation logic for NDArray values."""

from typing import Any, Optional, Tuple
import numpy as np
from .constants import ARRAY_KEY
from .metadata import metadata_to_array


class NDArrayValidator:
    """Callable validator for NumPy arrays with optional dtype/shape constraints.

    Shape entries of ``None`` match any size on that axis: ``shape=(None, 3)``
    accepts any 2D array whose second axis has length 3.

    Only numeric dtypes (bool, int, uint, float, complex) are supported; string,
    object, and datetime arrays are rejected.
    """

    NUMERIC_KINDS = "biufc"

    def __init__(
        self,
        dtype: Optional[str] = None,
        shape: Optional[Tuple[Optional[int], ...]] = None,
    ):
        self.dtype = None
        if dtype is not None:
            self.dtype = np.dtype(dtype)
            self._require_numeric_dtype(self.dtype)
        self.shape = shape

    def __call__(self, v: Any) -> np.ndarray:
        arr = self._to_numpy_array(v)
        self._require_numeric_dtype(arr.dtype)
        self._validate_shape(arr)
        return arr

    @staticmethod
    def _require_numeric_dtype(dtype: np.dtype):
        if dtype.kind not in NDArrayValidator.NUMERIC_KINDS:
            raise TypeError(
                f"Unsupported dtype {dtype!r}: only numeric dtypes "
                "(bool, int, uint, float, complex) are supported."
            )

    def _to_numpy_array(self, v: Any) -> np.ndarray:
        if isinstance(v, dict) and ARRAY_KEY in v:
            return metadata_to_array(v)
        if isinstance(v, (np.ndarray, list, tuple)):
            try:
                return np.array(v, dtype=self.dtype)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Cannot convert input to dtype {self.dtype}.") from e
        raise TypeError(
            f"Expected numpy.ndarray, list, tuple or metadata dict; got {type(v).__name__}"
        )

    def _validate_shape(self, arr: np.ndarray):
        if self.shape is None:
            return
        if arr.ndim != len(self.shape):
            raise ValueError(
                f"Expected {len(self.shape)}D array matching shape {self.shape}, "
                f"got {arr.ndim}D array with shape {arr.shape}"
            )
        for axis, (expected, actual) in enumerate(zip(self.shape, arr.shape)):
            if expected is not None and expected != actual:
                raise ValueError(
                    f"Expected shape {self.shape}, got {arr.shape} "
                    f"(axis {axis}: expected {expected}, got {actual})"
                )

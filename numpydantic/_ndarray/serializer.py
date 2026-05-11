"""Internal serialization logic for NDArray values."""

import base64
from typing import Literal, Union
import numpy as np
from pydantic import SerializationInfo
from .constants import (
    ARRAY_KEY,
    BASE64_ENCODING,
    DTYPE_KEY,
    ENCODING_KEY,
    SHAPE_KEY,
)


class NDArraySerializer:
    """Callable serializer for NumPy arrays."""

    def __init__(self, encoding: Literal["list", "base64"] = "list"):
        """Initialize serializer with encoding mode."""
        self.encoding = encoding

    def __call__(self, v: np.ndarray, info: SerializationInfo) -> Union[list, dict]:
        """Serialize numpy array to list or dict with metadata."""
        if self.encoding == BASE64_ENCODING:
            # Ensure row-major memory layout so tobytes() round-trips correctly.
            contiguous = np.ascontiguousarray(v)
            return {
                ARRAY_KEY: base64.b64encode(contiguous.tobytes(order="C")).decode(
                    "ascii"
                ),
                DTYPE_KEY: str(contiguous.dtype),
                SHAPE_KEY: list(contiguous.shape),
                ENCODING_KEY: BASE64_ENCODING,
            }

        if info.round_trip:
            return {
                ARRAY_KEY: v.tolist(),
                DTYPE_KEY: str(v.dtype),
                SHAPE_KEY: list(v.shape),
            }

        return v.tolist()

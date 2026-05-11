"""Internal constants and type aliases for NDArray serialization."""

from typing import Literal

ARRAY_KEY = "array"
DTYPE_KEY = "dtype"
SHAPE_KEY = "shape"
ENCODING_KEY = "encoding"

LIST_ENCODING = "list"
BASE64_ENCODING = "base64"
NDArrayEncoding = Literal["list", "base64"]

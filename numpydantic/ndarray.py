"""Public NDArray API."""

from typing import Literal, Optional, Tuple
import numpy as np
from typing_extensions import Annotated
from ._ndarray.spec import _NDArraySpec


def NDArray(
    dtype: Optional[str] = None,
    shape: Optional[Tuple[Optional[int], ...]] = None,
    encoding: Literal["list", "base64"] = "list",
) -> type:
    """Build a Pydantic-compatible NumPy array type.

    Args:
        dtype: NumPy dtype name (e.g. ``"float64"``). ``None`` accepts any dtype.
        shape: Expected shape as a tuple. ``None`` entries match any size on
            that axis (e.g. ``(None, 3)`` matches any 2D array with 3 columns).
        encoding: ``"list"`` (default) emits human-readable nested lists.
            ``"base64"`` emits a metadata envelope with a base64-encoded
            payload. Compact and bit-exact (NaN/inf and full float
            precision preserved).

    Returns:
        An ``Annotated[np.ndarray, _NDArraySpec(...)]`` suitable for use as
        a Pydantic model field.

    Notes:
        Round-trip serialization is driven by Pydantic's ``round_trip`` flag
        on ``model_dump`` / ``model_dump_json``. In ``"list"`` mode, plain
        dumps emit nested lists and ``round_trip=True`` switches to a metadata
        envelope so dtype/shape can be recovered. ``"base64"`` mode always
        emits the envelope regardless of ``round_trip``.

    Examples:
        Defining fields::

            >>> NDArray()                                  # any array
            >>> NDArray(dtype="float64")                   # dtype-only
            >>> NDArray(shape=(2, 3))                      # exact shape
            >>> NDArray(shape=(None, 3))                   # any first dim, 3 cols
            >>> NDArray(dtype="float32", encoding="base64")

        Round-trippable JSON dump::

            >>> model.model_dump_json(round_trip=True)
    """

    spec = _NDArraySpec(dtype=dtype, shape=shape, encoding=encoding)
    return Annotated[np.ndarray, spec]


__all__ = ["NDArray"]

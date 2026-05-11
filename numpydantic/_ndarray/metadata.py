"""Internal metadata payload parsing helpers for NDArray serialization."""

import base64
import binascii

import numpy as np

from .constants import ARRAY_KEY, BASE64_ENCODING, DTYPE_KEY, ENCODING_KEY, SHAPE_KEY


def metadata_to_array(v: dict) -> np.ndarray:
    """Deserialize metadata dict with base64 or list payload."""
    payload = v[ARRAY_KEY]
    if DTYPE_KEY in v:
        try:
            target_dtype = np.dtype(v[DTYPE_KEY])
        except TypeError as e:
            raise ValueError(f"Invalid dtype {v[DTYPE_KEY]!r} in metadata") from e
    else:
        target_dtype = None

    if isinstance(payload, str):
        if v.get(ENCODING_KEY) != BASE64_ENCODING:
            raise ValueError("String array payload requires encoding='base64'")
        if target_dtype is None or SHAPE_KEY not in v:
            raise ValueError("Base64 metadata requires dtype and shape")
        try:
            raw = base64.b64decode(payload.encode("ascii"), validate=True)
        except (ValueError, binascii.Error) as e:
            raise ValueError("Invalid base64 array payload") from e
        try:
            return (
                np.frombuffer(raw, dtype=target_dtype)
                .reshape(tuple(v[SHAPE_KEY]))
                .copy()
            )
        except (TypeError, ValueError) as e:
            raise ValueError("Invalid dtype/shape for base64 array payload") from e

    try:
        arr = np.asarray(payload, dtype=target_dtype)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Could not convert list payload to ndarray (dtype={target_dtype}): {e}"
        ) from e

    # Honor the declared shape so empty / zero-dim arrays survive the round-trip.
    # Without this, np.asarray([]) always yields shape (0,) regardless of metadata.
    if SHAPE_KEY in v:
        try:
            arr = arr.reshape(tuple(v[SHAPE_KEY]))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Could not reshape list payload to {v[SHAPE_KEY]}: {e}"
            ) from e
    return arr

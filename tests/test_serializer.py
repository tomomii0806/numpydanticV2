"""Tests for NDArray serialization behaviors."""

import base64

import numpy as np

from tests.conftest import Base64FloatMatrixModel, DefaultArrayModel, DefaultBase64Model


def test_list_mode_default_dump_emits_bare_list(
    array_2d_float: np.ndarray,
    list_2d_float: list[list[float]],
):
    """Default dump emits a bare nested list — dtype/shape are NOT preserved."""
    dumped = DefaultArrayModel(array=array_2d_float).model_dump()

    assert dumped["array"] == list_2d_float
    assert not isinstance(dumped["array"], dict)
    assert "dtype" not in dumped
    assert "shape" not in dumped


def test_list_mode_default_dump_loses_dtype_information():
    """Without round_trip, dtype is lost: re-validated array has a different dtype."""
    original = DefaultArrayModel(array=np.array([1, 2, 3], dtype=np.int8))

    dumped = original.model_dump()
    restored = DefaultArrayModel.model_validate(dumped)

    # Values survive the round trip...
    np.testing.assert_array_equal(restored.array, original.array)
    # ...but dtype does not — int8 is promoted to numpy's platform default int.
    assert original.array.dtype != restored.array.dtype
    assert restored.array.dtype == np.int64


def test_list_mode_round_trip_preserves_metadata():
    """round_trip=True preserves the exact dtype that the default dump loses."""
    original = DefaultArrayModel(array=np.array([1, 2, 3], dtype=np.int8))

    dumped = original.model_dump(round_trip=True)
    restored = DefaultArrayModel.model_validate(dumped)

    # Envelope carries dtype + shape...
    assert dumped["array"]["dtype"] == "int8"
    assert dumped["array"]["shape"] == [3]
    # ...so the restored array matches the original exactly, dtype included.
    np.testing.assert_array_equal(restored.array, original.array)
    assert restored.array.dtype == original.array.dtype == np.int8


def test_list_mode_non_contiguous_array(
    array_2d_float: np.ndarray,
):
    """Serializes non-contiguous arrays to nested lists."""
    non_contiguous = array_2d_float.T
    assert not non_contiguous.flags["C_CONTIGUOUS"]

    dumped = DefaultArrayModel(array=non_contiguous).model_dump()

    assert dumped["array"] == non_contiguous.tolist()


def test_list_mode_default_empty_array():
    """Default dump of an empty array is just an empty list."""
    empty = np.array([], dtype=np.float64)

    dumped = DefaultArrayModel(array=empty).model_dump()

    assert dumped["array"] == []


def test_list_mode_empty_array_round_trip():
    """Serializes empty arrays with list-mode metadata for round trips."""
    empty = np.array([], dtype=np.float64)

    dumped = DefaultArrayModel(array=empty).model_dump(round_trip=True)

    assert dumped["array"]["array"] == []
    assert dumped["array"]["dtype"] == "float64"
    assert dumped["array"]["shape"] == [0]


def test_base64_mode_serialization(
    array_2d_float: np.ndarray,
    metadata_base64_payload: dict,
):
    """Serializes arrays in base64 mode with a metadata envelope and exact bytes."""
    dumped = Base64FloatMatrixModel(matrix=array_2d_float).model_dump()["matrix"]

    # Envelope structure matches the expected payload.
    assert dumped == metadata_base64_payload

    # Decode the base64 string independently and verify the bytes reconstruct
    # the original array. This catches dtype/shape/byte-order regressions.
    decoded_bytes = base64.b64decode(dumped["array"])
    reconstructed = np.frombuffer(
        decoded_bytes, dtype=np.dtype(dumped["dtype"])
    ).reshape(dumped["shape"])

    np.testing.assert_array_equal(reconstructed, array_2d_float)
    assert reconstructed.dtype == array_2d_float.dtype


def test_base64_mode_non_contiguous_array(
    array_2d_float: np.ndarray,
):
    """Encodes non-contiguous arrays so reconstruction yields the original values."""
    non_contiguous = array_2d_float.T
    assert not non_contiguous.flags["C_CONTIGUOUS"]

    dumped = Base64FloatMatrixModel(matrix=non_contiguous).model_dump()["matrix"]

    # Independently decode and reconstruct from the envelope.
    decoded_bytes = base64.b64decode(dumped["array"])
    reconstructed = np.frombuffer(
        decoded_bytes, dtype=np.dtype(dumped["dtype"])
    ).reshape(dumped["shape"])

    # Reconstructured array should preserve the non-contiguous array.
    np.testing.assert_array_equal(reconstructed, non_contiguous)
    assert dumped["encoding"] == "base64"


def test_base64_mode_empty_array():
    """Serializes empty arrays to base64 metadata payloads."""

    empty = np.array([], dtype=np.uint8)
    dumped = DefaultBase64Model(array=empty).model_dump()["array"]

    assert dumped["array"] == ""
    assert dumped["dtype"] == "uint8"
    assert dumped["shape"] == [0]
    assert dumped["encoding"] == "base64"


def test_base64_mode_ignores_round_trip_flag(
    array_2d_float: np.ndarray,
):
    """base64 mode emits the same envelope whether round_trip is True or False."""
    model = Base64FloatMatrixModel(matrix=array_2d_float)

    default_dump = model.model_dump()
    round_trip_dump = model.model_dump(round_trip=True)

    assert default_dump == round_trip_dump

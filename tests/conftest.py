import base64

import numpy as np
import pytest
from pydantic import BaseModel, create_model
from numpydantic import NDArray

NDARRAY_ANY = NDArray()
NDArray_BASE64 = NDArray(encoding="base64")
NDARRAY_FLOAT64 = NDArray(dtype="float64")
NDARRAY_FLOAT64_2X2 = NDArray(dtype="float64", shape=(2, 2))
NDARRAY_FLOAT64_2X2_BASE64 = NDArray(dtype="float64", shape=(2, 2), encoding="base64")


class DefaultArrayModel(BaseModel):
    array: NDARRAY_ANY


class Float64MatrixModel(BaseModel):
    matrix: NDARRAY_FLOAT64_2X2


class Base64FloatMatrixModel(BaseModel):
    matrix: NDARRAY_FLOAT64_2X2_BASE64


class DefaultBase64Model(BaseModel):
    array: NDArray_BASE64


@pytest.fixture()
def array_1d_int() -> np.ndarray:
    return np.array([1, 2, 3], dtype=np.int32)


@pytest.fixture()
def array_2d_float() -> np.ndarray:
    return np.array([[1.0, 2.5], [3.5, 4.5]], dtype=np.float64)


@pytest.fixture()
def array_3d_uint8() -> np.ndarray:
    return np.arange(24, dtype=np.uint8).reshape(2, 3, 4)


@pytest.fixture()
def list_2d_float(array_2d_float: np.ndarray) -> list[list[float]]:
    return array_2d_float.tolist()


@pytest.fixture()
def tuple_2d_float(list_2d_float: list[list[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(row) for row in list_2d_float)


@pytest.fixture()
def metadata_list_payload(array_2d_float: np.ndarray) -> dict:
    return {
        "array": array_2d_float.tolist(),
        "dtype": str(array_2d_float.dtype),
        "shape": list(array_2d_float.shape),
    }


@pytest.fixture()
def metadata_base64_payload(array_2d_float: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array_2d_float)
    payload = base64.b64encode(contiguous.tobytes(order="C")).decode("ascii")
    return {
        "array": payload,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "encoding": "base64",
    }


@pytest.fixture()
def ndarray_any_type() -> type:
    return NDARRAY_ANY


@pytest.fixture()
def ndarray_float64_matrix_base64_type() -> type:
    return NDARRAY_FLOAT64_2X2_BASE64


@pytest.fixture()
def ndarray_model_factory():
    def _factory(field_type: type, field_name: str = "array") -> type[BaseModel]:
        return create_model("NDArrayModel", **{field_name: (field_type, ...)})

    return _factory

"""Integration tests for NDArray fields in Pydantic models."""

from typing import Optional
import numpy as np
import pytest
from pydantic import BaseModel, ValidationError, create_model
from tests.conftest import (
    Base64FloatMatrixModel,
    DefaultArrayModel,
    Float64MatrixModel,
)
from numpydantic import NDArray


class TestInputValidation:
    """Accepted input forms: arrays, lists, tuples, metadata payloads."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["array_2d_float", "list_2d_float", "tuple_2d_float", "metadata_list_payload"],
    )
    def test_validates_supported_inputs(
        self,
        fixture_name: str,
        request: pytest.FixtureRequest,
        array_2d_float: np.ndarray,
    ):
        """Validates ndarray, list, tuple, and metadata-dict inputs."""
        value = request.getfixturevalue(fixture_name)

        model = DefaultArrayModel.model_validate({"array": value})

        assert isinstance(model.array, np.ndarray)
        np.testing.assert_allclose(model.array, array_2d_float)

    def test_validates_3d_array(self, array_3d_uint8: np.ndarray):
        """Higher-rank arrays validate and preserve shape/dtype."""
        model = DefaultArrayModel.model_validate({"array": array_3d_uint8})

        assert model.array.shape == (2, 3, 4)
        assert model.array.dtype == np.uint8
        np.testing.assert_array_equal(model.array, array_3d_uint8)

    def test_validates_empty_array(self):
        """Empty arrays validate and keep their dtype/shape."""
        empty = np.array([], dtype=np.float64)

        model = DefaultArrayModel.model_validate({"array": empty})

        assert model.array.shape == (0,)
        assert model.array.dtype == np.float64


class TestDtypeConstraints:
    """Behavior of dtype-constrained NDArray fields."""

    def test_dtype_casts_compatible_values(
        self, ndarray_model_factory, array_1d_int: np.ndarray
    ):
        """Casts compatible inputs (int → float64) without error."""
        model_cls = ndarray_model_factory(NDArray(dtype="float64"))

        model = model_cls.model_validate({"array": array_1d_int})

        assert model.array.dtype == np.dtype("float64")
        np.testing.assert_allclose(model.array, array_1d_int.astype("float64"))

    def test_dtype_cast_failure_raises_validation_error(self, ndarray_model_factory):
        """Inputs that cannot be cast to the declared dtype surface as ValidationError."""
        model_cls = ndarray_model_factory(NDArray(dtype="float64"))

        with pytest.raises(ValidationError):
            model_cls.model_validate({"array": ["abc", "def"]})

    def test_non_numeric_dtype_rejected_at_model_definition(self):
        """Non-numeric dtypes (U/S/O/M/m) are rejected when the validator is built."""
        with pytest.raises(TypeError, match="Unsupported dtype"):

            class _M(BaseModel):
                x: NDArray(dtype="U5")


class TestShapeConstraints:
    """Behavior of shape-constrained NDArray fields, including partially-open shapes."""

    def test_shape_mismatch_rejected(self, array_1d_int: np.ndarray):
        """Rejects payloads that violate the declared shape."""
        with pytest.raises(ValidationError):
            Float64MatrixModel.model_validate({"matrix": array_1d_int})

    def test_partial_shape_concrete_dim_enforced(self):
        """When shape mixes None and concrete dims, the concrete dims still apply."""

        class RowsModel(BaseModel):
            rows: NDArray(dtype="float64", shape=(None, 3))

        valid = RowsModel(rows=np.zeros((5, 3)))
        assert valid.rows.shape == (5, 3)

        with pytest.raises(ValidationError, match="axis 1: expected 3, got 4"):
            RowsModel(rows=np.zeros((5, 4)))

    def test_partial_shape_rank_still_enforced(self):
        """Shapes with `None` entries still enforce rank — a 1D array fails a 2D spec."""

        class RowsModel(BaseModel):
            rows: NDArray(dtype="float64", shape=(None, 3))

        with pytest.raises(ValidationError):
            RowsModel(rows=np.zeros(3))

    def test_shape_constrained_round_trip_via_json(self, array_2d_float: np.ndarray):
        """Shape-constrained fields round-trip cleanly through JSON."""
        model = Float64MatrixModel(matrix=array_2d_float)

        payload = model.model_dump_json(round_trip=True)
        restored = Float64MatrixModel.model_validate_json(payload)

        assert restored.matrix.shape == (2, 2)
        assert restored.matrix.dtype == np.float64
        np.testing.assert_allclose(restored.matrix, array_2d_float)

    def test_zero_dim_shape_round_trip_via_json(self):
        """Shapes containing 0 (typed empty containers) round-trip via JSON.

        Regression: list-mode metadata previously ignored the shape field, so
        an empty payload always deserialized to (0,) regardless of the declared
        rank, breaking shape=(0, N) specs.
        """

        class EmptyRowsModel(BaseModel):
            rows: NDArray(dtype="float64", shape=(0, 3))

        original = np.zeros((0, 3))
        model = EmptyRowsModel(rows=original)

        payload = model.model_dump_json(round_trip=True)
        restored = EmptyRowsModel.model_validate_json(payload)

        assert restored.rows.shape == (0, 3)
        assert restored.rows.dtype == np.float64


class TestEncodingModes:
    """List vs base64 serialization round-trips and cross-encoding behavior."""

    def test_list_mode_round_trip_via_json(
        self,
        array_2d_float: np.ndarray,
        list_2d_float: list[list[float]],
        metadata_list_payload: dict,
    ):
        """List mode round-trips via JSON, with and without round_trip=True."""
        model = DefaultArrayModel(array=array_2d_float)

        dumped = model.model_dump()
        assert dumped["array"] == list_2d_float

        dumped_round_trip = model.model_dump(round_trip=True)
        assert dumped_round_trip["array"] == metadata_list_payload

        revalidated = DefaultArrayModel.model_validate(dumped_round_trip)
        np.testing.assert_allclose(revalidated.array, array_2d_float)

        json_payload = model.model_dump_json(round_trip=True)
        revalidated_json = DefaultArrayModel.model_validate_json(json_payload)
        np.testing.assert_allclose(revalidated_json.array, array_2d_float)

    def test_base64_mode_round_trip_via_json(
        self,
        array_2d_float: np.ndarray,
        metadata_base64_payload: dict,
    ):
        """Base64 mode round-trips via JSON with bit-exact bytes."""
        model = Base64FloatMatrixModel(matrix=array_2d_float)

        dumped = model.model_dump()
        assert dumped["matrix"] == metadata_base64_payload

        revalidated = Base64FloatMatrixModel.model_validate(dumped)
        np.testing.assert_allclose(revalidated.matrix, array_2d_float)

        json_payload = model.model_dump_json()
        revalidated_json = Base64FloatMatrixModel.model_validate_json(json_payload)
        np.testing.assert_allclose(revalidated_json.matrix, array_2d_float)

    def test_base64_preserves_nan_and_inf(self):
        """Base64 round-trip keeps NaN/Inf values exactly (motivation for the mode)."""

        class ExactModel(BaseModel):
            data: NDArray(dtype="float64", encoding="base64")

        original = np.array([np.nan, np.inf, -np.inf, 1.5])
        model = ExactModel(data=original)

        restored = ExactModel.model_validate_json(model.model_dump_json())
        assert np.array_equal(restored.data, original, equal_nan=True)

    def test_empty_array_round_trip_via_json(self):
        """Empty arrays survive a JSON round-trip with dtype and shape preserved."""
        empty = np.array([], dtype=np.float64)
        model = DefaultArrayModel(array=empty)

        payload = model.model_dump_json(round_trip=True)
        restored = DefaultArrayModel.model_validate_json(payload)

        assert restored.array.shape == (0,)
        assert restored.array.dtype == np.float64

    def test_base64_field_accepts_list_payload(
        self, metadata_list_payload: dict, array_2d_float: np.ndarray
    ):
        """Base64-mode fields accept list-encoded metadata payloads on input.

        The validator is permissive about the envelope shape; the encoding
        setting controls serialization output, not input strictness.
        """
        model = Base64FloatMatrixModel.model_validate({"matrix": metadata_list_payload})

        np.testing.assert_allclose(model.matrix, array_2d_float)

    def test_list_field_accepts_base64_payload(
        self, metadata_base64_payload: dict, array_2d_float: np.ndarray
    ):
        """List-mode fields accept base64-encoded metadata payloads on input."""
        model = DefaultArrayModel.model_validate({"array": metadata_base64_payload})

        np.testing.assert_allclose(model.array, array_2d_float)


class TestNDArrayFactory:
    """Validation that happens at NDArray(...) construction time (spec layer)."""

    def test_factory_rejects_invalid_dtype_string(self):
        """Unparseable dtype strings raise immediately."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            NDArray(dtype="invalid-dtype")

    def test_factory_rejects_non_tuple_shape(self):
        """Shape must be a tuple; lists and other sequences are rejected."""
        with pytest.raises(TypeError, match="shape must be a tuple"):
            NDArray(shape=[2, 2])

    def test_factory_rejects_negative_shape_dim(self):
        """Negative dimensions are rejected at construction."""
        with pytest.raises(ValueError, match="non-negative int"):
            NDArray(shape=(-1, 3))

    def test_factory_rejects_bool_shape_dim(self):
        """Booleans are not valid shape dimensions, even though they are ints."""
        with pytest.raises(ValueError, match="non-negative int"):
            NDArray(shape=(True, 3))

    def test_factory_rejects_invalid_encoding(self):
        """Encoding must be 'list' or 'base64' — anything else is rejected."""
        with pytest.raises(ValueError, match="Unsupported encoding"):
            NDArray(encoding="yaml")


class TestJSONSchema:
    """JSON Schema generation for NDArray fields."""

    def test_unconstrained_schema(self):
        """NDArray() with no constraints publishes a permissive object schema."""

        class AnyModel(BaseModel):
            x: NDArray()

        field = AnyModel.model_json_schema()["properties"]["x"]

        assert field["type"] == "object"
        assert "dtype" not in field["properties"]
        assert "shape" not in field["properties"]
        assert field["x-numpy-dtype"] is None
        assert field["x-numpy-shape"] is None
        assert field["x-numpy-encoding"] == "list"

    def test_dtype_and_shape_schema(self):
        """model_json_schema() reflects dtype/shape constraints from _NDArraySpec."""
        field = Float64MatrixModel.model_json_schema()["properties"]["matrix"]

        assert field["type"] == "object"
        assert field["properties"]["dtype"]["const"] == "float64"
        assert field["properties"]["shape"]["const"] == [2, 2]
        assert field["x-numpy-dtype"] == "float64"
        assert field["x-numpy-shape"] == [2, 2]
        assert field["x-numpy-encoding"] == "list"

    def test_partial_shape_schema(self):
        """Partial shapes constrain rank but expose the pattern via x-numpy-shape."""

        class PartialShapeModel(BaseModel):
            rows: NDArray(dtype="float32", shape=(None, 3))

        field = PartialShapeModel.model_json_schema()["properties"]["rows"]

        assert field["properties"]["shape"]["minItems"] == 2
        assert field["properties"]["shape"]["maxItems"] == 2
        # const is dropped for whildcards and the schema relies on min/maxItems for validation.
        assert "const" not in field["properties"]["shape"]
        assert field["x-numpy-shape"] == [None, 3]

    def test_base64_encoding_schema(self):
        """Base64 mode is reflected in the schema as a required const property."""

        class B64Model(BaseModel):
            data: NDArray(dtype="float64", encoding="base64")

        field = B64Model.model_json_schema()["properties"]["data"]

        assert field["properties"]["encoding"]["const"] == "base64"
        assert "encoding" in field["required"]


class TestModelComposition:
    """NDArray fields composed with other fields, options, and types."""

    def test_multiple_ndarray_fields_round_trip(
        self,
        ndarray_any_type,
        ndarray_float64_matrix_base64_type,
        array_1d_int: np.ndarray,
        array_2d_float: np.ndarray,
        metadata_base64_payload: dict,
    ):
        """Validates and round-trips multiple NDArray fields in a single model."""
        model_cls = create_model(
            "MultiArrayModel",
            first=(ndarray_any_type, ...),
            second=(ndarray_float64_matrix_base64_type, ...),
        )
        model = model_cls(first=array_1d_int, second=array_2d_float)

        dumped = model.model_dump(round_trip=True)
        assert dumped["first"]["array"] == array_1d_int.tolist()
        assert dumped["first"]["dtype"] == str(array_1d_int.dtype)
        assert dumped["first"]["shape"] == list(array_1d_int.shape)
        assert dumped["second"] == metadata_base64_payload

        revalidated = model_cls.model_validate(dumped)
        np.testing.assert_allclose(revalidated.first, array_1d_int)
        np.testing.assert_allclose(revalidated.second, array_2d_float)

    def test_optional_ndarray_field_accepts_none(
        self, ndarray_any_type, array_2d_float: np.ndarray
    ):
        """Allows optional NDArray fields to be omitted or null."""
        model_cls = create_model(
            "OptionalArrayModel", array=(Optional[ndarray_any_type], None)
        )

        model = model_cls.model_validate({})
        assert model.array is None
        assert model.model_dump()["array"] is None

        with_value = model_cls.model_validate({"array": array_2d_float})
        np.testing.assert_allclose(with_value.array, array_2d_float)


class TestInvalidPayloads:
    """End-to-end rejection of malformed inputs at model_validate."""

    @pytest.mark.parametrize(
        "payload, exc_type",
        [
            ("not-an-array", TypeError),
            (123, TypeError),
            ({"dtype": "float64"}, TypeError),
            (
                {
                    "array": "not-base64",
                    "dtype": "float64",
                    "shape": [2, 2],
                    "encoding": "base64",
                },
                ValidationError,
            ),
            (
                {
                    "array": "YWJj",
                    "dtype": "not-a-dtype",
                    "shape": [2, 2],
                    "encoding": "base64",
                },
                ValidationError,
            ),
            ({"array": "YWJj", "encoding": "list"}, ValidationError),
        ],
        ids=[
            "bare_string",
            "bare_int",
            "missing_array_key",
            "invalid_base64_string",
            "invalid_dtype_in_metadata",
            "string_payload_without_base64_encoding",
        ],
    )
    def test_invalid_payloads_raise(
        self,
        payload,
        exc_type: type[Exception],
    ):
        """Rejects invalid NDArray payloads during model validation."""
        with pytest.raises(exc_type):
            DefaultArrayModel.model_validate({"array": payload})

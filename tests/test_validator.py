import numpy as np
import pytest
from numpydantic._ndarray import NDArrayValidator


class TestNDArrayValidator:
    @pytest.mark.parametrize(
        "fixture_name",
        ["array_2d_float", "list_2d_float", "tuple_2d_float", "metadata_list_payload"],
    )
    def test_to_numpy_array_supported_inputs(
        self,
        fixture_name: str,
        request: pytest.FixtureRequest,
        array_2d_float: np.ndarray,
    ):
        """Converts supported input formats to ndarray."""
        value = request.getfixturevalue(fixture_name)
        validator = NDArrayValidator()

        result = validator(value)

        assert isinstance(result, np.ndarray)

        np.testing.assert_allclose(result, array_2d_float)

    def test_to_numpy_array_base64_metadata(
        self, metadata_base64_payload: dict, array_2d_float: np.ndarray
    ):
        """Decodes base64 metadata payloads into arrays."""
        validator = NDArrayValidator()

        result = validator(metadata_base64_payload)

        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, array_2d_float)

    @pytest.mark.parametrize("value", ["not-an-array", 123, {"dtype": "float64"}])
    def test_to_numpy_array_unsupported_types(self, value: object):
        """Rejects unsupported input types."""
        validator = NDArrayValidator()

        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            validator(value)

    def test_validate_dtype_matches(self, array_2d_float: np.ndarray):
        """Returns a fresh array preserving dtype when it already matches."""
        validator = NDArrayValidator(dtype="float64")

        result = validator(array_2d_float)

        assert result.dtype == np.dtype("float64")
        np.testing.assert_array_equal(result, array_2d_float)

    def test_validate_dtype_casts(self, array_1d_int: np.ndarray):
        """Casts arrays to the expected dtype when possible."""
        validator = NDArrayValidator(dtype="float64")

        result = validator(array_1d_int)

        assert result.dtype == np.dtype("float64")
        np.testing.assert_allclose(result, array_1d_int.astype("float64"))

    def test_validate_dtype_invalid_string(self):
        """Raises at construction when configured with an invalid dtype string."""
        with pytest.raises(TypeError, match="data type 'invalid-dtype' not understood"):
            NDArrayValidator(dtype="invalid-dtype")

    def test_validate_dtype_non_convertible(self):
        """Raises when values cannot be cast to the expected dtype."""
        validator = NDArrayValidator(dtype="float64")
        data = np.array(["a", "b"])

        with pytest.raises(ValueError, match="Cannot convert input to dtype"):
            validator(data)

    def test_validate_rejects_non_numeric_dtype_at_construction(self):
        """Rejects non-numeric dtype configured on the validator."""
        with pytest.raises(TypeError, match="only numeric dtypes"):
            NDArrayValidator(dtype="U5")  # Unicode string dtype

    def test_validate_rejects_non_numeric_array(self):
        """Rejects string/object arrays even when dtype is unconstrained."""
        validator = NDArrayValidator()
        with pytest.raises(TypeError, match="only numeric dtypes"):
            validator(np.array(["a", "b", "c"]))

    @pytest.mark.parametrize(
        "arr",
        [
            np.array(["a", "b"]),
            np.array([1, object()], dtype=object),
            np.array(["2020-01-01"], dtype="datetime64[D]"),
        ],
        ids=["string", "object", "datetime"],
    )
    def test_validate_rejects_all_non_numeric_kinds(self, arr: np.ndarray):
        """Rejects every non-numeric dtype kind (string/object/datetime)."""
        validator = NDArrayValidator()
        with pytest.raises(TypeError, match="only numeric dtypes"):
            validator(arr)

    @pytest.mark.parametrize(
        "dtype",
        [
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        ],
    )
    def test_validate_accepts_all_numeric_kinds(self, dtype: str):
        """Every numeric dtype the README promises is accepted."""
        validator = NDArrayValidator(dtype=dtype)
        arr = np.zeros(3, dtype=dtype)

        result = validator(arr)

        assert result.dtype == np.dtype(dtype)

    def test_validate_dtype_none_accepts_any_dtype(self):
        """dtype=None preserves the input dtype but returns a fresh array."""
        validator = NDArrayValidator()
        arr = np.array([1, 2, 3], dtype=np.int8)

        result = validator(arr)

        assert result.dtype == np.int8
        np.testing.assert_array_equal(result, arr)

    def test_validate_shape_exact_match(self, array_2d_float: np.ndarray):
        """Accepts arrays that match the exact configured shape."""
        validator = NDArrayValidator(shape=(2, 2))

        result = validator(array_2d_float)

        assert result.shape == (2, 2)

    def test_validate_shape_exact_failure(self, array_2d_float: np.ndarray):
        """Rejects arrays that do not match the exact configured shape."""
        validator = NDArrayValidator(shape=(1, 2))

        with pytest.raises(ValueError, match="Expected shape"):
            validator(array_2d_float)

    def test_validate_partial_shape_match(self, array_2d_float: np.ndarray):
        """Accepts arrays whose concrete dims line up with a partially-open shape."""
        validator = NDArrayValidator(shape=(None, 2))

        result = validator(array_2d_float)

        assert result.shape == (2, 2)

    def test_validate_partial_shape_axis_mismatch(self, array_2d_float: np.ndarray):
        """Rejects mismatches on a fixed axis and names the offending axis."""
        validator = NDArrayValidator(shape=(None, 3))

        with pytest.raises(ValueError, match=r"axis 1: expected 3, got 2"):
            validator(array_2d_float)

    def test_validate_shape_wrong_rank(self, array_2d_float: np.ndarray):
        """Rejects arrays whose rank doesn't match the shape spec length."""
        validator = NDArrayValidator(shape=(None, None, None))

        with pytest.raises(ValueError, match="Expected 3D array"):
            validator(array_2d_float)

    def test_validate_shape_none_accepts_any_shape(self):
        """shape=None accepts arrays of any rank and returns a fresh copy."""
        validator = NDArrayValidator()

        for arr in [np.zeros(3), np.zeros((2, 2)), np.zeros((1, 2, 3, 4))]:
            result = validator(arr)
            assert result.shape == arr.shape
            np.testing.assert_array_equal(result, arr)

    def test_validate_shape_all_none_dims(self, array_2d_float: np.ndarray):
        """A shape of all `None` entries only constrains rank."""
        validator = NDArrayValidator(shape=(None, None))

        result = validator(array_2d_float)

        assert result.shape == (2, 2)

    def test_validate_shape_empty_array(self):
        """Accepts empty 1D arrays when shape=(0,) is configured."""
        validator = NDArrayValidator(shape=(0,))

        result = validator(np.array([], dtype="float64"))

        assert result.shape == (0,)

    def test_validate_shape_empty_array_with_none_dim(self):
        """A `None` first dim accepts an empty (0, 3) array."""
        validator = NDArrayValidator(shape=(None, 3))

        result = validator(np.zeros((0, 3)))

        assert result.shape == (0, 3)

    def test_validate_dtype_and_shape_both_match(self, array_2d_float: np.ndarray):
        """Both dtype and shape constraints satisfied."""
        validator = NDArrayValidator(dtype="float64", shape=(2, 2))

        result = validator(array_2d_float)

        assert result.dtype == np.dtype("float64")
        assert result.shape == (2, 2)

    def test_validate_dtype_match_shape_fail(self, array_2d_float: np.ndarray):
        """dtype matches but shape doesn't — shape error wins."""
        validator = NDArrayValidator(dtype="float64", shape=(3, 3))

        with pytest.raises(ValueError, match="Expected shape"):
            validator(array_2d_float)

    def test_validate_dtype_cast_then_shape_match(self, array_1d_int: np.ndarray):
        """Cast happens first; shape is checked on the post-cast array."""
        validator = NDArrayValidator(dtype="float64", shape=(3,))

        result = validator(array_1d_int)

        assert result.dtype == np.dtype("float64")
        assert result.shape == (3,)

"""Bridge between NDArray fields and Pydantic's schema system.

Pydantic discovers custom types by looking for two methods on objects placed
inside ``Annotated[...]``:

* ``__get_pydantic_core_schema__`` — tells Pydantic how to validate and
  serialize values.
* ``__get_pydantic_json_schema__`` — tells Pydantic what JSON Schema to
  publish (e.g. for OpenAPI docs).

``_NDArraySpec`` implements both, using the dtype/shape/encoding the user
passed to ``NDArray(...)``.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from .constants import BASE64_ENCODING, LIST_ENCODING, NDArrayEncoding
from .serializer import NDArraySerializer
from .validator import NDArrayValidator


@dataclass(frozen=True)
class _NDArraySpec:
    """Holds the dtype, shape, and encoding for one NDArray field.

    What it does:

    - Stores the user's choices from ``NDArray(dtype=..., shape=..., encoding=...)``.
    - Hands Pydantic the schemas it needs (validation, serialization, JSON Schema).

    Why frozen:

    - Hashable, so Pydantic can cache schemas keyed by the spec.
    - Equal arguments produce equal specs (no surprise inequality).
    - Config can't drift after the model is defined.

    Shape conventions:

    - A ``None`` entry matches any size on that axis.
    - Example: ``shape=(None, 3)`` accepts any 2D array with 3 columns.
    """

    dtype: Optional[str] = None
    shape: Optional[Tuple[Optional[int], ...]] = None
    encoding: NDArrayEncoding = LIST_ENCODING

    def __post_init__(self):
        """Validate the spec on initialization."""
        if self.dtype is not None:
            try:
                np.dtype(self.dtype)
            except TypeError as e:
                raise ValueError(f"Invalid dtype '{self.dtype}': {e}") from e

        if self.shape is not None:
            if not isinstance(self.shape, tuple):
                raise TypeError(
                    f"shape must be a tuple, got {type(self.shape).__name__}"
                )
            for i, dim in enumerate(self.shape):
                if dim is None:
                    continue
                if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                    raise ValueError(
                        f"shape[{i}] must be a non-negative int or None, got {dim!r}"
                    )

        if self.encoding not in (LIST_ENCODING, BASE64_ENCODING):
            raise ValueError(
                f"Unsupported encoding '{self.encoding}'. "
                f"Expected '{LIST_ENCODING}' or '{BASE64_ENCODING}'."
            )

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Tell Pydantic how to validate and serialize the field.

        Wires up ``NDArrayValidator`` (for incoming values) and
        ``NDArraySerializer`` (for ``model_dump`` / ``model_dump_json``).

        Notes:
            ``source_type`` and ``handler`` are part of Pydantic's protocol
            signature. Unused here because the spec already carries every input
            the schemas need..
        """
        validator = NDArrayValidator(self.dtype, self.shape)
        serializer = NDArraySerializer(self.encoding)
        return core_schema.no_info_plain_validator_function(
            function=validator,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serializer,
                info_arg=True,
                return_schema=core_schema.any_schema(),
                when_used="always",
            ),
        )

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Build the JSON Schema published for this field.

        - Describes the envelope: ``{array, dtype, shape, encoding}``.
        - Pins values with ``const`` when fully known; ``None`` entries fall
          back to rank-only checks via ``minItems`` / ``maxItems``.
        - ``x-numpy-*`` keys carry the original spec for NumPy-aware tooling.
        """
        properties: dict[str, Any] = {
            "array": {
                "description": "Array payload as nested lists or a base64-encoded string.",
            }
        }
        required: list[str] = ["array"]

        if self.dtype is not None:
            properties["dtype"] = {"type": "string", "const": self.dtype}

        if self.shape is not None:
            shape_schema: dict[str, Any] = {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": len(self.shape),
                "maxItems": len(self.shape),
            }
            if all(d is not None for d in self.shape):
                shape_schema["const"] = list(self.shape)
            properties["shape"] = shape_schema

        if self.encoding == BASE64_ENCODING:
            properties["encoding"] = {"type": "string", "const": BASE64_ENCODING}
            required.append("encoding")

        return {
            "type": "object",
            "title": "NDArray",
            "properties": properties,
            "required": required,
            "x-numpy-dtype": self.dtype,
            "x-numpy-shape": list(self.shape) if self.shape is not None else None,
            "x-numpy-encoding": self.encoding,
        }

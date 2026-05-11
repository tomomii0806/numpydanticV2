# NumPydantic v2

This library introduces custom Pydantic v2 type annotations for NumPy arrays with dtype validation, shape validation, and round-trip JSON serialization.

## Features

- **Typed arrays**: dtype and shape (axes can be left unconstrained) enforced at validation time
- **Lossless JSON round-trips**: dtype and shape preserved across `model → JSON → model`
- **Two encoding modes**: human-readable `list`, or compact `base64` for large / exact-float arrays

## Installation

This assignment uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
uv sync --extra dev
```

This creates a `.venv/`, installs the package, and pins everything via `uv.lock`. You can run any command in your env with `uv run <cmd>` (no activation needed).

<details>
<summary>Without uv</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
</details>

## Quick Start

```python
import numpy as np
from pydantic import BaseModel
from numpydantic import NDArray


class Matrix(BaseModel):
    data: NDArray(dtype="float64", shape=(None, 3))   # any number of rows, 3 columns


m = Matrix(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

json_payload = m.model_dump_json(round_trip=True)
restored = Matrix.model_validate_json(json_payload)

assert restored.data.shape == (2, 3)
assert restored.data.dtype == np.float64
```

For a hands-on walkthrough of all modes (list, list with `round_trip=True`, base64, validation errors, JSON schema), see [`demo.ipynb`](demo.ipynb) — open it on GitHub or run `uv run jupyter notebook demo.ipynb`.

## API

```python
NDArray(
    dtype:    Optional[str] = None,
    shape:    Optional[Tuple[Optional[int], ...]] = None,
    encoding: Literal["list", "base64"] = "list",
) -> Annotated[np.ndarray, _NDArraySpec]
```

| Parameter  | Description |
|---|---|
| `dtype`    | NumPy dtype name (`"float64"`, `"int32"`, …). `None` accepts any dtype. |
| `shape`    | Shape tuple. `None` entries match any size on that axis (`(None, 3)` = any first dim, 3 cols). |
| `encoding` | `"list"` (default, readable) or `"base64"` (compact, exact). |

### Examples

```python
NDArray()                                     # any array
NDArray(dtype="float32")                      # dtype only
NDArray(shape=(2, 3))                         # exact shape
NDArray(shape=(None, 3))                      # any first dim, fixed 3 cols
NDArray(shape=(None, None))                   # any 2D array
NDArray(dtype="uint8", shape=(224, 224, 3))   # exact dtype + shape
NDArray(dtype="float64", encoding="base64")   # compact for large arrays
```

## Encoding modes

| Mode     | Output (default)                          | Output (round-trip)                                      | When to use |
|----------|--------------------------------------------|----------------------------------------------------------|-------------|
| `list`   | `[[1.0, 2.0], [3.0, 4.0]]`                 | `{"array": [[…]], "dtype": "...", "shape": [...]}`       | Default. Human-readable, easy to inspect by hand. |
| `base64` | `{"array": "<b64>", "dtype": "...", …}`    | same                                                     | Large arrays, exact float preservation (NaN/±inf survive). |

`round_trip=True` is Pydantic's built-in dump flag. It switches `list` mode from a bare nested list to the metadata envelope so dtype/shape can be recovered. `base64` mode always emits the envelope and ignores the flag.

## Performance

Benchmark on a 1M-element `float64` array (7.63 MB raw), 5 trials, run via `python bench.py`:

| mode                       | JSON size (MB) | dump (ms) | parse (ms) |
|----------------------------|---------------:|----------:|-----------:|
| list (`round_trip=False`)  |          18.72 |      94.6 |      108.8 |
| list (`round_trip=True`)   |          18.72 |      94.4 |      108.9 |
| base64                     |          10.17 |      21.7 |       30.8 |

> Note: this result was produced on a 2020 MacBook with Intel processor.

For arrays of this size, `base64` is ~1.8× smaller, ~4× faster to dump, and ~3.5× faster to parse. `list` is preferred for small/config-like arrays where readability matters; switch to `base64` for arrays larger than ~10K elements or when you need exact float preservation.

## JSON schema

`model_json_schema()` reflects the configured constraints. For `NDArray(dtype="float64", shape=(2, 2))`:

```json
{
  "type": "object",
  "title": "NDArray",
  "properties": {
    "array": {"description": "Array payload as nested lists or a base64-encoded string."},
    "dtype": {"type": "string", "const": "float64"},
    "shape": {"type": "array", "items": {"type": "integer"},
              "minItems": 2, "maxItems": 2, "const": [2, 2]}
  },
  "required": ["array"],
  "x-numpy-dtype": "float64",
  "x-numpy-shape": [2, 2],
  "x-numpy-encoding": "list"
}
```

`const` can only pin a fixed value, so shapes with `None` entries fall back to a number-of-dimensions check via `minItems`/`maxItems`. The original `None` pattern is preserved on `x-numpy-shape` (`[null, 3]`) for NumPy-aware tooling.

## How it works

`NDArray(...)` returns `Annotated[np.ndarray, _NDArraySpec(...)]`. `_NDArraySpec` is a frozen dataclass that implements two Pydantic v2 hooks:

- `__get_pydantic_core_schema__` wires an `NDArrayValidator` and `NDArraySerializer` into Pydantic's core schema.
- `__get_pydantic_json_schema__` emits a JSON Schema describing the round-trip envelope plus dtype/shape/encoding constraints.

Validation runs in three staged layers, so failures surface as early as possible:

1. **Spec construction** (at the `NDArray(...)` call): `_NDArraySpec.__post_init__` checks that `dtype` parses, `shape` is a well-formed tuple of `int|None`, and `encoding` is `"list"` or `"base64"`. Malformed type definitions raise immediately.
2. **Schema build** (at model class definition): Pydantic invokes `__get_pydantic_core_schema__`, which constructs `NDArrayValidator(dtype, shape)` once. Non-numeric dtypes are rejected here.
3. **Value validation** (at `model_validate*`): the validator's `__call__` dispatches on input type (ndarray / list / tuple / metadata dict) via `isinstance` checks, copies into a fresh array via `np.array(v, dtype=...)` (always copies, so the model's array is not aliased to the caller's input), and enforces the shape constraints.

```
numpydantic/
├── __init__.py            # public package exports
├── ndarray.py             # NDArray(...) factory
└── _ndarray/
    ├── __init__.py
    ├── constants.py       # JSON keys + encoding literal
    ├── metadata.py        # metadata-dict → ndarray
    ├── validator.py       # NDArrayValidator (dtype + shape)
    ├── serializer.py      # NDArraySerializer (list / base64)
    └── spec.py            # _NDArraySpec (Pydantic core/JSON schema)

demo.ipynb                 # walkthrough notebook (all modes, validation, schema)
bench.py                   # list-vs-base64 benchmark
tests/                     # pytest suite (validator, serializer, integration)
```

## Testing

```bash
uv run pytest -v
uv run python bench.py
```

## Supported dtypes

All NumPy numeric dtypes by name: `int8/16/32/64`, `uint8/16/32/64`, `float16/32/64`, `complex64/128`, `bool`.

String, object, datetime, and timedelta dtypes are explicitly rejected. They don't have a clean JSON representation and tend to introduce platform-dependent surprises. Restricting to numeric tensors matches the precedent set by JAX.

## Requirements

- Python 3.10+
- NumPy
- Pydantic v2

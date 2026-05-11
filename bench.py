"""Benchmark list vs base64 NDArray encoding on a 1M-element float64 array.

Reports payload size and serialize/deserialize time for each mode.
Run: ``python bench.py``
"""

import time
from statistics import median
import numpy as np
from pydantic import BaseModel
from numpydantic import NDArray


N_ELEMENTS = 1_000_000
N_TRIALS = 5


class ListModel(BaseModel):
    data: NDArray(dtype="float64")


class Base64Model(BaseModel):
    data: NDArray(dtype="float64", encoding="base64")


def time_call(fn, trials=N_TRIALS):
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return median(samples)


def bench(model_cls, label, arr, round_trip):
    instance = model_cls(data=arr)

    payload = instance.model_dump_json(round_trip=round_trip)
    size_mb = len(payload.encode("utf-8")) / (1024 * 1024)

    dump_time = time_call(lambda: instance.model_dump_json(round_trip=round_trip))
    parse_time = time_call(lambda: model_cls.model_validate_json(payload))

    restored = model_cls.model_validate_json(payload)
    assert np.allclose(restored.data, arr)
    assert restored.data.dtype == arr.dtype

    return {
        "label": label,
        "size_mb": size_mb,
        "dump_ms": dump_time * 1000,
        "parse_ms": parse_time * 1000,
    }


def main():

    # Generate a large random array for testing.
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(N_ELEMENTS, dtype=np.float64)

    print(
        f"Benchmark: {N_ELEMENTS:,} float64 elements "
        f"({arr.nbytes / (1024 * 1024):.2f} MB raw), median of {N_TRIALS} trials"
    )
    print()

    # Run benchmarks for list and base64 encoding modes, with and without round-trip metadata.
    rows = [
        bench(ListModel, "list (round_trip=False)", arr, round_trip=False),
        bench(ListModel, "list (round_trip=True)", arr, round_trip=True),
        bench(Base64Model, "base64", arr, round_trip=False),
    ]

    # Print results
    header = f"{'mode':<28} {'size (MB)':>10} {'dump (ms)':>12} {'parse (ms)':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<28} "
            f"{r['size_mb']:>10.2f} "
            f"{r['dump_ms']:>12.1f} "
            f"{r['parse_ms']:>12.1f}"
        )


if __name__ == "__main__":
    main()

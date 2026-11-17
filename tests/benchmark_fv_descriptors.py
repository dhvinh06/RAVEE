"""Optional synthetic microbenchmark; no PCA/GMM fitting, videos or cache writes.

Run with: python -B -m tests.benchmark_fv_descriptors
"""

import json
from statistics import median
from time import perf_counter

import numpy as np

from tests.test_feature_extraction import legacy_descriptors, synthetic_tracks
from toolbox.feature_extraction import read_descriptors


def milliseconds(operation, repeats=9):
    operation()  # Warm both paths; release each result before the next call.
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        operation()
        samples.append((perf_counter() - started) * 1000)
    return median(samples)


def main():
    results = []
    for count in (2000, 20000):
        tracks = synthetic_tracks(count)
        matrix = read_descriptors(tracks, 'all', None)
        expected = legacy_descriptors(tracks)
        assert np.array_equal(matrix, expected) and matrix.dtype == expected.dtype
        del expected
        # A representative local frame includes repeated references, in order.
        indices = np.random.default_rng(23).integers(0, count, size=1024).tolist()
        assert np.array_equal(matrix[indices], legacy_descriptors(tracks, indices))
        results.append({'tracks': count, 'descriptor_shape': list(matrix.shape),
                        'dtype': str(matrix.dtype), 'repeats': 9,
                        'assembly_before_ms': milliseconds(lambda: legacy_descriptors(tracks)),
                        'assembly_after_ms': milliseconds(lambda: read_descriptors(tracks, 'all', None)),
                        'frame_rows': len(indices),
                        'frame_before_ms': milliseconds(lambda: legacy_descriptors(tracks, indices)),
                        'frame_after_ms': milliseconds(lambda: matrix[indices]),
                        'array_equal': True})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

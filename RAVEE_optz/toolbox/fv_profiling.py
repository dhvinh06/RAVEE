"""Optional, in-memory wall-clock profiling; never stores arrays or cache inputs."""

from contextlib import contextmanager, nullcontext
from time import perf_counter


class FVProfile:
    """Aggregate repeated frame timings instead of retaining one record per frame."""

    def __init__(self):
        self.timings = {}
        self.metadata = {}

    @contextmanager
    def measure(self, stage):
        started = perf_counter()
        try:
            yield
        finally:
            timing = self.timings.setdefault(stage, {'seconds': 0.0, 'calls': 0})
            timing['seconds'] += perf_counter() - started
            timing['calls'] += 1

    def matrix(self, name, value):
        self.metadata[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                               'bytes': int(value.nbytes)}

    def as_dict(self):
        return {'timings': self.timings, 'metadata': self.metadata}


def measure(profile, stage):
    return nullcontext() if profile is None else profile.measure(stage)

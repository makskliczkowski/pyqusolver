import time
import statistics
import sys
from contextlib import contextmanager

try:
    from QES.general_python.lattices.square import SquareLattice
except ImportError:
    SquareLattice = object # Dummy if import fails

def print_header(name):
    print(f"\nBenchmark: {name}")
    print("-" * 40)

def run_benchmark(name, func, repeats=3, **kwargs):
    """
    Runs a benchmark function multiple times and reports statistics.
    """
    times = []

    # Warmup
    print(f"  [{name}] Warmup...")
    try:
        start_warm = time.perf_counter()
        func(**kwargs)
        end_warm = time.perf_counter()
        print(f"  [{name}] Warmup finished in {end_warm - start_warm:.4f}s")
    except Exception as e:
        print(f"  [{name}] Failed during warmup: {e}")
        raise e

    # Repeats
    print(f"  [{name}] Running {repeats} repeats...")
    for i in range(repeats):
        start = time.perf_counter()
        func(**kwargs)
        end = time.perf_counter()
        dt = end - start
        times.append(dt)
        print(f"    Run {i+1}: {dt:.4f} s")

    mean_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time = min(times)
    max_time = max(times)

    print(f"  [{name}] Result: {mean_time:.4f} +/- {stdev_time:.4f} s (Min: {min_time:.4f}, Max: {max_time:.4f})")
    return {
        "mean": mean_time,
        "stdev": stdev_time,
        "min": min_time,
        "max": max_time,
        "times": times
    }

class BenchmarkSquareLattice(SquareLattice):
    """
    Wrapper around SquareLattice to fix float indexing issue.
    """
    def get_coordinates(self, *args):
        if len(args) == 1:
            # Cast index to int to avoid IndexError with float indices
            return super().get_coordinates(int(args[0]))
        return super().get_coordinates(*args)

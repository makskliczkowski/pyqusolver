import time
import statistics
import functools

def benchmark(name=None, n_repeat=5, n_warmup=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bench_name = name or func.__name__

            # Warmup
            for _ in range(n_warmup):
                func(*args, **kwargs)

            times = []
            for _ in range(n_repeat):
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)

            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0.0

            print(f"{bench_name:<50}: {mean_time:.6f} s +/- {std_time:.6f} s (N={n_repeat})")
            return mean_time
        return wrapper
    return decorator

class BenchmarkTimer:
    def __init__(self, name):
        self.name = name
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        duration = self.end - self.start
        print(f"{self.name:<50}: {duration:.6f} s")

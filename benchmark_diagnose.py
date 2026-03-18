import time
import numpy as np

# Dummy functions
def compute_autocorr_time(c):
    return 1.5

def compute_rhat(chains):
    return 1.0

# Generate dummy chains
chains = np.random.randn(100, 1000)

def loop_version(chains):
    taus = []
    for c in chains:
        taus.append(compute_autocorr_time(c))
    avg_tau = np.mean(taus)
    ess_total = np.sum([len(c) / t for c, t in zip(chains, taus)])
    return avg_tau, ess_total

def list_comp_version(chains):
    taus = [compute_autocorr_time(c) for c in chains]
    avg_tau = np.mean(taus)
    ess_total = np.sum([len(c) / t for c, t in zip(chains, taus)])
    return avg_tau, ess_total

# Warmup
loop_version(chains)
list_comp_version(chains)

# Benchmark loop version
start = time.perf_counter()
for _ in range(10000):
    loop_version(chains)
loop_time = time.perf_counter() - start

# Benchmark list comprehension version
start = time.perf_counter()
for _ in range(10000):
    list_comp_version(chains)
comp_time = time.perf_counter() - start

print(f"Loop version: {loop_time:.4f} seconds")
print(f"List comprehension version: {comp_time:.4f} seconds")
print(f"Improvement: {(loop_time - comp_time) / loop_time * 100:.2f}%")

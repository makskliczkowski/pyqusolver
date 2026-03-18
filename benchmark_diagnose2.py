import timeit
import numpy as np

chains = None


def get_chains():
    global chains
    if chains is None:
        chains = np.random.randn(100, 1000)
    return chains


def compute_autocorr_time(c):
    return np.sum(c) * 1.5


def loop_version():
    taus = []
    for c in get_chains():
        taus.append(compute_autocorr_time(c))
    return taus


def list_comp_version():
    taus = [compute_autocorr_time(c) for c in get_chains()]
    return taus


if __name__ == '__main__':
    np.random.seed(0)
    n = 10000
    loop_time = timeit.timeit(loop_version, number=n)
    comp_time = timeit.timeit(list_comp_version, number=n)

    print(f"Loop version: {loop_time:.4f} seconds")
    print(f"List comp version: {comp_time:.4f} seconds")
    print(f"Improvement: {(loop_time - comp_time) / loop_time * 100:.2f}%")

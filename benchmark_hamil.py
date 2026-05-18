import time
from typing import Any, Optional, Iterable, Sequence
from collections.abc import Mapping
import numpy as np

class DummyHamiltonian:
    def __init__(self, ns):
        self.ns = ns

    # Original
    def _coefficient_for_bond_orig(self, coefficient: Any, i: int, j: int, idx: int):
        if callable(coefficient):
            return coefficient(int(i), int(j), int(idx))
        if isinstance(coefficient, dict):
            if (i, j) in coefficient:
                return coefficient[(i, j)]
            if (j, i) in coefficient:
                return coefficient[(j, i)]
            return 0.0
        if isinstance(coefficient, (list, tuple)):
            if idx < len(coefficient):
                return coefficient[idx]
            raise ValueError(f"Bond coefficient sequence too short: need index {idx}, size={len(coefficient)}.")
        if hasattr(coefficient, "ndim"):
            if coefficient.ndim == 0:
                return coefficient.item()
            if idx < coefficient.size:
                return coefficient[idx]
            raise ValueError(f"Bond coefficient array sequence too short: need index {idx}, size={coefficient.size}.")
        from collections.abc import Mapping
        if isinstance(coefficient, Mapping):
            if (i, j) in coefficient:
                return coefficient[(i, j)]
            if (j, i) in coefficient:
                return coefficient[(j, i)]
            return 0.0
        return coefficient

    # Optimized
    def _coefficient_for_bond_opt(self, coefficient: Any, i: int, j: int, idx: int):
        if callable(coefficient):
            return coefficient(i, j, idx)
        if isinstance(coefficient, (list, tuple)):
            if idx < len(coefficient):
                return coefficient[idx]
            raise ValueError(f"Bond coefficient sequence too short: need index {idx}, size={len(coefficient)}.")
        if hasattr(coefficient, "ndim"):
            if coefficient.ndim == 0:
                return coefficient.item()
            if idx < coefficient.size:
                return coefficient[idx]
            raise ValueError(f"Bond coefficient array sequence too short: need index {idx}, size={coefficient.size}.")
        if isinstance(coefficient, dict) or hasattr(coefficient, "get"):
            if (i, j) in coefficient:
                return coefficient[(i, j)]
            if (j, i) in coefficient:
                return coefficient[(j, i)]
            return 0.0
        return coefficient

def run_bench():
    h = DummyHamiltonian(100)

    coef_scalar = 1.0
    coef_list = list(range(100))
    coef_tuple = tuple(range(100))
    coef_dict = {(i, i+1): float(i) for i in range(100)}
    coef_arr = np.arange(100)
    def coef_call(i, j, idx): return float(i)

    vals = [coef_scalar, coef_list, coef_tuple, coef_dict, coef_arr, coef_call]

    start = time.time()
    for _ in range(10000):
        for v in vals:
            for s in range(99):
                h._coefficient_for_bond_orig(v, s, s+1, s)
    end = time.time()
    print(f"Original: {end - start:.4f}s")

    start = time.time()
    for _ in range(10000):
        for v in vals:
            for s in range(99):
                h._coefficient_for_bond_opt(v, s, s+1, s)
    end = time.time()
    print(f"Optimized: {end - start:.4f}s")

if __name__ == '__main__':
    run_bench()

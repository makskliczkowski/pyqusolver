## 2024-05-24 - [Avoid list literals for static membership checks]
**Learning:** Using `val in [...]` causes a new list to be created each time the expression is evaluated, whereas `val in {...}` in a membership test can be compiled to checking against a `frozenset` constant in CPython bytecode, which can be faster for static string membership checks in hot paths.
**Action:** Replace `in ["val1", "val2"]` with `in {"val1", "val2"}` for static string membership checking.

## 2025-02-27 - [Avoid abstract base classes in isinstance hot paths]
**Learning:** Using `isinstance(val, (numbers.Number, Mapping, Iterable))` in hot paths like `_ADD_CONDITION` in `hamil.py` (which is called for every site and bond during Hamiltonian construction) is significantly slower than using concrete types like `(int, float, complex, np.number, dict, list, tuple, set)`. Furthermore, removing redundant `try/except` around `np.asarray()` processing when dealing with scalars and arrays provides a massive speedup (up to ~28x for float inputs and ~4x for array inputs).
**Action:** Replace abstract base classes with concrete types in performance-critical `isinstance` checks, and avoid using `try/except` for control flow where simpler vectorized operations or type checks suffice.

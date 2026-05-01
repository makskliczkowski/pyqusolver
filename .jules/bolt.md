## 2025-04-02 - Enum Membership Testing Overhead
**Learning:** While inline set literals (`in {'a'}`) are faster for strings and ints, inline set literals for Python Enums (e.g., `in {MyEnum.A}`) are actually ~3x slower than list literals due to lack of constant-folding for Enums by the Python compiler. Tuple literals `in (MyEnum.A,)` perform identically to lists.
**Action:** For Enums, avoid inline set literals `in {MyEnum.A}`. Stick to `in [MyEnum.A]` or explicitly precompute `frozenset` at the class/module level if it's in a hot loop.

## 2025-02-27 - [Avoid abstract base classes in isinstance hot paths]
**Learning:** Using `isinstance(val, (numbers.Number, Mapping, Iterable))` in hot paths like `_ADD_CONDITION` in `hamil.py` (which is called for every site and bond during Hamiltonian construction) is significantly slower than using concrete types like `(int, float, complex, np.number, dict, list, tuple, set)`. Furthermore, removing redundant `try/except` around `np.asarray()` processing when dealing with scalars and arrays provides a massive speedup (up to ~28x for float inputs and ~4x for array inputs).
**Action:** Replace abstract base classes with concrete types in performance-critical `isinstance` checks, and avoid using `try/except` for control flow where simpler vectorized operations or type checks suffice.

## 2025-02-27 - [Avoid O(N^2) array allocations in coefficient lookups]
**Learning:** Calling `np.asarray(coefficient)` inside `_coefficient_for_site` and `_coefficient_for_bond` loops for every site/bond creates redundant array instances when `coefficient` is already a list or tuple. This turns an $O(N)$ initialization loop into $O(N^2)$ memory allocations.
**Action:** Handle `list` and `tuple` types directly without converting them to `np.asarray()` inside inner loops. Check `len(coefficient)` and index directly.
## 2025-05-01 - Hoist invariant boolean checks out of Numba/NumPy loops
**Learning:** In performance-critical functions wrapped with `@numba.njit` (e.g., `sigma_x_np`, `sigma_y_np`, etc.), checking loop-invariant booleans (like `if not spin`) inside the loop prevents optimal compilation and adds branch overhead.
**Action:** Hoist the invariant conditional check outside the loop, creating two slightly duplicate but optimally fast loops.

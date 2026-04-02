## 2025-04-02 - Enum Membership Testing Overhead
**Learning:** While inline set literals (`in {'a'}`) are faster for strings and ints, inline set literals for Python Enums (e.g., `in {MyEnum.A}`) are actually ~3x slower than list literals due to lack of constant-folding for Enums by the Python compiler. Tuple literals `in (MyEnum.A,)` perform identically to lists.
**Action:** For Enums, avoid inline set literals `in {MyEnum.A}`. Stick to `in [MyEnum.A]` or explicitly precompute `frozenset` at the class/module level if it's in a hot loop.

## 2025-04-02 - String Concatenation Pattern
**Learning:** Repeated `+=` string concatenation (e.g., `tmp += f"{gen}={sec},"`), especially in properties accessed frequently like `get_sym_info` during hashing or directory name generation, adds measurable O(N^2) overhead.
**Action:** Always replace loop-based `+=` string concatenation with list appends and `",".join(parts)`. It is consistently 10-20% faster and cleaner.

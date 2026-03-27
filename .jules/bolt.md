## 2024-05-24 - [Avoid list literals for static membership checks]
**Learning:** Using `val in [...]` causes a new list to be created each time the expression is evaluated, whereas `val in {...}` in a membership test can be compiled to checking against a `frozenset` constant in CPython bytecode, which can be faster for static string membership checks in hot paths.
**Action:** Replace `in ["val1", "val2"]` with `in {"val1", "val2"}` for static string membership checking.

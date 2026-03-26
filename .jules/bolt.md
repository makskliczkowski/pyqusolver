## 2024-05-24 - [Avoid list literals for static membership checks]
**Learning:** Using `val in [...]` causes the list to be created (or at best converted to a tuple internally), but `val in {...}` compiles directly to checking against a `frozenset` constant in CPython bytecode, which can be faster for static string membership checks in hot paths.
**Action:** Replace `in ["val1", "val2"]` with `in {"val1", "val2"}` for static string membership checking.

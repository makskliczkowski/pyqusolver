## 2026-03-26 - [CRITICAL] Prevent Arbitrary Code Execution via np.load
**Vulnerability:** Found insecure deserialization of Numpy arrays (`np.load`) with `allow_pickle=True` enabled when loading checkpoints.
**Learning:** By default, Numpy allows loading Python objects from .npy and .npz files using `pickle`. This enables an attacker to construct malicious Numpy files that execute arbitrary Python code upon loading.
**Prevention:** Always verify if pickled Python objects are required. When loading simple numeric arrays, set `allow_pickle=False` explicitly to block the execution of untrusted code. We secured `nqs_checkpoint_manager.py` and `mes.py`.

## 2026-03-27 - [CRITICAL] Prevent Arbitrary Code Execution via np.load
**Vulnerability:** Found remaining insecure deserialization of Numpy arrays (`np.load`) missing `allow_pickle=False` in `nqs_exact.py`, `dqmc_solver.py`, and `lazy_entry.py` (in `general_python` submodule).
**Learning:** Even after initial fixes, vulnerable patterns can persist in submodules or less frequently accessed test/solver code. `np.load` calls without explicit `allow_pickle=False` remain a significant risk for arbitrary code execution if an attacker supplies a malicious `.npy` or `.npz` file.
**Prevention:** Consistently apply `allow_pickle=False` to all `np.load` calls across the entire codebase, including submodules, unless deserialization of Python objects is explicitly required and the source is strictly trusted.

## 2026-03-26 - [CRITICAL] Prevent Arbitrary Code Execution via np.load
**Vulnerability:** Found insecure deserialization of Numpy arrays (`np.load`) with `allow_pickle=True` enabled when loading checkpoints.
**Learning:** By default, Numpy allows loading Python objects from .npy and .npz files using `pickle`. This enables an attacker to construct malicious Numpy files that execute arbitrary Python code upon loading.
**Prevention:** Always verify if pickled Python objects are required. When loading simple numeric arrays, set `allow_pickle=False` explicitly to block the execution of untrusted code. We secured `nqs_checkpoint_manager.py` and `mes.py`.

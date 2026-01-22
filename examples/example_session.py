
import os
import QES

print("Initial backend:", QES.get_backend_manager().name)

with QES.run(backend='numpy', seed=123, precision='float32') as session:
    print("Inside session (numpy, seed=123, precision=float32)")
    print("Backend:", QES.get_backend_manager().name)
    print("Seed:", QES.get_backend_manager().default_seed)
    print("Precision Env Var:", os.environ.get("PY_FLOATING_POINT"))

print("Session ended.")

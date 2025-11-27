import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union, List, Tuple, Dict

import jax
import jax.numpy as jnp

#! ----------------------------------------------------------------------
os.environ['PY_BACKEND']            = 'jax'         # Backend for numerical operations
#! ----------------------------------------------------------------------

# get the QES path from the environment variable if set
qes_path    = Path(os.environ.get("QES_PYPATH", "/usr/local/QES")).resolve()
if not qes_path.exists() or not qes_path.is_dir():
    raise FileNotFoundError(f"QES QES_PYPATH '{qes_path}' does not exist or is not a directory. "
                            f"If QES is installed, please set the QES_PYPATH environment variable.")
    
print(f"Using QES path: {qes_path}")
cwd         = Path.cwd()
file_path   = cwd
mod_path    = file_path.parent.resolve()
lib_path    = qes_path / 'QES'
gen_python  = lib_path / 'general_python'
extra_paths = [file_path, mod_path, qes_path, lib_path, gen_python]
for p, label in zip(extra_paths, ["file_path", "mod_path", "qes_path", "lib_path", "gen_python"]):
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Required path '{p}' does not exist or is not a directory. "
                                f"Please ensure QES is installed correctly.")
    print(f"-> Adding to sys.path - {label}: {p}")
    sys.path.insert(0, str(p))
    
# -----------------------------------------------------------------------
# Now import the required QES modules
# -----------------------------------------------------------------------

try:
    from QES.general_python.algebra.utils import get_backend, log as logger
except ImportError as e:
    raise ImportError(f"Failed to import QES modules. Ensure QES is installed and QES_PYPATH is set correctly.\nOriginal error: {e}")

# NQS
try:
    #! NQS
    from QES.NQS.nqs                                            import *
    #! Models
    from QES.Algebra.Model.Interacting.Spin.heisenberg_kitaev   import HeisenbergKitaev, HoneycombLattice

except ImportError as e:
    raise ImportError(f"Failed to import QES NQS modules. Ensure QES is installed and QES_PYPATH is set correctly.\nOriginal error: {e}")
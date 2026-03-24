import sys
sys.path.insert(0, "Python")

from QES.NQS.nqs import NQS

# see if density matrix type is implemented in model?
import inspect
from QES.NQS.src.nqs_physics import DensityMatrixPhysics

print("DensityMatrixPhysics methods:")
for name, method in inspect.getmembers(DensityMatrixPhysics, inspect.isfunction):
    print(name)

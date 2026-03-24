import sys
sys.path.insert(0, "Python")

from QES.NQS.src.nqs_physics import DensityMatrixPhysics
import inspect

# Look at DensityMatrixPhysics methods and what it has access to
print(inspect.getsource(DensityMatrixPhysics))

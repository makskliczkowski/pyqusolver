import sys
import numpy as np
sys.path.insert(0, "Python")
from QES.NQS.nqs import NQS

# create mock for test
class MockTrainer:
    def __init__(self):
        from QES.NQS.src.nqs_train import NQSTrainStats
        self.stats = NQSTrainStats()

class MockModel:
    def __init__(self):
        self.ns = 4
        self.eig_val = None
        self.eigenvalues = np.array([1.0, 2.0])
    def diagonalize(self, **kwargs):
        self.eig_val = self.eigenvalues

class MockProblem:
    def __init__(self):
        self.typ = "density_matrix"

class MockNQS:
    def __init__(self):
        self._logger = None
        self._nqsproblem = MockProblem()
        self.trainer = MockTrainer()
        self.model = MockModel()
        self._nthstate = 0

from QES.NQS.src.nqs_exact import get_exact_impl

try:
    get_exact_impl(MockNQS())
except Exception as e:
    print(e)

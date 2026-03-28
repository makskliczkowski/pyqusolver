from QES.Algebra.Hilbert.hilbert_base import BaseHilbertSpace
import numpy as np

class MockLocalSpace:
    def __init__(self, typ="spin-1/2"):
        self.typ = typ
    def __str__(self):
        return self.typ

class MockSpace(BaseHilbertSpace):
    def __init__(self):
        self._ns = 10
        self._nh = 1024
        self._local_space = MockLocalSpace()
        self._sym_container = None
        self._global_syms = []
        self._has_complex_symmetries = False
        self._logger = None

    @property
    def check_global_symmetry(self):
        return len(self._global_syms) > 0

    @property
    def state_convention(self):
        return {"name": "test_conv"}

s = MockSpace()
print("STR OUTPUT:", str(s))
print("SYM OUTPUT:", s.get_sym_info())

s._global_syms = [type("MockSym", (), {"name": "U1", "get_val": lambda self: 1, "get_name_str": lambda self: "U1"})()]
print("STR OUTPUT WITH SYM:", str(s))
print("SYM OUTPUT WITH SYM:", s.get_sym_info())

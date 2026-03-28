import timeit

class TestHilbertSpace:
    def __init__(self):
        self._ns = 100
        self._nh = 100000
        self._local_space = "spin-1/2"
        self.state_convention = {"name": "z-basis"}

        # mock symmetries
        self._sym_container = type("SymContainer", (), {
            "generators": [("op", (f"S_{i}", i)) for i in range(5)]
        })()

        self.check_global_symmetry = True
        self._global_syms = [type("GlobalSym", (), {"name": f"G_{i}", "get_name_str": lambda self, j=i: f"G_{j}", "get_val": lambda self: 1})() for i in range(5)]

    def get_sym_info_old(self) -> str:
        tmp = ""
        if self._sym_container is not None and self._sym_container.generators:
            for op, (gen_type, sector) in self._sym_container.generators:
                tmp += f"{gen_type}={sector},"

        if self.check_global_symmetry:
            for g in self._global_syms:
                tmp += f"{g.get_name_str() if hasattr(g, 'get_name_str') else g.name}={g.get_val() if hasattr(g, 'get_val') else ''},"

        return tmp[:-1] if tmp else ""

    def get_sym_info_new(self) -> str:
        parts = []
        if self._sym_container is not None and getattr(self._sym_container, "generators", None):
            parts.extend(f"{gen_type}={sector}" for _, (gen_type, sector) in self._sym_container.generators)

        if self.check_global_symmetry:
            parts.extend(f"{g.get_name_str() if hasattr(g, 'get_name_str') else g.name}={g.get_val() if hasattr(g, 'get_val') else ''}" for g in self._global_syms)

        return ",".join(parts)

    def str_old(self):
        info = f"HilbertSpace: Ns={self._ns}, Nh={self._nh}"
        if self._local_space:
            info += f", Local={self._local_space}"
            try:
                conv_name = self.state_convention.get("name", "unknown")
                info += f", StateConvention={conv_name}"
            except Exception:
                pass
        sym_info = self.get_sym_info_old()
        if sym_info:
            info += f", Symmetries=[{sym_info}]"
        return info

    def str_new(self):
        parts = [f"HilbertSpace: Ns={self._ns}, Nh={self._nh}"]
        if self._local_space:
            parts.append(f"Local={self._local_space}")
            try:
                conv_name = self.state_convention.get("name", "unknown")
                parts.append(f"StateConvention={conv_name}")
            except Exception:
                pass
        sym_info = self.get_sym_info_new()
        if sym_info:
            parts.append(f"Symmetries=[{sym_info}]")
        return ", ".join(parts)


t = TestHilbertSpace()
print("str_old: ", t.str_old())
print("str_new: ", t.str_new())

# Timing
n = 100000
print("str_old:", timeit.timeit(t.str_old, number=n))
print("str_new:", timeit.timeit(t.str_new, number=n))

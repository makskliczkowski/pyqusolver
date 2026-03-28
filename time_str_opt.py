import time
import timeit

def orig_get_sym_info(sym_container, global_syms):
    tmp = ""
    if sym_container is not None and getattr(sym_container, "generators", None):
        for op, (gen_type, sector) in sym_container.generators:
            tmp += f"{gen_type}={sector},"

    if len(global_syms) > 0:
        for g in global_syms:
            tmp += f"{g.get_name_str() if hasattr(g, 'get_name_str') else g.name}={g.get_val() if hasattr(g, 'get_val') else ''},"

    return tmp[:-1] if tmp else ""

def new_get_sym_info(sym_container, global_syms):
    parts = []
    if sym_container is not None and getattr(sym_container, "generators", None):
        parts.extend(f"{gen_type}={sector}" for _, (gen_type, sector) in sym_container.generators)

    if len(global_syms) > 0:
        parts.extend(f"{g.get_name_str() if hasattr(g, 'get_name_str') else g.name}={g.get_val() if hasattr(g, 'get_val') else ''}" for g in global_syms)

    return ",".join(parts)


class MockSymContainer:
    def __init__(self, n):
        self.generators = [("op", (f"gen_{i}", i)) for i in range(n)]

class MockGlobalSym:
    def __init__(self, i):
        self.name = f"g_{i}"
    def get_name_str(self):
        return self.name
    def get_val(self):
        return 1

sym = MockSymContainer(5)
glob = [MockGlobalSym(i) for i in range(5)]

print(orig_get_sym_info(sym, glob))
print(new_get_sym_info(sym, glob))

t1 = timeit.timeit(lambda: orig_get_sym_info(sym, glob), number=100000)
t2 = timeit.timeit(lambda: new_get_sym_info(sym, glob), number=100000)

print(f"Original: {t1:.4f}s")
print(f"New: {t2:.4f}s")

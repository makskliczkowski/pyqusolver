import timeit

setup_code = """
symmetries = ['translation', 'reflection', 'parity_z', 'parity_x', 'parity_y']
"""

test_list = """
for sym in symmetries:
    sym_lower = sym.lower().replace("_", "").replace("-", "")
    if sym_lower in ["translation", "momentum", "trans"]:
        pass
    elif sym_lower in ["reflection", "parityspatial", "spatial", "mirror"]:
        pass
    elif sym_lower in ["inversion", "inv", "spatialinversion"]:
        pass
    elif sym_lower in ["parityz", "spinparity", "pz", "spinflip"]:
        pass
    elif sym_lower in ["parityx", "px"]:
        pass
    elif sym_lower in ["parityy", "py"]:
        pass
"""

test_tuple = """
for sym in symmetries:
    sym_lower = sym.lower().replace("_", "").replace("-", "")
    if sym_lower in ("translation", "momentum", "trans"):
        pass
    elif sym_lower in ("reflection", "parityspatial", "spatial", "mirror"):
        pass
    elif sym_lower in ("inversion", "inv", "spatialinversion"):
        pass
    elif sym_lower in ("parityz", "spinparity", "pz", "spinflip"):
        pass
    elif sym_lower in ("parityx", "px"):
        pass
    elif sym_lower in ("parityy", "py"):
        pass
"""

test_set = """
for sym in symmetries:
    sym_lower = sym.lower().replace("_", "").replace("-", "")
    if sym_lower in {"translation", "momentum", "trans"}:
        pass
    elif sym_lower in {"reflection", "parityspatial", "spatial", "mirror"}:
        pass
    elif sym_lower in {"inversion", "inv", "spatialinversion"}:
        pass
    elif sym_lower in {"parityz", "spinparity", "pz", "spinflip"}:
        pass
    elif sym_lower in {"parityx", "px"}:
        pass
    elif sym_lower in {"parityy", "py"}:
        pass
"""

n = 1000000
time_list = timeit.timeit(test_list, setup=setup_code, number=n)
time_tuple = timeit.timeit(test_tuple, setup=setup_code, number=n)
time_set = timeit.timeit(test_set, setup=setup_code, number=n)

print(f"Time with list:  {time_list:.4f} seconds")
print(f"Time with tuple: {time_tuple:.4f} seconds")
print(f"Time with set:   {time_set:.4f} seconds")

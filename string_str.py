import timeit

def orig():
    tmp = ""
    for op, (gen_type, sector) in [("o", ("a", 0)), ("o", ("b", 1)), ("o", ("c", 2))]:
        tmp += f"{gen_type}={sector},"
    return tmp[:-1] if tmp else ""

def new():
    return ",".join(f"{gen_type}={sector}" for _, (gen_type, sector) in [("o", ("a", 0)), ("o", ("b", 1)), ("o", ("c", 2))])

def new_list():
    return ",".join([f"{gen_type}={sector}" for _, (gen_type, sector) in [("o", ("a", 0)), ("o", ("b", 1)), ("o", ("c", 2))]])

print(timeit.timeit(orig, number=1000000))
print(timeit.timeit(new, number=1000000))
print(timeit.timeit(new_list, number=1000000))

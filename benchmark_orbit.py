import time

# Mock translator
def translator(state):
    # simple bit shift for 10 bits
    return ((state << 1) & 1023) | (state >> 9), None

states = list(range(1024))

def with_set():
    orbits = []
    for state in states:
        orbit = []
        seen = set()
        cur = state
        while cur not in seen:
            orbit.append(cur)
            seen.add(cur)
            cur, _ = translator(cur)
        orbits.append(orbit)
    return orbits

def without_set():
    orbits = []
    for state in states:
        orbit = [state]
        cur, _ = translator(state)
        while cur != state:
            orbit.append(cur)
            cur, _ = translator(cur)
        orbits.append(orbit)
    return orbits

if __name__ == "__main__":
    t0 = time.time()
    for _ in range(100):
        with_set()
    t1 = time.time()
    print(f"With set: {t1 - t0:.4f}s")

    t0 = time.time()
    for _ in range(100):
        without_set()
    t1 = time.time()
    print(f"Without set: {t1 - t0:.4f}s")

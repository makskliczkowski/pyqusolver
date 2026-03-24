with open('Python/QES/NQS/src/nqs_exact.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'if nqs_instance._nqsproblem.typ == "wavefunction":' in line:
            print("Start:", i)
        if 'else:' in line and 'raise NotImplementedError' in lines[i+1]:
            print("End:", i)

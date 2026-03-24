import sys
sys.path.insert(0, "Python")

with open('Python/QES/NQS/src/nqs_exact.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'if nqs_instance._nqsproblem.typ == "wavefunction":' in line:
            start_index = i
            break

print("Code block starting at: ", start_index)
for line in lines[start_index:start_index+10]:
    print(line, end='')

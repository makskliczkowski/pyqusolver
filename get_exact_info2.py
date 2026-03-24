import sys
sys.path.insert(0, "Python")

with open('Python/QES/NQS/src/nqs_exact.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'else:' in line and 'raise NotImplementedError' in lines[i+1]:
            end_index = i
            break

print("Code block ending at: ", end_index)
for line in lines[end_index:end_index+4]:
    print(line, end='')

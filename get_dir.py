import sys
sys.path.insert(0, "Python")

with open('Python/QES/general_python/common/directories.py', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'def parent' in line or 'property' in line and 'parent' in line:
            print(line.strip())

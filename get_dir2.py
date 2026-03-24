with open('Python/QES/general_python/common/directories.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'def parent' in line:
            for l in lines[i-1:i+5]:
                print(l.strip())

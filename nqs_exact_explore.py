with open('Python/QES/NQS/src/nqs_exact.py', 'r') as f:
    lines = f.readlines()
    for line in lines[42:99]:
        print(line, end='')

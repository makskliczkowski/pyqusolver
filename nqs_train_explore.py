with open('Python/QES/NQS/src/nqs_train.py', 'r') as f:
    for line in f.readlines():
        if 'class NQSTrainStats' in line or 'has_exact' in line or 'exact_predictions' in line:
            print(line.strip())

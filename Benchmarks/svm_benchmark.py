#!/usr/bin/env python

from sklearn import svm
import csv_io
import math

def main():
    train = csv_io.read_data("../Data/train.csv")
    target = [x[0] for x in train]
    train = [x[1:] for x in train]
    test = csv_io.read_data("../Data/test.csv")

    svc = svm.SVC(probability=True)
    svc.fit(train, target)
    predicted_probs = svc.predict_proba(test)
    predicted_probs = ["%f" % x[1] for x in predicted_probs]
    csv_io.write_delimited_file("../Submissions/svm_benchmark.csv",
                                predicted_probs)

if __name__=="__main__":
    main()

#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
import csv_io
import math

def main():
    train = csv_io.read_data("../Data/train.csv")
    target = [x[0] for x in train]
    train = [x[1:] for x in train]
    test = csv_io.read_data("../Data/test.csv")

    rf = RandomForestClassifier(n_estimators=100, min_split=2)
    rf.fit(train, target)
    predicted_probs = rf.predict_proba(test)
    predicted_probs = ["%f" % x[1] for x in predicted_probs]
    csv_io.write_delimited_file("../Submissions/rf_benchmark.csv",
                                predicted_probs)

if __name__=="__main__":
    main()

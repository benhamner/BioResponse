#!/usr/bin/env python

import csv_io
import numpy as np

def main():
    train = csv_io.read_data("../Data/train.csv")
    targets = [int(x[0]) for x in train]
    num_targets = len(targets)
    num_ones = np.sum(targets)
    optimized_value = float(num_ones) / num_targets

    test = csv_io.read_data("../Data/test.csv")
    
    predicted_probs = ["%f" % optimized_value for x in test] 
    csv_io.write_delimited_file("../Submissions/optimized_value_benchmark.csv",
                                predicted_probs)

if __name__=="__main__":
    main()

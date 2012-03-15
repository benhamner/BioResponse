#!/usr/bin/env python

import csv_io
import numpy as np

def main():
    test = csv_io.read_data("../Data/test.csv")
    
    predicted_probs = ["%f" % .5 for x in test] 
    csv_io.write_delimited_file("../Submissions/uniform_benchmark.csv",
                                predicted_probs)

if __name__=="__main__":
    main()

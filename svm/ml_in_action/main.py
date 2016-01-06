#coding=utf8

import numpy as np
import logging

from loaddata import load_data_set
from new_svm import SVM

logging.basicConfig(level=logging.INFO)

def main() :
    fname = "data/testSet.txt"
    svm = SVM()
    svm.load_data(fname)
    svm.SMOTraining()
    decorate_line = "---------"
    print "{0}predict point (1,1){1}".format(decorate_line , decorate_line)
    y = svm.predict([1,1])
    print np.sign(y)
    print "{0}w(matrix) and b{1}".format(decorate_line , decorate_line)
    print svm.get_w()
    print svm.get_b()
    print "{0}support vectors{1}".format(decorate_line , decorate_line)
    print svm.get_support_vectors()

if __name__ == "__main__" :
    main()

#coding=utf8

import logging

def load_data_set(fname) :
    try :
        f = open(fname)
    except IOError , e :
        logging.info("failed to open file '{fname}'".format(fname=fname))
        exit(1)
    X = []
    Y = []
    for line in f :
        line_parts = line.strip().split()
        X.append(map(float , line_parts[:-1]))
        Y.append(float(line_parts[-1]))
    return X , Y


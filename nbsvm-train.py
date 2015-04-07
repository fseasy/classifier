#!/usr/bin/env python
#coding=utf-8

import os
import sys
import argparse
import numpy as np
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
def main(postrain , negtrain , ngram , out) :
    ngram = int(ngram)
    print postrain
    print negtrain
    print ngram
    print out
    logging.info("counting")
    logging.info("abstract features")
    logging.info("compute log-count ratio")
    logging.info("generate training data in libSVM format")
    logging.info("generate model using libLinear")
    logging.info("compute w and b")
    logging.info("output model")

    postrain.close()
    negtrain.close()
if __name__ == "__main__" :
    '''
    usage:
    python nvsvm-train.py --postrain /path/to/positive_train_file\
    --negtrain /path/to/negtive_train_file\
    --ngram 1/2 --out /path/to/model_file
    '''
    parser = argparse.ArgumentParser(description="Run NBSVM train program")
    parser.add_argument('--postrain',help="path to positive train data",type=argparse.FileType('r'),default="data/postrain")
    parser.add_argument('--negtrain',help="path to negtive train data",type=argparse.FileType('r'),default="data/negtrain")
    parser.add_argument('--ngram',help="1 or 2 to decide using the unigram or bigram",type=int,default="2",choices=[1,2])
    parser.add_argument('--out',help="path to model file ",default="")
    args = vars(parser.parse_args())
    
    main(**args)

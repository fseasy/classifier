#!/usr/bin/env python
#coding=utf-8

import os
import sys
import argparse
import numpy as np
from collections import Counter
import logging

from fileprocessing import *

logging.basicConfig(level=logging.INFO)

def counting(f_obj , ngram) :
    '''
    input >  f_obj : file obj for read , every line represent a doc
             ngram : 1 or 2 stands for using unigram or bigram as the feature
    return > a Counter ,  storing all the grams and corresponding DF value at the f_obj
    
    description >
        here , we use the binarized vector , so just recode the gram occurence at a doc , that is to say :
        recorde the DF of every gram
    '''
    tokens_df_con = Counter()
    for line in f_obj.xreadlines() :
        tokens = tokenize(line , ngram)
        tokens = list(set(tokens))
        tokens_df_con.update(tokens)
    return tokens_df_con

def main(postrain , negtrain , ngram , out) :
    
    #print postrain
    #print negtrain
    #print ngram
    #print out
    logging.info("counting")
    pos_con = counting(postrain , ngram)
    neg_con = counting(negtrain , ngram)
    print pos_con
    print neg_con
    #logging.info("abstract features")
    #logging.info("compute log-count ratio")
    #logging.info("generate training data in libSVM format")
    #logging.info("generate model using libLinear")
    #logging.info("compute w and b")
    #logging.info("output model")

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

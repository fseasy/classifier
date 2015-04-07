#!/usr/bin/env python
#coding=utf-8
import numpy as np
import os
import sys
import argparse
import logging
try :
    import cPickle as pickle
except :
    import pickle

from fileprocessing import *
logging.basicConfig(level=logging.INFO)

def load_model(model) :
    w = pickle.load(model)
    b = pickle.load(model)
    dic = pickle.load(model)
    r = pickle.load(model)
    ngram = pickle.load(model)
    return w,b,dic,r,ngram


def NBSVM_predict(X,w,b) :
    dimension = w.shape[0]
    w.shape = (dimension , 1) # transpose
    data_size = len(X)
    X_m = np.zeros((data_size , dimension))
    # build the matrix
    # X = [ {idx:val , idx:val}  ]
    i = 0
    for x in X :
        print x
        for idx in x :
            X_m[i,idx -1 ] = x[idx] # the idx is the the feature idx start from 1 , where matrix start from 0 . and x is a dict .
        i += 1
    #print X_m
    #print w
    
    #predict function : y = sign(x*w + b )
    #here X_m 's shape is (data_size,dimension) , w is (dimension,1) , b will be broadcasting to (data_size,1) , result is matrix with shape (data_size , 1)
    Y_pre = np.dot(X_m , w ) + b ; # mul is not the star *  but the numpy.dot !!
    #print Y_pre
    #print np.sign(Y_pre)
    return np.sign(Y_pre)

def evaluation_using_liblinear(ty,py) :
    return linearutil.evaluations(ty,py)




def main(postest,negtest,model) :
    logging.info("loading model from '%s'" %(model.name))
    w,b,dic,r,ngram = load_model(model)
    w = np.array(w)

    #print w
    #print b
    #print dic
    #print r
    #print ngram
    logging.info('vectorize test file')
    pos_vec = vectorize_docs(postest,dic,r,ngram)
    neg_vec = vectorize_docs(negtest,dic,r,ngram)
    Y , X = ready_SVM_data([POSITIVE_LABEL , NEGATIVE_LABEL] , [pos_vec , neg_vec]) 
    #print Y
    #print X
    Y_predict = NBSVM_predict(X,w,b)
    ACC , MSE , SCC = evaluation_using_liblinear(Y,Y_predict)
    print ACC
    print MSE
    print SCC

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="using NBSVM to predict")
    parser.add_argument("--postest",help="path to positive test file" , default="data/postest" , type=argparse.FileType('r'))
    parser.add_argument("--negtest",help="path to negative test file" , default="data/negtest" , type=argparse.FileType('r'))
    parser.add_argument("--model",help="path to NBSVM model",default="out.model" , type=argparse.FileType('r'))
    parser.add_argument("--liblinear",help="path to liblinear",default="/home/xx/bin/liblinear-1.96/python")

    args = vars(parser.parse_args())
    liblinear_path = args.pop('liblinear')
    if not os.path.exists(liblinear_path) :
        raise Exception
    sys.path.append(liblinear_path)
    import liblinearutil as linearutil
    
    main(**args)


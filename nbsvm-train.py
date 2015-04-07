#!/usr/bin/env python
#coding=utf-8

import os
import sys
import argparse
import numpy as np
from collections import Counter
import logging
try :
    import cPickle as pickle
except :
    import pickle

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

def abstract_features(poscon , negcon) :
    '''
    input > poscon : positive token Container 
            negcon : negative token Container
    output > dict , key-val as : token:index ; index from 1 to len(dict)
    '''
    keys = list(set(poscon.keys() + negcon.keys()))
    dic = {}
    idx = 1
    for k in keys :
        dic[k] = idx 
        idx += 1
    return dic

def compute_logcount_ratio(dic , pos_con , neg_con , alpha=1) :
    '''
    input > dic : feature dict
            pos_con : positive set DF container
            neg_con : negative set DF container
            alpha : smoothing parameter
    return > r , vector of log-count ratio
    '''
    vector_d = len(dic)
    alpha = np.ones(vector_d) * alpha
    p_v = np.zeros(vector_d)
    q_v = np.zeros(vector_d)
    # calculate p , q . 
    # p = sum(f_i) where f_i is the document vector belongs to positive set , and f_i_j is the binarized value , so sum(f_i_j) for all i equal to the occurence of feature f_j at the positive set . that is , the DF of the f_j , equal to pos_con[f_j]
    for f in pos_con :
        f_idx = dic[f] - 1
        f_df = pos_con[f]
        p_v[f_idx] = f_df
    for f in neg_con :
        q_v[dic[f] - 1] = neg_con[f]
    # add the smoothing parameter
    p_v = p_v + alpha ;
    q_v = q_v + alpha ;
    # calc the normalization num by 1st norm
    p_v_1_norm = abs(p_v).sum()
    q_v_1_norm = abs(q_v).sum()
    #calc log-cont ratio
    r = np.log((p_v/p_v_1_norm)/(q_v/q_v_1_norm))
    return r

def train_using_liblinear(Y , X , options) :
    '''
    input > Y : labels 
            X : document vectors in sparse format , ie : [{1:3,5:10},{...}]
            options : libLinear train options
    return > m , model of SVM

    just call SVM train function
    '''
    m = train(Y,X,options)
    return m

def compute_NBSVM_param(m , beta=0.25) :
    '''
    intput > m : model of libLinear
             beta : smoothing parameter
    return > w : weight vector
             b : bias
    w' = (1 - beta)*w_mean + beta*w
    where w_mean = norm1(w) / size(V) , V is the feature space
    '''
    dimension = m.get_nr_feature()
    labels = m.get_labels()
    #we need get the label idx for positive label
    label_idx = 0 
    for i in range(0,len(labels)) :
        if labels[i] == POSITIVE_LABEL :
            label_idx = i
            break

    w , b = m.get_decfun(label_idx)
    w = np.array(w)
    w_norm1 = abs(w).sum()
    w_mean = w_norm1 / dimension 
    w_new = (1 - beta) * w_mean * np.ones(dimension) + beta * w 
    return list(w_new) , b

def output_model(w,b,dic,r,ngram,o_obj) :
    pickle.dump(w , o_obj)
    pickle.dump(b , o_obj)
    pickle.dump(dic , o_obj)
    pickle.dump(r , o_obj)
    pickle.dump(ngram , o_obj)
    

def main(postrain , negtrain , ngram , out) :
    
    #print postrain
    #print negtrain
    #print ngram
    #print out
    logging.info("counting")
    pos_con = counting(postrain , ngram)
    neg_con = counting(negtrain , ngram)
    #print pos_con
    #print neg_con
    logging.info("abstract features")
    dic = abstract_features(pos_con , neg_con)
    print dic
    logging.info("compute log-count ratio")
    r = compute_logcount_ratio(dic , pos_con , neg_con )
    print r
    
    logging.info("generate training data in libSVM format")
    postrain.seek(0,os.SEEK_SET) # the file has been read to the end , so move to the head
    negtrain.seek(0,os.SEEK_SET)
    pos_f_vecs = vectorize_docs(postrain , dic , r , ngram)
    neg_f_vecs = vectorize_docs(negtrain , dic , r , ngram)
    #print pos_f_vecs
    #print neg_f_vecs
    Y , X = ready_SVM_data([POSITIVE_LABEL,NEGATIVE_LABEL],[pos_f_vecs,neg_f_vecs])
    print Y
    print X
    logging.info("generate model using libLinear")
    
    m = train_using_liblinear(Y,X,"-s 1 -c 1") # -s 1 : using L2-resularized L2-loss SVM ; -c 1 : C = 1
    logging.info("compute w and b")
    w , b = compute_NBSVM_param(m)

    logging.info("output model")
    
    output_model(w , b , dic , r , ngram , out)
    postrain.close()
    negtrain.close()
    out.close()

    logging.info("FINISHED")


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
    parser.add_argument('--out',help="path to model file ",type=argparse.FileType('w') , default="out.model")
    parser.add_argument('--liblinear',help="" , default="/home/xx/bin/liblinear-1.96/python")
    args = vars(parser.parse_args())
    if not os.path.exists(args['liblinear']) :
        raise Exception
    sys.path.append(args['liblinear'])
    args.pop('liblinear') # the follow does't need it any more , delete it for a call
    from liblinearutil import * # it may be not good  - -

    main(**args)

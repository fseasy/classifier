#!/usr/bin/env python
#coding=utf-8

def tokenize(sentence , ngram) :
    words = sentence.split()
    cnt = len(words)
    tokens = []
    # get gram from 1 to ngram
    for gram in range(1,ngram+1) :
        for i in range(0,cnt - gram) :
            token_l = words[i:i+gram]
            tokens.append('_'.join(token_l))
    return tokens

def vectorize_file(f_obj , dic , r , ngram) :
    f_vector = [] 
    for line in f_obj.xreadlines() :
        tokens = tokenize(line , ngram)
        # for this , just need the occurence of every token
        tokens = list(set(tokens))
        index = []
        for token in tokens :
            if token in dic :
                index.append(dic[token])
        index.sort() # let the index arranged from small to big
        line_vector = []
        for i in index :
            line_vector.append((i,r[i-1])) # f = r * f_ori 
        f_vector.append(line_vector)
    return f_vector

if __name__ == "__main__" :
    dic = {
            "i":1 ,
            "is_a":2
            }
    r = [23,9]
    v = vectorize_file(open("data/postrain") , dic , r , 2)
    print v

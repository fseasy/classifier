#coding=utf8

import numpy as np
from loaddata import load_data_set

def smp_simple( X , Y , C , toler , max_iter) :
    X_mat = np.mat(X)
    Y_mat = np.mat(Y).transpose()
    samples_num , dimentsion_num = np.shape(X_mat)
    alphas = np.zeros((samples_num , 1)) # alpha vector for SVM
    b = 0 # bias for SVM
    product_mat = X_mat * X_mat.T # pre-computing X * X
    iter_num = 0 
    while iter_num < max_iter :
        alpha_paris_changed = 0
        for i in range(samples_num) :
            # functional distance , or f_x = \sum_{j=1}^m { \alpha_j * y_j * x_j * x_i } + b
            #                          in matrix representation , f_x = Product_X_i * ( alpha .* y ) , here 
            #                          Product_X_i = (X * X^T)[i,:] , that is the i-th x dot product others x 's vector
            func_dis = product_mat[i,:] * np.multiply(alpha , Y_mat ) + b
            E_i = func_dis - Y_mat[i,0] # row i col 0 of Y_mat is the y_i








def main() :
    X , Y = load_data_set("data/testSet.txt")
    

if __name__ == "__main__" :
    main()

#coding=utf8

import numpy as np
import logging

class SVM(object) :
    def __init__(self , cost=10 ) :
        self.alphas = None
        self.b = 0
        self.X = None 
        self.Y = None
        self.sample_num = 0 # num of samples
        self.dim_num = 0 # num of dimension
        self.kernel_result = None
        self.C = cost
        self.alphas_update_cnts = None

    def load_data(self , fname) :
        X = []
        Y = []
        with open(fname) as f :
            for line in f :
                line_parts = line.strip().split()
                X.append(map(float , line_parts[:-1]))
                Y.append(float(line_parts[-1]))
        self.Y = np.array(Y)
        self.X = np.mat(X)
        self.sample_num , self.dim_num = np.shape(self.X)
        self.kernel_result = self.X * self.X.T # just using linear , that is , X * X^T
                                               # it is a symmetric matrix . each ith-row or ith-col is the x_i * x_j , j = 0,1,...,m-1 result
        self.alphas = np.zeros(self.sample_num)
        #self.alphas = np.random.random(self.sample_num)
        self.alphas_update_cnts = np.zeros(self.sample_num)
    
    def predict(self , x ) :
        if type(x) == list :
            x = np.mat(x).transpose()
        elif type(x) == np.ndarray :
            x = np.mat(x)
        if x.shape[0] == 1 :
            x = x.transpose()
        alpha_y = np.multiply(self.alphas , self.Y)
        alpha_y_mat = np.mat(alpha_y) # 1 x m
        y = np.dot( alpha_y_mat , np.dot(self.X ,  x) )[0,0] + self.b # ( 1 x m ) * ( m x n * n x 1 ) = 1
        return y
    
    def _predict_inner_sample(self , ith) :
        alpha_y = np.multiply(self.alphas , self.Y)
        alpha_y_mat = np.mat(alpha_y) # 1 x m
        y = np.dot( self.kernel_result[ith] , alpha_y_mat.T )[0,0] + self.b # ( 1 x m ) * ( m x 1 ) = 1
        return y

    def _is_violate_KKT(self , i , epsilon) :
        predict_yi = self._predict_inner_sample(i)
        true_yi = self.Y[i]
        if (   ( predict_yi * true_yi < 1 - epsilon and self.alphas[i] < self.C ) 
                or ( predict_yi * true_yi > 1 + epsilon and self.alphas[i] > 0  ) ) : 
            return True
        else :
            return False

    def _calc_e_ith(self , ith) :
        return self._predict_inner_sample(ith) - self.Y[ith]
    
    def _select_alpha_j(self , i , ei ) :
        has_visited = np.nonzero(self.alphas_update_cnts)[0]
        max_diff = -1
        j = i
        for test_j in has_visited :
            if test_j == i : continue
            et = self._calc_e_ith(test_j)
            diff = abs(ei - et)
            if diff > max_diff :
                max_diff = diff
                j = test_j
        while j == i : # if j == i , we randomize it  
            j = np.random.randint(self.sample_num)
        ej = self._calc_e_ith(j)
        return j , ej

    def _update_alpha_ij(self , i , j , ei , ej) :
        '''
        update alhpa_i and alpha_j , b .
        return : True , if update ok
                 False , update failed .
        '''
        alphai_old = self.alphas[i]
        alphaj_old = self.alphas[j]
        yi = self.Y[i]
        yj = self.Y[j]
        s = yi * yj
        kii = self.kernel_result[i,i]
        kjj = self.kernel_result[j,j]
        kij = self.kernel_result[i,j]
        alphaj_new = alphaj_old + yj * (ei - ej) / (kii + kjj - 2 * kij)
        # clip according to the range
        L = max(0 , alphaj_old + s * alphai_old - 0.5 * ( s + 1) * self.C)
        H = min(self.C , alphaj_old + s * alphai_old - 0.5 * ( s - 1) * self.C)
        if L == H :
            return False
        if alphaj_new < L :
            alphaj_new = L
        elif alphaj_new > H :
            alphaj_new = H
        if abs(alphaj_new - alphaj_old ) < 1E-5 :
            # no update !
            return False
        alphai_new = alphai_old + s*(alphaj_old - alphaj_new)
        # update alphas
        self.alphas[i] = alphai_new
        self.alphas[j] = alphaj_new
        # udpate b
        bi = -ei - yi * kii * (alphai_new - alphai_old) - yj * kij * ( alphaj_new - alphaj_old) + self.b
        bj = -ej - yj * kjj * (alphaj_new - alphaj_old) - yi * kij * ( alphai_new - alphai_old) + self.b
        if 0 < alphai_new < self.C :
            self.b = bi 
        elif 0 < alphaj_new < self.C :
            self.b = bj
        else :
            self.b = (bi + bj) / 2
        return True

    def SMOTraining(self , max_ite = 1000 , epsilon=1E-3) :
        '''
        SMO implementation .
        return > 1 : exit with KKT condition is all fulfilled
                 0 : maximum iterate num  is reached ! (KKT is not fulfilled)
        '''
        ite_cnt = 0
        update_pair_cnt= 0
        pass_for_all = True # if True , select alpha_i from entire dataset . Else , just from the non-bound ( C \not \in {0 , C})
        while ite_cnt < max_ite :
            update_pair_cnt = 0
            # ready candidate samples for selection of alpha_i (also for x_i)
            candidate_idx = range(self.sample_num) # entire dataset
            if not pass_for_all :
                candidate_idx = np.nonzero( ( self.alphas < self.C ) & ( self.alphas > 0 ) )[0] # get idx which is not at bound 
                #                                                                                      or , in the hyperplane wx + b = +- 1 
            for i in candidate_idx :
                # First , we should check if x_i has violate the KKT conditions
                if not self._is_violate_KKT(i , epsilon) : 
                    continue
                # here i is the x_i 's idx 
                # we should find a j , that |E_i - E_j| has the max vlaue  
                ei = self._calc_e_ith(i)
                j , ej = self._select_alpha_j(i , ei)
                # i , j has been selected , update it !
                update_state = self._update_alpha_ij(i , j , ei , ej)
                self.alphas_update_cnts[i] += 1
                self.alphas_update_cnts[j] += 1
                if update_state == True :
                    update_pair_cnt += 1
            if pass_for_all :
                if update_pair_cnt > 0 :
                    pass_for_all = False # next outer circle , we select alpha from non-bound
                else :
                    # no alpha is updated enven for entire dataset
                    # that is , all KKT conditions is fulfilled !
                    logging.info("After {ite} iterations , SMO Training done .".format( ite=(ite_cnt + 1) ))
                    return 1
            else :
                # pass for non-bound
                if update_pair_cnt == 0 :
                    # all KKT at non-bound  is fullfilled . 
                    # next circle should go pass entire dataset .
                    pass_for_all = True 
            ite_cnt += 1
            if ite_cnt % 100 == 0 :
                logging.info("{ite} iterations has been done .".format(ite=ite_cnt))
        return 0
    
    def get_w(self) :
        alpha_y = np.multiply(self.alphas , self.Y)
        alpha_y_mat = np.mat(alpha_y).transpose() # 1 x m
        w = np.dot(self.X.T , alpha_y_mat)
        return w

    def get_b(self) :
        return self.b

    def get_support_vectors(self) :
        idx = np.nonzero( ( self.alphas > 0 ) & ( self.alphas < self.C ) )
        return self.X[idx]

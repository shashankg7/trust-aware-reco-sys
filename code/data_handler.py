#! /usr/bin/env python

import numpy as np
from scipy.io import loadmat
import pdb

class data_handler():
    
    def __init__(self, rating_path, trust_path):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.n_users = 0
        self.n_prod = 0
        self.n_cat = 6
    
    def get_stats(self):
        return self.n_users, self.n_prod, self.n_cat

    def load_matrices(self):
        # Loading Matrices from data
        f1 = open(self.rating_path)
        f2 = open(self.trust_path)
        R = loadmat(f1)
        W = loadmat(f2)
        # Converting R and W from dictionary to array
        R = R['rating_with_timestamp']
        W = W['trust']
        self.n_users = max(R[:, 0])
        self.n_prod = max(R[:, 1])
        # Selecting entries with the 6 categories given in the paper
        cat_id = [7, 8, 9, 10, 11, 19]
        R = R[np.in1d(R[:, 2],cat_id)]
        R = R[R[:, 5].argsort()]
        R_size = R.shape[0]
        # Choosing 70% data for training and rest for testing
        R_train = R[:R_size*0.7]
        R_test = R[R_size*0.7:]
        # Making all eligible Product-Category pairs
        prod_cat = np.unique(R_train[:, 1:3])
        # Making the mu matrix
        mu = {}
        for cat in cat_id:
            cat_rating = R_train[np.where(R_train[:, 2] == cat), 3]
            mu[cat] = np.mean(cat_rating)
        pdb.set_trace()
        return R_train, R_test, prod_cat, mu
            
if __name__ == "__main__":
    data = data_handler("../data/rating_with_timestamp.mat", "../data/trust.mat")
    R_train, R_test, PF_pair, mu = data.load_matrices()
    print "done"

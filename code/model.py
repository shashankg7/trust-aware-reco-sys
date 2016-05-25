#! /usr/bin/env python

import numpy as np
from data_handler import data_handler
import tables as tb
import math
import pdb

class baseline_tensor():

    def __init__(self):
        # Loading matrices from given data
        data = data_handler("../data/rating_with_timestamp.mat", "../data/trust.mat")
        self.R_train, self.R_test, self.W, self.PF, self.mu = data.load_matrices()
        # Getting unique users and products used in Training data
        self.prod = np.unique(self.R_train[:, 1])
        self.users = np.unique(self.R_train[:, 0])
        self.n_users, self.n_prod, self.n_cat = data.get_stats()
        common_users = np.intersect1d(self.R_test[:, 0], self.users)
        common_prod = np.intersect1d(self.R_test[:, 1], self.prod)
        self.R_test = self.R_test[np.in1d(self.R_test[:, 0], common_users)]
        self.R_test = self.R_test[np.in1d(self.R_test[:, 1], common_prod)]
        # Creating R_train and R_test dictionaries
        self.R_train_ui = dict(zip(zip(self.R_train[:, 0], self.R_train[:, 1]), self.R_train[:, 3])) 
        self.R_test_ui = dict(zip(zip(self.R_test[:, 0], self.R_test[:, 1]), self.R_test[:, 3]))
        # Initializing parameters to be estimated
        #self.A = self.createA(self.n_users + 1, self.n_cat + 1)
        self.A = np.random.rand(self.n_users + 1, self.n_users + 1, self.n_cat)
        self.B = np.random.rand(self.n_users + 1, self.n_cat)
        self.C = np.random.rand(self.n_prod + 1)
        self.E = self.R_train_ui

    def createA(self, n, c):
        f = tb.open_file('A.h5', 'w')
        A = f.create_carray(f.root, 'A', tb.Float32Atom(), shape = (n, n, c))
        n_batch = int(math.ceil(n/1000))
        for i in xrange(0, n_batch):
            A[i*1000:(i+1)*1000, i*1000:(i+1)*1000, :] = np.random.uniform(low = 0, high = 1, size = (1000, 1000, c))
        return A

    def calc_cost(self, l):
        cost = sum(self.E[key] * self.E[key] for key in self.E)
        cost += l*(np.vdot(self.B, self.B) + np.vdot(self.C, self.C))
        return cost

    def getNui(self, u, i):
        # Generates Users who are trusted by user u and have rated product i
        # Users who have rated product i
        rat_u = self.R_train[np.where(self.R_train[:, 1] == i), 0]
        # Users trusted by u
        trust_u = self.W[u]
        return np.intersect1d(rat_u, trust_u)

    def calculateRcap(self, u, i):
        # First part of Rcap
        n1 = 0
        d1 = 0
        for k in xrange(self.n_cat):
            if (i,cat_map[k]) in self.PF:
                n1 += (self.mu[k] + self.B[u, k])
                d1 += 1
        Rcap = alpha * (n1/d1 + self.C[i])
        # Second part of Rcap
        n2 = 0
        d2 = 0
        V = self.getNui(u, i)
        for k in xrange(self.n_cat):
            if((i,cat_map[k]) in self.PF):
                for v in V:
                    n2 += self.R_train_ui[v, i] * self.A[v, u, k]
                    d2 += np.sum(self.A[v, u, k])
        if d2 != 0:
            Rcap += (1 - alpha) * (n2/d2)
        return Rcap

    def model(self, alpha = 0.1, l = 0.1, lr_a = 0.1, lr_b = 0.1, lr_c = 0.1, n_it = 1):
        # Optimization Routine
        self.Rcap = np.zeros_like(self.R_train_ui)
        cat_map = {0:7, 1:8, 2:9, 3:10, 4:11, 5:19}
        for it in xrange(n_it):
            print it
            cost = self.calc_cost(l)
            for key in self.R_train_ui:
                u, i = key
                Rcap = calculateRcap(u, i)
                # Defining E, p and q
                self.E[u, i] = self.R_train_ui[u, i] - Rcap
                p = d2
                q = n2
                # Updating C
                grad_c = -alpha * self.E[u, i] + l * self.C[i]
                self.C[i] -= lr_c * grad_c
                # Updating B
                grad_b = -alpha * self.E[u, i] + l * (self.B[u, :]/d1)
                self.B[u, :] -= lr_b * grad_b
                # Updating A
                for k in xrange(self.n_cat):
                    if (i,cat_map[k]) in self.PF:
                        for v in V:
                            if p != 0:
                                grad_a = (alpha - 1) * self.E[u,i] * ((p * self.R_train_ui[v, i] - q)/(p * p))
                                self.A[v, u, k] -= lr_a * grad_a
                                if self.A[v, u, k] < 0:
                                    self.A[v, u, k] = 0
                                elif self.A[v, u, k] > 1:
                                    self.A[v, u, k] = 1
                print test()
    
    def test(self):
        error = 0
        U = 0
        for key in self.R_test_ui:
            U += 1
            u, i = key
            R_Rcap = R[u, i] - calculateRcap[u, i]
            error += R_Rcap * R_Rcap
        return math.sqrt(error/U)
        
if __name__ == "__main__":
    model = baseline_tensor()
    model.model()

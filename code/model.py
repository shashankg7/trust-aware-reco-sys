#! /usr/bin/env python

import numpy as np
from data_handler import data_handler
import tables as tb

class baseline_tensor():

    def __init__(self):
        # Loading matrices from given data
        data = data_handler("../data/rating_with_timestamp", "../data/trust")
        self.R_train, self.R_test, self.W, self.PF, self.mu = data.load_matrices()
        self.n_users, self.n_prod, self.n_cat = data.get_stats()    
        # Initializing parameters to be estimated
        self.A = self.createA(self.n_users + 1, self.n_cat + 1)
        self.B = np.random.rand(self.n_users + 1, self.n_cat + 1)
        self.C = np.random.rand(self.n_prod + 1)
        self.E = R_train

    def createA(self, n, c):
        f = tb.open_file('A.h5', 'w')
        A = f.create(f.root, 'A', tb.Float32Atom(), shape = (n, n, c))
        n_batch = math.ceil(n/1000)
        for i in xrange(0, n_batch):
            A[i*1000:(i+1)*1000, i*1000:(i+1)*1000, :] = np.random.uniform(low = 0, high = 1, size = (1000, 1000, c))
        return A

    def calc_cost(self, l):
        cost = sum(dict[key]*dict[key] for key in self.E)
        cost += l*(np.sum(np.dot(self.B * self.B)) + np.sum(np.dot(self.C + self.C)))
        return cost

    def model(self, alpha = 0.1, l = 0.1, lr_a = 0.1, lr_b = 0.1, lr_c = 0.1, n_it = 100):
        # Optimization Routine
        self.Rcap = np.zeros_like(self.R_train)
        for it in xrange(n_it):
            cost = calc_cost(l)
            

if __name__ == "__main__":
    model = baseline_tensor()

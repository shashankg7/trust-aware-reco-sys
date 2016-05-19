
import numpy as np
import scipy
from data_handler import data_handler
import tables as tb
import math

class baseline_tensor(object):
    '''
    Implementation of baseline method from mTrust, eq 10-13.
    '''
    def createA(self, n, c):
        # Create a randomly initialized A tensor
        f = tb.open_file('A.h5', 'w')
        A = f.create_carray(f.root, 'A', tb.Float32Atom(), shape=(n +1,
                                                                  n +1,
                                                                  c +1))
        n_batch = math.ceil(n/1000)
        for i in xrange(0, n_batch):
            A[i*1000:(i+1)*1000, i*1000:(i+1)*1000,:] = np.random.uniform(low=0,
                                                                          high=1,
                                                                          size=\
                                                                          (1000\
                                                                           ,1000\
                                                                           , c+1))
        return A

    def getNui(self, u, i):
        # Generates user who are trusted by u and have rated prod. i
        # Users who have rated product i
        rat_u = self.R_train[np.where(R[:, 1] == i), 0]
        # Users who are trusted by user u
        trust_u = np.where(self.W[u, :] == 1)
        # handle index
        trust_u += 1
        return np.intersect1d(rat_u, trust_u)


    def __init__(self):
        # load data and initialize paramters
        data = data_handler("../data/rating.mat", "../data/trust.mat")
        self.R_train, self.R_test, self.W, self.PF, self.mu = data.load_matrices()
        self.n_users, self.n_prod, self.n_cat = data.get_stats()
        # initializing model paramters (A, B, c)
        self.B = np.random.rand(n_users + 1, n_cat +1)
        self.C = np.random.rand(n_prod + 1, 1)
        self.A = self.createA(self.n_users, self.n_cat)
        pdb.set_trace()

    def cal_cost(self, E, l):
        return np.sum(np.dot(E, E)) + l * np.sum(np.dot(self.B, self.B) + np.sum(np.dot(self.C, self.C)))

    def model(self, alpha=0.1, l=0.1, lr_a=0.1, lr_b=0.1, lr_c=0.1, n_iter=100):
        # optimization routine
        self.R1 = np.zeros_like(self.R_train)
        E = self.R_train 
        for iter in n_iter:
            cost = self.calc_cost(E, l)
            for u in xrange(1, self.R1.shape[0]+1):
                for i in xrange(1,self.R1.shape[1]+1):
                    # first part of Rcap sum
                    t = np.dot(self.PF[i, :], self.mu + self.B[u,:])
                    d = np.sum(self.PF[i, :])
                    self.R1[u,i] += alpha * (t/d + self.C[i])
                    # second part of rcap sum
                    v = self.getNui(u,i)
                    t1 = np.dot(self.PF[i, :], np.dot(self.A[v, u, :], self.R_train[v, i]))
                    d = np.dot(self.PF[i, :], np.sum(self.A[v, u, :], axis=1))
                    self.R1[u,i] += (1-alpha) * (t1/d)
                    E[u, i] = self.R_train[u,i] - self.R1[u,i]
                    p = d
                    q = t1
                    # Update part for B
                    grad_B = l * (B[u,:]/np.sum(self.PF[i,:])) - alpha * E[u, i]
                    self.B[u, :] -= lr_b * grad_B
                    # Update part for c
                    grad_c = -alpha * E[u, i] + l * self.C[i]
                    self.C[i] -= lr_c * grad_c
                    # Update part for A
                    for v in xrange(1,self.R1.shape[0]+1)
                        for k in self.A.shape[2]:
                            grad_A = (alpha - 1) * E[u, i] * (self.PF[i, k] * R_train[v,i] * p - q * PF[i, k] )/(p * p)
                            if A[v, u , k] - lr_a * grad_A < 0:
                                A[v, u, k] = 0
                            elif A[v, u , k] - lr_a * grad_A > 1:
                                A[v, u, k] = 1
                            else:
                                A[v, u, k] -= lr_a * grad_A


if __name__ == "__main__":
    model = baseline_tensor()

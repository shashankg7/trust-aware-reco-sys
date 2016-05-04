
import numpy as np
import scipy
from data_handler import data_handler

class baseline_tensor(object):
    '''
    Implementation of baseline method from mTrust, eq 10-13.
    '''
    def __init__(self):
        # load data and initialize paramters
        data = data_handler("../data/rating.mat", "../data/trust.mat")
        self.R_train, self.R_test, self.W, self.PF = data.load_matrices()
        self.n_users, self.n_prod, self.n_cat = data.get_stats()
        # initializing model paramters (A, B, c)
        self.A = np.random.uniform(low=-1, high=1, size=(n_users, n_users, n_cat))
        pdb.set_trace()

if __name__ == "__main__":
    model = baseline_tensor()


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
        self.R, self.R1, self.W = data.load_matrices()
        self.n_users = self.W.shape[0]
        self.n_facets = 6





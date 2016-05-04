
from scipy.sparse import lil_matrix
import numpy as np
import pdb
from scipy.io import loadmat
import time

class data_handler(object):
    def __init__(self, rating_path, trust_path):
        self.rating_path = rating_path
        self.trust_path = trust_path

    def load_matrices(self):
        f = open(self.rating_path)
        f1 = open(self.trust_path)
        R = loadmat(f)
        W = loadmat(f1)
        R = R['rating_with_timestamp']
        W = W['trust']
        n_users = max(R[:, 0])
        n_prod = max(R[:, 1])
        n_cat = 6
        cat_id = [7,8,9,10,11,19]
        cat_map = {7:1, 8:2, 9:3, 10:4, 11:5, 19:6}
        # Selecting records with category id falling in 6 pre-defined categories
        # choosen in paper
        R = R[np.in1d(R[:, 2], cat_id)]
        # Sorting based on time-stamp
        R = R[R[:, 5].argsort()]
        #pdb.set_trace()
        pf = np.zeros((n_prod + 1, n_cat + 1))
        for i in xrange(1, n_prod + 1):
            cat = R[R[:, 1] == i, 2]
            cat = map(lambda x:cat_map[x], cat)
            pf[i, cat] = 1
        # Sparse matrix, due to large memory requirement
        # TO-DO : Look for a way to use R matrix only.
        r = lil_matrix((n_users+1, n_prod+1))
        print r.shape
        #pdb.set_trace()
        for i in xrange(1, n_users+1):
            # for each user, product id's he rated
            ids = R[R[:, 0] == i, 1]
            ratings = R[ids, 3]
            r[i, ids] = ratings
        # train-test split based on time-stamp
        train_idx = int(0.7 * r.shape[0])
        #pdb.set_trace()
        R_train = r[:train_idx, :]
        R_test = r[train_idx:, :]
        #pdb.set_trace()
        return R_train, R_test, W, pf

if __name__ == "__main__":
    data_handle = data_handler("../data/rating.mat",
                               "../data/trust.mat")
    t = time.time()
    R_t,R_test, W, PF = data_handle.load_matrices()
    print time.time() - t
   # pdb.set_trace()
    print R_t.shape, W.shape
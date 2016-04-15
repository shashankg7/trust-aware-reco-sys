
from scipy.sparse import lil_matrix
import numpy as np
import pdb


class data_handler(object):
    def __init__(self, rating_path, trust_path):
        self.rating_path = rating_path
        self.trust_path = trust_path

    def load_matrices(self):
        R = lil_matrix((49291, 139739))
        W = lil_matrix((49291, 49291))
        f1 = open(self.rating_path)
        f2 = open(self.trust_path)
        for record in f1:
            user_id, item_id, rating = map(lambda x:int(x), record.split())
            R[user_id, item_id] = rating
        f1.close()
        for record in f2:
            user1, user2, trust = map(lambda x:int(x), record.split())
            W[user1, user2] = trust

        return R, W

if __name__ == "__main__":
    data_handle = data_handler("../data/ratings_data.txt",
                               "../data/trust_data.txt")
    R, W = data_handle.load_matrices()
    pdb.set_trace()
    print R.shape, W.shape

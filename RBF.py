import numpy as np
import GMM
from scipy.linalg import pinv


class RBF:

    def __init__(self, input_dim, centers_num, out_dim, class_nums, iteration):
        self.input_dim = input_dim
        self.centers_num = centers_num
        self.out_dim = out_dim
        self.class_nums = class_nums
        self.iteration = iteration
        self.centers = []
        self.sigma = []
        self.w = None
        self.H = None

    def train(self, X):
        for i in range(self.class_nums):
            X_class = X.loc[X['target'] == i]
            X_class_ = X_class.iloc[:, :X_class.shape[1] - 1]
            X_class_ = np.asarray(X_class_)
            mus, sigmas = GMM.train_GMM(X_class_, self.centers_num, self.iteration)
            self.centers.append(mus)
            self.sigma.append(sigmas)

        temp = X
        X = X.iloc[:, :X.shape[1] - 1]
        self.H = np.zeros((self.centers_num * self.class_nums, X.shape[0]), dtype=float)
        X = np.asarray(X)

        for i in range(self.class_nums):
            for j in range(self.centers_num):
                for k in range(X.shape[0]):
                    self.H[i * self.class_nums + j][k] = GMM.getPdf(X[k], self.centers[i][j], self.sigma[i][j])

        Y = np.zeros((temp.shape[0], self.class_nums))
        a = temp['target']
        a = np.asarray(a)
        for i in range(Y.shape[0]):
            b = a[i]
            Y[i][b] = 1
        self.w = np.dot(pinv(self.H.T), Y)

    def predict(self, X):
        h = np.zeros((self.centers_num * self.class_nums, X.shape[0]), dtype=float)
        X = np.asarray(X)

        for i in range(self.class_nums):
            for j in range(self.centers_num):
                for k in range(X.shape[0]):
                    h[i * self.class_nums + j][k] = GMM.getPdf(X[k], self.centers[i][j], self.sigma[i][j])

        O = np.dot(h.T, self.w)

        return O

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SSL:

    def __init__(self, iteration, class_num):
        self.iter = iteration
        self.class_num = class_num

    def set_G(self, X, ssu, spd):
        G = []

        for i in range(len(spd)):
            subspace = X.iloc[:, ssu[spd[i]]]
            G.append(self.gi(subspace))

        return G

    def gi(self, X):
        X_ = X.iloc[:, :X.shape[1] - 1]

        neigh = NearestNeighbors(n_neighbors=10)
        neigh.fit(X_)

        G = neigh.kneighbors_graph(X_)
        G = G.toarray()
        return G

    def set_diff(self, G):
        D = np.zeros((len(G), len(G)))

        for i in range(len(G)):
            for j in range(i + 1, len(G)):
                D[i][j] = np.linalg.norm(G[i] - G[j])

        D = np.triu(D) + np.triu(D, 1).T

        return D

    def set_reli(self, X, ssu, spd, D):

        flag = [-1, -1]
        reli = []

        for ite in range(self.iter):
            min_ = D[0][0]

            for i in range(D.shape[0]):
                k = 0
                temp = 0
                while k < len(reli):
                    if reli[k] == i:
                        temp = 1
                        break
                    k += 1
                if temp == 1:
                    continue
                for j in range(i + 1, D.shape[1]):
                    k = 0
                    temp = 0
                    while k < len(reli):
                        if reli[k] == j:
                            temp = 1
                            break
                        k += 1
                    if temp == 1:
                        continue
                    if D[i][j] > min_:
                        min_ = D[i][j]
                        flag = [i, j]

            r = [-1, -1]
            for i in range(len(flag)):
                X_ = X.iloc[:, ssu[spd[flag[i]]]]
                temp = X_.iloc[:, :X_.shape[1]-1]
                X_mean = np.mean(temp, axis=0)
                S_K_W = np.zeros((X_.shape[1] - 1, X_.shape[1] - 1))
                S_K_B = 0

                for j in range(self.class_num):
                    X_class = X_.loc[X_['target'] == j]
                    X_class_ = X_class.iloc[:, :X_.shape[1] - 1]
                    X_class_mean = np.mean(X_class_, axis=0)
                    S_K_W += np.dot((X_class_ - X_class_mean).T, (X_class_ - X_class_mean))
                    S_K_B += len(X_class_) * np.dot(X_class_mean - X_mean, X_class_mean - X_mean)

                r[i] = S_K_B / S_K_W.trace()

            if r[0] > r[1]:
                reli.append(flag[1])
            else:
                reli.append(flag[0])

        return reli

import numpy as np
import pandas as pd


class DPSS:

    def __init__(self, attr_num, class_num):
        self.fori = None
        self.attr_num = attr_num
        self.class_num = class_num
        self.threhold_1 = None
        self.threhold_2 = None

    def dpss(self, X):
        self.Fori(X)
        Fd = self.set_skey(X)
        new_X = X.iloc[:, Fd]
        F = self.set_F(new_X)
        ssu = self.set_ssu(F)
        spu = self.set_spd(new_X, ssu)
        return spu

    def Fori(self, X):
        X_ = X.iloc[:, :self.attr_num]
        X_mean = np.mean(X_, axis=0)
        S_K_W = np.zeros((self.attr_num, self.attr_num))
        S_K_B = 0

        for i in range(self.class_num):
            X_class = X.loc[X['target'] == i]
            X_class_ = X_class.iloc[:, :self.attr_num]
            X_class_mean = np.mean(X_class_, axis=0)
            S_K_W += np.dot((X_class_ - X_class_mean).T, (X_class_ - X_class_mean))
            S_K_B += len(X_class_) * np.dot(X_class_mean - X_mean, X_class_mean - X_mean)

        self.fori = S_K_B / S_K_W.trace()
        print(self.fori)
        self.threhold_1 = 0.5 * self.fori
        self.threhold_2 = 1.5 * self.fori

    def set_skey(self, X):
        X_ = X.iloc[:, :self.attr_num]
        S_K_W = np.zeros((self.attr_num, self.attr_num))
        S_K_B = np.zeros(self.attr_num)

        # X_mean是特征均值，大小为[1,特征数]
        X_mean = np.mean(X_, axis=0)

        for i in range(self.class_num):
            X_class = X.loc[X['target'] == i]
            X_class_ = X_class.iloc[:, :self.attr_num]

            # X_class_mean是类内特征均值，大小为[1,特征数]
            X_class_mean = np.mean(X_class_, axis=0)

            S_K_W += np.dot((X_class_ - X_class_mean).T, (X_class_ - X_class_mean))  # 类内散度
            S_K_B += len(X_class_) * (X_class_mean - X_mean) * (X_class_mean - X_mean)  # 类间散度

        Fd = np.array([-1])
        i = 0
        while i < self.attr_num:
            if S_K_B[i] / S_K_W[i][i] > self.threhold_1:
                Fd = np.insert(Fd, -1, i)
            i += 1

        return Fd

    def set_F(self, X):
        F = np.zeros((X.shape[1] - 1, X.shape[1] - 1))
        for i in range(F.shape[1]):
            for j in range(i + 1, F.shape[1]):
                index = np.array([i, j, -1])
                temp = X.iloc[:, index]
                F[i][j] = self.Fij(temp)

        F = np.triu(F) + np.triu(F, 1).T
        print(F)
        return F

    def set_ssu(self, X):
        ssu = []
        for i in range(X.shape[1]):
            temp = np.array([-1])
            for j in range(X.shape[1]):
                if X[i][j] > self.threhold_2:
                    temp = np.insert(temp, -1, j)
            ssu.append(temp)
        return ssu

    def Fij(self, X):
        X_ = X.iloc[:, :2]
        X_mean = np.mean(X_, axis=0)
        S_K_W = np.zeros((2, 2))
        S_K_B = np.zeros(2)

        for i in range(self.class_num):
            X_class = X.loc[X['target'] == i]
            X_class_ = X_class.iloc[:, :2]
            X_class_mean = np.mean(X_class_, axis=0)
            S_K_W += np.dot((X_class_ - X_class_mean).T, (X_class_ - X_class_mean))
            S_K_B += len(X_class_) * (X_class_mean - X_mean) * (X_class_mean - X_mean)

        S_W_ij = 0.0
        S_B_ij = 0.0
        for i in range(2):
            S_W_ij += S_K_W[i][i]
            S_B_ij += S_K_B[i]

        return S_B_ij / S_W_ij

    def set_spd(self, X, index):
        spd = []
        for i in range(len(index)):
            X_ = X.iloc[:, index[i]]
            temp = X_.iloc[:, :X_.shape[1] - 1]
            X_mean = np.mean(temp, axis=0)
            S_K_W = np.zeros((X_.shape[1] - 1, X_.shape[1] - 1))
            S_K_B = 0

            for j in range(self.class_num):
                X_class = X_.loc[X_['target'] == j]
                X_class_ = X_class.iloc[:, :X_.shape[1] - 1]
                X_class_mean = np.mean(X_class_, axis=0)
                S_K_W += np.dot((X_class_ - X_class_mean).T, (X_class_ - X_class_mean))
                S_K_B += len(X_class_) * np.dot(X_class_mean - X_mean, X_class_mean - X_mean)

            if S_K_B / S_K_W.trace() > self.fori:
                spd.append(i)

        return spd

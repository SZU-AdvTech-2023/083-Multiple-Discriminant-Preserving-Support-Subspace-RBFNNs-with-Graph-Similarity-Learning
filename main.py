import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DPSS import DPSS
from SSL import SSL
from RBF import RBF

data = pd.read_csv(".//heart.csv")
a = pd.read_csv(".//123.csv")

train, test = train_test_split(data, test_size=0.3)
# train_X = train.iloc[:, :attr_num]
# train_y = train['target']

X = train
attr_num = X.shape[1] - 1
# test_X = test.iloc[:, :attr_num]
# test_y = test['target']

dpss = DPSS(attr_num=attr_num, class_num=2)

# Fd = dpss.set_skey(train)
# new_train = train.iloc[:, Fd]


dpss.Fori(X)
Fd = dpss.set_skey(X)
print("筛选出来的特征的：")
print(Fd)
new_X = X.iloc[:, Fd]
F = dpss.set_F(new_X)
ssu = dpss.set_ssu(F)
spd = dpss.set_spd(new_X, ssu)
print(ssu)
ssl = SSL(iteration=4, class_num=2)
G = ssl.set_G(new_X, ssu, spd)

diff = ssl.set_diff(G)

reli = ssl.set_reli(X, ssu, spd, diff)

new_spd = np.delete(spd, reli)

rbf = []
O = []

for i in range(len(new_spd)):
    temp = RBF(attr_num, 2, 2, 2, 50)
    temp.train(new_X.iloc[:, ssu[new_spd[i]]])
    rbf.append(temp)

for i in range(len(rbf)):
    temp = test.iloc[:, ssu[new_spd[i]]]
    temp = temp.iloc[:, :temp.shape[1] - 1]
    O.append(rbf[i].predict(temp))

print(O)
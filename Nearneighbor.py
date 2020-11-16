import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat
import os
import sys
import math
from matplotlib import pyplot as plt
import pandas as pd


# In[13]:
def normalize(array):
    max = np.amax(array)
    min = np.amin(array)
    return (array - min) / (max - min)


fp = os.getcwd()
rfp = os.path.join("data", "sync1.mat")
mat = loadmat(rfp)
mat
p_act = np.array(mat['xk'][:, 0:4])
p_ref = np.array(mat['uk'])
u_v = np.array(mat['xk'][:, 6:])

split = int(.75 * len(p_act))

norm_p = normalize(p_act)
norm_u_v = normalize(u_v)
#norm_p, norm_u_v = shuffle(norm_p, norm_u_v)

weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
kn = KNeighborsRegressor()
kn.set_params(n_neighbors=2, weights=weights[0], algorithm=algorithm[0], p=1)
kn.fit(norm_p[:split], norm_u_v[:split])
prediction = kn.predict(norm_p[split:])

plt.plot(norm_u_v[split:, 0])
plt.plot(prediction[:, 0])
# plt.plot(norm_u_v[split:,1])
# plt.plot(prediction[0,1])

plt.grid()
plt.legend(['actual u', 'predicted u', 'actual v', 'predicted v'])
plt.show()
plt.plot(norm_u_v[split:, 1])
plt.plot(prediction[:, 1])
plt.legend(['actual v', 'predicted v'])
plt.show()

# get best parameters: distance,auto, k=2 or 3, p=1
# for weight in weights:
#     print(weight)
#     for alg in algorithm:
#         print(alg)
#         for i in range(1, 16, 1):
#             for j in range(1, 3, 1):
#                 kn.set_params(n_neighbors=i, weights=weight, algorithm=alg, p=j)
#                 kn.fit(norm_p[:split], norm_u_v[:split])
#                 prediction = kn.predict(norm_p[split:])
#                 print(mean_squared_error(norm_u_v[split:], prediction))

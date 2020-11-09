import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat
import os
import sys
import math
from matplotlib import pyplot
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
split=int(.75*len(p_act))


sk_u = linear_model.LinearRegression(normalize=True)
for i in range(100):
    p_act,u_v=shuffle(p_act,u_v)
    sk_u.fit(p_act[:split], u_v[:split])
    prediction=sk_u.predict(p_act[split:])
    print(mean_squared_error(prediction,u_v[split:]))

np.savetxt('u_v.csv',u_v,delimiter=',',newline=';')
np.savetxt('LinearRegression_Predict_u_v.csv',sk_u.predict(p_act),delimiter=',',newline=';')


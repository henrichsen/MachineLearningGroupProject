#!/usr/bin/env python
# coding: utf-8

# In[3]:


from scipy.io import loadmat
import numpy as np
import os
import sys
import math
from matplotlib import pyplot


# In[13]:


fp = os.getcwd()
rfp = os.path.join("data","sync1.mat")
mat = loadmat(rfp)
mat
p_act = mat['xk'][:,0:4]
p_ref = mat['uk']
u_v = mat['xk'][:,6:]


# In[ ]:





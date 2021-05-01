#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from numpy.linalg import eigvalsh
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import fabs


def golden_section(f, a=0, b=100, max_iter=50, eps=1e-3, speaks = True):
    l=a
    r=b
    fi=(1 + 5**(1/2))/2
    count=0
    fprev = f(a)
    x1 = a+fi*(b-a)
    x2 = b+fi*(a-b)
    while(           (count < max_iter and            (fabs(fprev-(f(l) + f(r))/2) > eps or              fabs(r - l) > eps)) or            count == 0):
        fprev = (f(l) + f(r))/2
        if f(x1) < f(x2):
            r = x2
            x2 = x1
            x1 = l + fi*(r-l)
        else:
            l=x1
            x1=x2
            x2=r+fi*(l-r)
        count+=1
        if speaks:
            print(f"{count} f(x1) {f(x1)} f(x2) {f(x2)} x1 {x1} x2 {x2} l {l} r {r}")
            
            
    return (l+r)/2

def gradient(f, x, alpha=1e-8):
    x = np.array(x, dtype = np.float64) 
    deriv = []
    for i in range(len(x)):
        x_plus_a, x_minus_a = np.copy(x), np.copy(x)
        x_plus_a[i] = x_plus_a[i] + alpha
        x_minus_a[i] = x_minus_a[i] - alpha
        deriv.append( (f(x_plus_a) - f(x_minus_a))/(2*alpha) ) 
    return np.array(deriv)



# In[ ]:





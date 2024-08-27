# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from signal_generator import *

def DFT(x):
    N = len(x)
    
    n = np.arange(N)
    k = n.reshape(N,1)
    W = np.exp(-2j*np.pi*k*n/N)
    Xf = np.dot(W,x)
    
    return Xf
    

Vp = 1
Vdc = 0
fs = 200
f = 5
N = fs/f
fase = 0


[t,x] = signal_gen(Vp, Vdc, f, fase, N, fs)
# plt.plot(t,x)

Xf = DFT(x)
Xf_n = np.round(Xf,2)/N

plt.figure(1)
plt.stem(np.arange(len(t))*fs/N,abs(Xf_n))
plt.xlabel("bins")
plt.ylabel("Magnitud")

plt.figure(2)
plt.stem(np.arange(len(t))*fs/N,np.angle(Xf_n)*180/np.pi);
plt.xlabel("bins")
plt.ylabel("Fase")
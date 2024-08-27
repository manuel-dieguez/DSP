#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:06:45 2024

@author: Manuel Dieguez
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

def signal_gen(Vp, Vdc, frec, fase, N, fs, tipo="senoidal",duty=0.5):
    
    ## Primero generamos el vector tiempo
    t = np.arange(0, N/fs, 1/fs)
    
    if tipo == "senoidal":
        x = Vdc + Vp*np.sin(2*np.pi*frec*t + fase)
    elif tipo == "cuadrada":
        x = Vdc + Vp*sc.signal.square(2*np.pi*frec*t + fase, duty)
    else:
        x = 0
        print("Tipo de senal invalido")
        
    return [t,x]


Amp = 2
Vdc = 1
frec = 10
fase = np.pi*90/180
fs = 1000
N = fs/frec  #Para meter un ciclo


[t,x] = signal_gen(Amp, Vdc, frec, fase, N, fs)
plt.figure(1)
plt.plot(t, x)
plt.title("Señal generada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()


[t,x] = signal_gen(Amp, Vdc, frec, 0, N, fs, tipo="cuadrada", duty = 0.7)
plt.figure(2)
plt.plot(t, x)
plt.title("Señal generada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()


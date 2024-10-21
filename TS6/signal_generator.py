#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:24:13 2024

@author: manuel
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
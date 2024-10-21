#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:13:17 2024

@author: manuel
"""

import numpy as np
#import numpy.matlib as matlib
import scipy as sc
import matplotlib.pyplot as plt
from signal_generator import *
from tabulate import tabulate

def estimador_amp(x,N,fs):
    freqs = np.fft.fftfreq(N,1/fs)
    bfreqs = (freqs>=0)
    indice_max = np.argmax(np.abs(x[:,bfreqs]),axis =1)
    x_max = 2*np.abs(x[np.arange(x.shape[0]),indice_max])
    return x_max
        

N_exp = 200

Vp = 2
Vdc = 0
fs = 1000
N = fs      ## IMPORTANTE PARA VER LA DISTRIBUCION DE PROBABILIDAD
f = fs/N*(N/4)
fase = 0

## Ruido
SNR_db = 10
SNR = 10**(SNR_db/10)
Ps = Vp**2/2
Pn = Ps/SNR
fr = np.random.uniform(-fs/(2*N),fs/(2*N),N_exp)
t = np.linspace(0,N/fs,N)
#t_matrix = np.matlib.repmat(t,len(fr),1)
#fr_matrix = np.matlib.repmat(fr,len(t),1)
t_matrix = np.tile(t, (len(fr),1))
fr_matrix = np.tile(fr.reshape(-1, 1), (1, len(t)))

## Nuestro generador de senal va a generar nuestra senal discreta muestreada a 
## fs pero sin cuantizar su ampltiud
x_analogica   = Vdc + Vp*np.sin(2*np.pi*(f+fr_matrix)*t_matrix + fase)
## Ahora agregamos ruido a nuestra senal
n_matrix = np.random.normal(0, np.sqrt(Pn),(len(fr),len(t)))
#n_matrix = np.tile(n,(len(fr),1))
x_matrix = x_analogica + n_matrix

## Genero las ventanas

window_bartlett = np.bartlett(N)
window_hann = sc.signal.windows.hann(N)
window_blackman = np.blackman(N)
window_flat_top = sc.signal.windows.flattop(N)

## Ploteamos las diferentes ventanas
plt.figure(1)
plt.plot(np.arange(N), window_bartlett, "b-",label = "Ventana de Bartlett")
plt.plot(np.arange(N), window_hann    , "r-",label = "Ventana de Hann")
plt.plot(np.arange(N), window_blackman, "g-",label = "Ventana de Blackman")
plt.plot(np.arange(N), window_flat_top, "k-",label = "Ventana de Flat Top")
plt.grid()
plt.legend()
plt.show()

## Ploteamos las FFT de las ventanas
eps = 1e-10  # Pequeño valor para evitar log(0)
window_bartlett_fft = np.abs(np.fft.fft(window_bartlett))
magnitude_bartlett = np.maximum(window_bartlett_fft/np.max(window_bartlett_fft), eps)


freqs = np.fft.fftfreq(N,d=fs)  # Frecuencias normalizadas entre -0.5 y 0.5
freqs_shifted = np.fft.fftshift(freqs)
# plt.figure(2)
# plt.plot(freqs_shifted, np.fft.fftshift(20*np.log10(magnitude_bartlett)), "b-",label = "Ventana de Bartlett")
# plt.xlabel("bins")
# plt.grid()
# plt.legend()
# plt.show()
bins = np.linspace(0,Vp,51)
## Calculamos la fft con la ventana de Rectangular
x_fft_rect = np.fft.fft(x_matrix)/N
# plt.figure(2)
# plt.plot(freqs_shifted, np.fft.fftshift(20*np.log10(np.abs(x_fft_rect[2,:]))), "b-",)
# plt.xlabel("bins")
# plt.grid()
# plt.legend()
# plt.show()


a_est_rect = estimador_amp(x_fft_rect,N,fs)
media_rect = np.mean(a_est_rect)
sesgo_rect = np.mean(a_est_rect) - Vp
var_rect   = np.var(a_est_rect)
rect_hist = np.histogram(a_est_rect)


## Calculamos la fft con la ventana de Bartlett
x_fft_bartlett = np.fft.fft(x_matrix*np.tile(window_bartlett, (len(fr), 1)))/N
# plt.figure(3)
# plt.plot(freqs_shifted, np.fft.fftshift(20*np.log10(np.abs(x_fft_bartlett[2,:]))), "b-")
# plt.xlabel("bins")
# plt.grid()
# plt.legend()
# plt.show()
a_est_bartlett = estimador_amp(x_fft_bartlett,N,fs)
media_bartlett = np.mean(a_est_bartlett)
sesgo_bartlett = np.mean(a_est_bartlett) - Vp
var_bartlett   = np.var(a_est_bartlett)
bartlett_hist = np.histogram(a_est_bartlett)


## Calculamos la fft con la ventana de Hann
x_fft_hann = np.fft.fft(x_matrix*np.tile(window_hann, (len(fr), 1)))/N
# plt.figure(4)
# plt.plot(freqs_shifted, np.fft.fftshift(20*np.log10(np.abs(x_fft_hann[2,:]))), "b-")
# plt.xlabel("bins")
# plt.grid()
# plt.legend()
# plt.show()
a_est_hann = estimador_amp(x_fft_hann,N,fs)
media_hann = np.mean(a_est_hann)
sesgo_hann = np.mean(a_est_hann) - Vp
var_hann   = np.var(a_est_hann)
hann_hist = np.histogram(a_est_hann)

## Calculamos la fft con la ventana de Blackman
x_fft_blackman = np.fft.fft(x_matrix*np.tile(window_blackman, (len(fr), 1)))/N
# plt.figure(5)
# plt.plot(freqs_shifted, np.fft.fftshift(20*np.log10(np.abs(x_fft_blackman[2,:]))), "b-")
# plt.xlabel("bins")
# plt.grid()
# plt.legend()
# plt.show()
a_est_blackman = estimador_amp(x_fft_blackman,N,fs)
media_blackman = np.mean(a_est_blackman)
sesgo_blackman = np.mean(a_est_blackman) - Vp
var_blackman   = np.var(a_est_blackman)
blackman_hist = np.histogram(a_est_blackman)

## Calculamos la fft con la ventana Flat top
x_fft_flattop = np.fft.fft(x_matrix*np.tile(window_flat_top, (len(fr), 1)))/N
# plt.figure(6)
# plt.plot(freqs_shifted, np.fft.fftshift(20*np.log10(np.abs(x_fft_flattop[2,:]))), "b-")
# plt.xlabel("bins")
# plt.grid()
# plt.legend()
# plt.show()
a_est_flattop = estimador_amp(x_fft_flattop,N,fs)
media_flattop = np.mean(a_est_flattop)
sesgo_flattop = np.mean(a_est_flattop) - Vp
var_flattop   = np.var(a_est_flattop)
flattop_hist = np.histogram(a_est_flattop,bins)[0]

## Ploteamos los histogramas
plt.figure(7)
plt.hist(a_est_rect     ,bins,color='red'    ,alpha=0.8, label= f"Rectangular: $\mu = {media_rect:.2f}$ $\sigma = {var_rect:.2e}$")
plt.hist(a_est_blackman ,bins,color='black'  ,alpha=1, label= f"Blackman: $\mu = {media_blackman:.2f}$ $\sigma = {var_blackman:.2e}$")
plt.hist(a_est_hann     ,bins,color='purple' ,alpha=1, label= f"Hann: $\mu = {media_hann:.2f}$ $\sigma = {var_hann:.2e}$")
plt.hist(a_est_bartlett ,bins,color='orange' ,alpha=0.4, label= f"Bartlett: $\mu = {media_bartlett:.2f}$ $\sigma = {var_bartlett:.2e}$")
plt.hist(a_est_flattop  ,bins,color ="blue"  ,alpha=0.6, label= f"Flat Top: $\mu = {media_flattop:.2f}$ $\sigma = {var_flattop:.2e}$")
plt.axvline(Vp, linestyle = "dashed",color= "red", label = f"Valor real del parametro $a_0$")
plt.legend()
plt.xlabel("Valores del estimador")
plt.ylabel("Frecuencia")
plt.grid()
plt.show()


datos_tabla = [
    ["Rectangular", sesgo_rect, var_rect],
    ["Bartlett", sesgo_bartlett, var_bartlett],
    ["Hann", sesgo_hann, var_hann],
    ["Blackman", sesgo_blackman, var_blackman],
    ["Flat Top", sesgo_flattop, var_flattop]
]

encabezados = ["Ventana", "Sesgo", "Varianza"]

tabla = tabulate(datos_tabla, headers = encabezados, tablefmt="fancy_grid")
print(tabla)


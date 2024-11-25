#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:14:23 2024

@author: manuel
"""

import numpy as np
#import numpy.matlib as matlib
import scipy as sc
import matplotlib.pyplot as plt
from signal_generator import *

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


x_fft = np.fft.fft(x_matrix)/N
freqs = np.fft.fftfreq(N,1/fs)
#freqs = np.fft.fftshift(freqs)
bfreqs = (freqs>=0)

## En cada columna tenemos la fft de una realizacion

# plt.figure(1)
# plt.plot(t,x_analogica[4,:])
# plt.xlabel("Frecuencia")
# plt.ylabel("Tiempo")
# plt.grid()
# plt.title("Una de las senoidales")

# plt.figure(1)
# plt.hist(n)
# plt.ylabel("Frecuencia")
# plt.xlabel("Amplitud")
# plt.grid()
# plt.title("Distribucion del ruido")

# Ploteamos una de las fft
plt.figure(2)
plt.plot(freqs[bfreqs],20*np.log10(2*np.abs(x_fft[10,bfreqs])))
plt.axhline(20*np.log10(2*np.abs(x_fft[10,np.argmax(np.abs(x_fft[10,bfreqs]))])), color= 'red', linestyle = 'dashed', 
            label = f'Valor maximo = {20*np.log10(2*np.abs(x_fft[10,np.argmax(np.abs(x_fft[10,bfreqs]))])): 2f}' )
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.grid()
plt.legend()
plt.title("FFT de una de las muestras")


## Obtenemos los indices que maximizan la amplitud
indice_maxamp = np.argmax(np.abs(x_fft[:,bfreqs]),axis =1)
Vp_max = 2*np.abs(x_fft[np.arange(x_fft.shape[0]),indice_maxamp])

## Vamos a trabajar con el estimador de amplitud
#Vp_est = np.mean(Vp_max)
a0 = 2*np.abs(x_fft[:,N//4])
Vp_est = np.mean(a0)  ## // es division entera
Vp_est_var = np.var(a0)
a0_sesgo = Vp_est - Vp

plt.figure(3)
plt.hist(a0, density=False, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(Vp_est,color = 'red', linestyle = 'dashed', label = f'Estimacion = {Vp_est:4f}')
plt.title(f'Distribución de valores en N/4 = {N/4 * fs/N } Hz')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.legend()
plt.show()

## Probamos otro estimador, tomando los valores de amplitud maxima
a1 = 2*np.abs(x_fft[np.arange(x_fft.shape[0]),indice_maxamp])
Vp_est2 = np.mean(a1)
Vp_est2_var = np.var(a1)
a1_sesgo = Vp_est2 - Vp

plt.figure(4)
plt.hist(a1, density=False, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(Vp_est2,color = 'red', linestyle = 'dashed', label = f'Estimacion = {Vp_est2:4f}')
plt.title('Distribución de valores maximos de las realizaciones')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.legend()
plt.show()

## Estimamos la frecuencia
f0 = indice_maxamp*fs/N
f_est = np.mean(f0)
f_est_var = np.var(f0)
f_sesgo = f_est - f

plt.figure(5)
plt.hist(f0, density=False, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(f_est,color = 'red', linestyle = 'dashed', label = f'Estimacion = {f_est:4f}')
plt.title('Distribución del estimador frecuencia')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.legend()
plt.show()




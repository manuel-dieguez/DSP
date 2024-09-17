#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:16:56 2024

@author: manuel
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from signal_generator import *

Vp = 1
Vdc = 0
fs = 1000
N = fs      ## IMPORTANTE PARA VER LA DISTRIBUCION DE PROBABILIDAD
f = fs/N
fase = 0


# Parametros del ADC
bits = 4
Vref = 2*Vp
num_pasos = 2**(bits)
q = 2*Vref/(num_pasos)      ## No le resto 1 a num_pasos por como hago el redondeo

## Ruido
kn = 1
Pn = kn * q**2/2

## Nuestro generador de senal va a generar nuestra senal discreta muestreada a 
## fs pero sin cuantizar su ampltiud
[t,x_analogica]   = signal_gen(Vp, Vdc, f, fase, N, fs)
## Ahora agregamos ruido a nuestra senal
n = np.random.normal(0, np.sqrt(Pn),len(t))
x = x_analogica + n

plt.figure(1)
plt.title("Senal analogica")
plt.plot(t,x)
plt.ylabel("Magnitud [V]")
plt.xlabel("Tiempo [s]")
plt.grid()
plt.show()


x_disc = np.round(x/q)*q 

plt.figure(2)
plt.plot(t,x, "o--", color = 'green'  ,label = 'Senal con ruido')
plt.plot(t,x_analogica, "o--", color = 'red', ls='dotted' ,label = 'Senal sin ruido')
plt.step(t,x_disc,where = 'post' , color = 'blue' ,label = 'Senal cuantizada')
plt.title(f"Cuantizacion del ADC de {bits} bits")
## Para mostrar los valores discretos y sus puntos medios
grid_levels = np.linspace(-Vref, Vref, num_pasos+1) 
plt.yticks(grid_levels)
plt.xlabel("Tiempo[s]")
plt.ylabel("Magnitud[V]")
plt.legend()        # Para que muestre los labels
plt.grid()
plt.show()


# ## Pasamos a realizar el analisis estadistico del error
e = x_disc - x

# Ploteamos el error y verificamos que se encue
plt.figure(3)
plt.plot(t, e, color = 'blue')
plt.axhline(q/2, color = "red", linestyle = '--', label = f"+q/2 ({q/2:.4f})")
plt.axhline(-q/2, color = "red", linestyle = '--', label = f"-q/2 ({-q/2:.4f})")
plt.title("Senal de error en el tiempo")
plt.xlabel("Tiempo[s]")
plt.ylabel("Magnitud[V]")
plt.legend()
plt.grid()
plt.show()

n_fft = np.fft.fft(n)/N
x_analog_fft = np.fft.fft(x_analogica)/N
x_fft = np.fft.fft(x)/N
x_disc_fft = np.fft.fft(x_disc)/N
e_fft = np.fft.fft(e)/N
## Creamos el vector de frecuencias ordenado, uso estas funciones por simplicidad
## y ahorrarme el cheque de si N es par o impar
freqs = np.fft.fftfreq(N,1/fs)
#freqs= np.fft.fftshift(freqs)
bfreqs = (freqs >= 0) #fs/2

n_mean = np.mean(np.abs(n_fft)**2)
e_mean = np.mean(np.abs(e_fft)**2)

plt.figure(4)
## Ploteamos la salida del DAC
plt.plot(freqs[bfreqs], 20*np.log10(2*np.abs(x_disc_fft[bfreqs])), color = 'blue', label = '$s_{Q} (ADC out)$', )


## Ploteamos la senal de entrada sin ruido
plt.plot(freqs[bfreqs], 20*np.log10(2*np.abs(x_analog_fft[bfreqs])), color = 'green', ls= 'dotted', label = 's (analogica)', )

## Ploteamos la senal con ruido
plt.plot(freqs[bfreqs], 20*np.log10(2*np.abs(x_fft[bfreqs])), color = 'red', ls= 'dotted', label = f' $s_{{R}}$ = s + n' )

plt.plot(np.array([freqs[bfreqs][0],freqs[bfreqs][-1]]), 10*np.log10(2*np.array([e_mean,e_mean])), '--c', lw= 4,label = f'$\\overline{{n_Q}}$ = {10*np.log10(2*e_mean):.2f}(piso digital)')
plt.plot(np.array([freqs[bfreqs][0],freqs[bfreqs][-1]]), 10*np.log10(2*np.array([n_mean,n_mean])), '--r', lw= 4,label = f'$\\overline{{n}}$ = {10*np.log10(2*n_mean):.2f}(piso analog.)')

plt.title('Densidad espectral de potencia')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.ylim((1.5*np.min(10* np.log10(2* np.array([n_mean, n_mean]))),10))
plt.grid(True)
plt.legend()
plt.show()

## Histograma de la senal

plt.figure(5)
plt.hist(n, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribución del Ruido')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()


plt.figure(6)
plt.hist(e, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, 1/q, 1/q, 0]), '--r' )
plt.title('Distribución de la Señal')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()
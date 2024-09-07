#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:55:45 2024

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
q = 2*Vref/(num_pasos -1)


## Nuestro generador de senal va a generar nuestra senal discreta muestreada a 
## fs pero sin cuantizar su ampltiud
[t,x_analogica]   = signal_gen(Vp, Vdc, f, fase, N, fs)
## Ahora agregamos ruido a nuestra senal, ver de meterle ruido normal
n = np.random.normal(0, q/np.sqrt(12) ,len(t))
varianza = np.var(n)
media   = np.mean(n)
print(f'La media del ruido es: {media:.6f}')
print(f'La varianza del ruido es: {varianza:.6f}')
x = x_analogica + n

plt.figure(1)
plt.title("Senal analogica")
plt.plot(t,x)
plt.ylabel("Magnitud [V]")
plt.xlabel("Tiempo [s]")
plt.grid()
plt.show()


## Cada nivel de amplitud representa un numero de cuentas, para ello aplicamos 
## una  regla de 3 simples y hacemos un redondeo no sesgado con numpy.round
##      2*Vref  --> 2**bits -1
##      x(t)    --> ? = x/q
## Si bien al hacerlo de esta forma se toman en cuenta 2**bits + 1 pasos, no afecta
## a este analisis y es mas sencillo, aparte de evitar desfasaje en la cuantizacion

x_disc = np.round(x/q)*q 

plt.figure(2)
plt.step(t,x_disc,where = 'post' , color = 'blue' ,label = 'Senal cuantiazada')
plt.plot(t,x, "o--", color = 'grey', alpha=0.4  ,label = 'Senal sin cuantizar')
plt.title("Cuantizacion del ADC")
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

## Histograma de la senal

plt.figure(4)
plt.hist(n, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribución del Ruido')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()


plt.figure(5)
plt.hist(e, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribución de la Señal')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()
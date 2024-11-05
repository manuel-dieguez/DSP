#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:28:54 2024

@author: manuel
"""

## TODO:    Normalizar amplitudes
##          Revisar LaTex en plots  

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from bartlett_periodogram import *
import scipy.io as sio
from scipy.io.wavfile import write

fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')

## Realicemos un periodogrma de la senal

N = len(wav_data)
freqs = np.fft.fftfreq(N,d= 1/fs_audio)

t = np.linspace(0, N/fs_audio, N )
plt.figure(1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Audio en el tiempo")
plt.plot(t,wav_data)
plt.grid()
plt.show()

fft_audio = np.fft.fft(wav_data)
per_audio = np.abs(fft_audio)**2/N


plt.figure(2)
plt.title("Periodograma")
plt.xlabel("Frecuencias")
plt.ylabel(f"Amplitud $frac{1}{N}|X(f)|^{2}$")  ## Revisar LaTex
plt.plot(freqs[:N//2],10*np.log10(per_audio[:N//2]))
plt.grid()
plt.show()

## Procedemos a realizar un Periodograma ventaneado

window_blackman = np.blackman(N)
fft_audio_per_vent = np.fft.fft(wav_data*window_blackman)
per_vent_audio = np.abs(fft_audio_per_vent)**2/N

plt.figure(3)
plt.title("Periodograma Ventaneado")
plt.xlabel("Frecuencias")
plt.ylabel(f"Amplitud $frac{1}{N}|X(f)|^{2}$")  ## Revisar LaTex
plt.plot(freqs[:N//2],10*np.log10(per_vent_audio[:N//2]))
plt.grid()
plt.show()

## Metodo de Bartlett

segment_size = int(N/24)
[freqs_bartlett, fft_bartlett] = bartlett_periodogram(wav_data, fs_audio, segment_size)
N_bartlett = len(freqs_bartlett)

plt.figure(4)
plt.title("Periodograma Bartlett")
plt.xlabel("Frecuencias")
plt.ylabel(f"Amplitud")  ## Revisar LaTex
plt.plot(freqs_bartlett[:N_bartlett//2],10*np.log10(fft_bartlett[:N_bartlett//2]))
plt.grid()
plt.show()


w=24
[freqs_welch, fft_welch] = sc.signal.welch(wav_data, fs_audio, window = "hann", nperseg = int(N/w))

plt.figure(5)
plt.title("Welch")
plt.xlabel("Frecuencias")
plt.ylabel(f"Amplitud")  ## Revisar LaTex
plt.plot(freqs_welch,10*np.log10(fft_welch))
plt.grid()
plt.show()


## Tomamos el BW a partir de Welch

BW_total_pow = 0.7
sumatoria = 0

per_norm = fft_welch/np.sum(np.abs(fft_welch))

for i in range(0,len(fft_welch)):
    sumatoria += np.abs(per_norm[i])
    index_bw = i
    if sumatoria >= BW_total_pow:
        break

BW = freqs_welch[i]
fft_filter = np.zeros_like(fft_audio)
fft_filter[:index_bw*w] = fft_audio[:w*index_bw] ## w para reajustar lo que se achico el indice en welch

wav_data_restored = N*np.fft.ifft(fft_filter)       ## No es el mejor porque transformo el brickwall

plt.figure(6)
plt.title("Data Restored")
plt.xlabel("Tiempo")
plt.ylabel(f"Amplitud")  ## Revisar LaTex
plt.plot(wav_data_restored)
#plt.plot(freqs_welch,10*np.log10(fft_filter), color="red")
plt.grid()
plt.show()


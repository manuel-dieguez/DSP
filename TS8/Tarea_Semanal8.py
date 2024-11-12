#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:14:04 2024

@author: manuel
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
#from bartlett_periodogram import *
import scipy.io as sio
from scipy.io.wavfile import write
%matplotlib qt


def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz
N = 40000 
w = 10

# para listar las variables que hay en el archivo

mat_struct = sio.loadmat('./ECG_TP4.mat')

#ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
ecg_one_lead = mat_struct['ecg_lead'].flatten()
qrs_detections = mat_struct['qrs_detections'].flatten()
ecg_one_lead_cut = ecg_one_lead[0:N]

plt.figure(1)
plt.title("Electrocardiograma")
plt.ylabel("Amplitud")
plt.title("Tiempo [s]")
plt.plot(np.arange(0,N/fs_ecg,1/fs_ecg),ecg_one_lead_cut)
plt.grid()

## Peridograma simple de la senal completa
freqs_full = np.fft.fftfreq(len(ecg_one_lead), d=1/fs_ecg)
fft_ecg_full = np.fft.fft(ecg_one_lead)
Pxx_ecg_full = np.abs(fft_ecg_full)**2/N

plt.figure(2)
plt.title("Periodograma completa")
plt.xlabel("Frecuencias")
plt.ylabel(f"Amplitud")  ## Revisar LaTex
plt.plot(freqs_full,10*np.log10(Pxx_ecg_full))
plt.grid()
plt.show()


## Peridograma simple de la senal
freqs = np.fft.fftfreq(N, d=1/fs_ecg)
fft_ecg = np.fft.fft(ecg_one_lead_cut)
Pxx_ecg = np.abs(fft_ecg)**2/N

plt.figure(3)
plt.title("Periodograma senal recortada")
plt.xlabel("Frecuencias")
plt.ylabel(f"Amplitud")  ## Revisar LaTex
plt.plot(freqs,10*np.log10(Pxx_ecg))
plt.grid()
plt.show()


# %%
## Hacemos la estimacion de Welch para los primeros 10 segundos del ECG
[freqs_welch, fft_welch] = sc.signal.welch(ecg_one_lead_cut, fs_ecg, window = "hann", nperseg = int(N/w), detrend ="linear") ## Detrend saca la pendiente

plt.figure(4)
plt.title("Welch")
plt.xlabel("Frecuencias")
plt.ylabel(f"Amplitud")  ## Revisar LaTex
plt.plot(freqs_welch,10*np.log10(fft_welch))
plt.grid()
plt.show()

# %%
## Separamos cada latido y realizamos la PSD
###  COMPARAR CON Y SIN DETREND

heartbeats = np.zeros([len(qrs_detections), 600])
heartbeats_ecg = np.zeros([len(qrs_detections), 600])
for i in range(len(qrs_detections)):
    heartbeats[i,:] = ecg_one_lead[qrs_detections[i]-250:qrs_detections[i]+350]
    heartbeats_ecg[i, :] = sc.signal.detrend(heartbeats[i,:])
    
fft_matrix = np.fft.fft(heartbeats_ecg)
Pxx_matrix = np.abs(fft_matrix)**2/600
for i in range(Pxx_matrix.shape[0]):
    Pxx_matrix[i, :] /= np.max(Pxx_matrix[i, :])
    Pxx_matrix[i,:] = np.fft.fftshift(Pxx_matrix[i,:])

Pxx_prom = np.mean(Pxx_matrix, axis = 0)
freqs_prom = np.fft.fftfreq(600, d = 1/fs_ecg)
freqs_prom = np.fft.fftshift(freqs_prom)

plt.figure(5)
plt.title("Periodograma de latidos")
plt.xlabel("Frecuencias")
plt.ylabel(f"Amplitud")  ## Revisar LaTex
plt.ylim([-100,5])
for i in range(heartbeats_ecg.shape[0]):
    plt.plot(freqs_prom[300:],10*np.log10(Pxx_matrix[i,300:]))
plt.plot(freqs_prom[300:],10*np.log10(Pxx_prom[300:]), "--r",linewidth = 3, label = "Promedio de latidos")
plt.grid()
plt.legend()
plt.show()

# %%
## Tomamos el BW a partir de Welch

BW_total_pow = 0.6
sumatoria = 0

per_norm = fft_welch/np.sum(np.abs(fft_welch))

for i in range(0,len(fft_welch)):
    sumatoria += np.abs(per_norm[i])
    index_bw = i
    if sumatoria >= BW_total_pow:
        break

BW = freqs_welch[i]
fft_filter = np.zeros_like(fft_ecg)
fft_filter[:index_bw*w] = fft_ecg[:w*index_bw] ## w para reajustar lo que se achico el indice en welch

ecg_restored = np.fft.ifft(fft_filter)       ## No es el mejor porque transformo el brickwall

plt.figure(6)
plt.title("Data Restored")
plt.xlabel("Tiempo")
plt.ylabel(f"Amplitud")  ## Revisar LaTex
plt.plot(ecg_restored)
plt.grid()
plt.show()

# %%
# Disenamos el filtro

ripple = 0.5
att = 40

fci = 0.5
fcs = 30
fsi = 0.1
fss = 45

frecs = np.array([0, fsi, fci, fsi, fss, fs_ecg/2])/(fs_ecg/2)
gains = np.array([-att, -att, ripple, ripple, -att, -att])
gains = 10**(gains/20)

filtro_sos_iir = sc.signal.iirdesign([fci,fcs],[fsi,fss], ripple, att, ftype = 'cheby1', output = 'sos', fs = fs_ecg)
freqs_filter,iir_filter = sc.signal.sosfreqz(filtro_sos_iir, worN = 10000)
                                   
plt.figure(7)
plt.plot(freqs_filter/np.pi*(fs_ecg/2),10*np.log10(np.abs(iir_filter)))
#plt.plot(freqs_filter/np.pi*(fs_ecg/2),np.angle(iir_filter))
plt.show()

filtro_sos_iir2 = sc.signal.iirdesign([fci,fcs],[fsi,fss], ripple, att, ftype = 'butter', output = 'sos', fs = fs_ecg)
freqs_filter2,iir_filter2 = sc.signal.sosfreqz(filtro_sos_iir2, worN = 10000)
                                   
plt.figure(8)
plt.plot(freqs_filter2/np.pi*(fs_ecg/2),10*np.log10(np.abs(iir_filter2)))
plt.show()


ecg_filtrado_cheby = sc.signal.sosfilt(filtro_sos_iir, ecg_one_lead_cut)

plt.figure(9)
plt.plot(np.arange(N),ecg_filtrado_cheby)
plt.plot(np.arange(N),ecg_one_lead_cut)
plt.show()

ecg_filtrado_cheby_bidireccional = sc.signal.sosfiltfilt(filtro_sos_iir,ecg_one_lead_cut)
ecg_filtrado_cheby_bidireccional_full = sc.signal.sosfiltfilt(filtro_sos_iir,ecg_one_lead)
plt.figure(10)
plt.plot(np.arange(N),ecg_filtrado_cheby_bidireccional)
plt.show()


plt.figure(11)
plt.plot(np.arange(len(ecg_one_lead))[0:3*N],ecg_filtrado_cheby_bidireccional_full[0:3*N])
plt.show()

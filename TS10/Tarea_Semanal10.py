#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:14:04 2024

@author: manuel
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import write

# %%

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz
N = 40000 
t = np.arange(0, N/fs_ecg, 1/fs_ecg)
w = 20

# para listar las variables que hay en el archivo

mat_struct = sio.loadmat('./ECG_TP4.mat')

#ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
ecg_one_lead = mat_struct['ecg_lead'].flatten()
qrs_detections = mat_struct['qrs_detections'].flatten()
ecg_one_lead_cut = ecg_one_lead[0:N]

t_full = np.arange(0,len(ecg_one_lead), 1/fs_ecg)
#%%
plt.figure(1)
plt.title("Electrocardiograma")
plt.ylabel("Amplitud")
plt.title("Tiempo [s]")
#plt.plot(np.arange(0,N/fs_ecg,1/fs_ecg),ecg_one_lead_cut)
plt.plot(np.arange(0,len(ecg_one_lead),1/fs_ecg), ecg_one_lead)
plt.grid()

# ## Peridograma simple de la senal completa
# freqs_full = np.fft.fftfreq(len(ecg_one_lead), d=1/fs_ecg)
# fft_ecg_full = np.fft.fft(ecg_one_lead)
# Pxx_ecg_full = np.abs(fft_ecg_full)**2/N

# plt.figure(2)
# plt.title("Periodograma completa")
# plt.xlabel("Frecuencias")
# plt.ylabel(f"Amplitud")  ## Revisar LaTex
# plt.plot(freqs_full,10*np.log10(Pxx_ecg_full))
# plt.grid()
# plt.show()


# ## Peridograma simple de la senal
# freqs = np.fft.fftfreq(N, d=1/fs_ecg)
# fft_ecg = np.fft.fft(ecg_one_lead_cut)
# Pxx_ecg = np.abs(fft_ecg)**2/N

# plt.figure(3)
# plt.title("Periodograma senal recortada")
# plt.xlabel("Frecuencias")
# plt.ylabel(f"Amplitud")  ## Revisar LaTex
# plt.plot(freqs,10*np.log10(Pxx_ecg))
# plt.grid()
# plt.show()


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

BW_total_pow = 0.95
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
# Disenamos el filtro para la senal recortada

ripple = 0.5
att = 40

fci = 0.5
fcs = 30
fsi = 0.1
fss = 45

zoom_region = np.arange(4000,9000)

frecs = np.array([0, fsi, fci, fsi, fss, fs_ecg/2])/(fs_ecg/2)
gains = np.array([-att, -att, ripple, ripple, -att, -att])
gains = 10**(gains/20)


## Vamos a implementar dos filtrados IIR

sos_iir_cheby = sc.signal.iirdesign([fci,fcs],[fsi,fss], ripple, att, ftype = 'cheby1', output = 'sos', fs = fs_ecg)
freqs_iir_cheby, iir_cheby_response = sc.signal.sosfreqz(sos_iir_cheby, worN = N)
                                   
plt.figure(7)
plt.title("Respuesta en frecuencia IIR Cheby")
plt.xlabel("Frecuencias[Hz]")
plt.ylabel("Amplitud[dB]")
plt.plot(freqs_iir_cheby/np.pi*(fs_ecg/2),20*np.log10(np.abs(iir_cheby_response)))
plt.grid()
plt.show()

plt.figure(8)
plt.title("Respuesta en frecuencia IIR Cheby")
plt.xlabel("Frecuencias[Hz]")
plt.ylabel("Angulo [deg]")
plt.plot(freqs_iir_cheby/np.pi*(fs_ecg/2),np.angle((iir_cheby_response))*180/np.pi)
plt.grid()
plt.show()

# Senal filtrada
ecg_iir_cheby = sc.signal.sosfilt(sos_iir_cheby, ecg_one_lead)

plt.figure(9)
plt.title("Senal filtrada con IIR Cheby")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.plot(t_full[zoom_region],ecg_one_lead[zoom_region], label = "Senal sin filtrar")
plt.plot(t_full[zoom_region],ecg_iir_cheby[zoom_region], label = "Senal filtrada", color="orange")
plt.legend()
plt.grid()
plt.show()

## Probamos un butter bidireccional

sos_iir_butter = sc.signal.iirdesign([fci,fcs],[fsi,fss], ripple/2, att/2, ftype = 'cheby1', output = 'sos', fs = fs_ecg)
freqs_iir_butter, iir_butter_response = sc.signal.sosfreqz(sos_iir_butter, worN = N)

## Genero una respuesta al impulso para filtrar bidireccionalmente

h_butter_bidir = np.multiply(iir_butter_response, np.conj(iir_butter_response))

plt.figure(10)
plt.title("Respuesta en frecuencia IIR Butter Bidireccional")
plt.xlabel("Frecuencias[Hz]")
plt.ylabel("Amplitud[dB]")
plt.plot(freqs_iir_butter/np.pi*(fs_ecg/2),20*np.log10(np.abs(h_butter_bidir)))
plt.grid()
plt.show()

plt.figure(11)
plt.title("Respuesta en frecuencia IIR Butter Bidireccional")
plt.xlabel("Frecuencias[Hz]")
plt.ylabel("Angulo [deg]")
plt.plot(freqs_iir_butter/np.pi*(fs_ecg/2),np.angle((h_butter_bidir))*180/np.pi)
plt.grid()
plt.show()

ecg_filt_butter_bid = sc.signal.sosfiltfilt(sos_iir_butter, ecg_one_lead_cut)

plt.figure(12)
plt.title("Senal filtrada con Butter bidireccional")
plt.xlabel("Tiempo[s]")
plt.ylabel("Amplitud")
plt.plot(t, ecg_one_lead_cut, label = "Senal sin filtrar")
plt.plot(t, ecg_filt_butter_bid, label = "Senal filtrada", color="orange")
plt.legend()
plt.grid()
plt.show()

# %%
## Filtrado FIR

## Probamos metodo de cuad minimos

cant_coeff = 251
demora = int((cant_coeff-1)/2)
att_veces = 10**(-att/20)
ripple_veces = 10**(-ripple/20)

bands = np.array([0, fsi, fci, fcs, fss, fs_ecg/2])
desired = (att_veces, att_veces, 1, 1, att_veces, att_veces)

coeff_firls = sc.signal.firls(cant_coeff, bands, desired, fs = fs_ecg)

freqs_fir_lse, fir_lse = sc.signal.freqz(coeff_firls, a=1, worN = 2**20, fs = fs_ecg)

fir_lse_db = 20*np.log10(np.abs(fir_lse)/np.max(np.abs(fir_lse)))
 
plt.figure(13)
plt.title("Respuesta en frecuencia FIR Minimun Square method")
plt.xlabel("Frecuencias[Hz]")
plt.ylabel("Amplitud [dB]")
plt.plot(freqs_fir_lse, 20*np.log10(np.abs(fir_lse)) ) 
plt.grid()
plt.show()

plt.figure(13)
plt.title("Respuesta en frecuencia FIR Minimun Square method")
plt.xlabel("Frecuencias[Hz]")
plt.ylabel("Amplitud [dB]")
plt.plot(freqs_fir_lse, np.angle(fir_lse)*180/np.pi ) 
plt.grid()
plt.show()

ecg_fir_lse = sc.signal.lfilter(coeff_firls,  1, ecg_one_lead_cut)
plt.figure(15)
plt.title("ECG Filtrado con FIR LSE")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.plot(t[:-demora], ecg_one_lead_cut[:-demora], label = "Senal sin filtrar")
plt.plot(t[:-demora], ecg_fir_lse[demora:], color = "orange", label = "Senal filtrada") 
plt.grid()
plt.legend()
plt.show()

## Probamos un filtrado por ventanas

cant_coeff_win = 1501
demora_win = int((cant_coeff_win-1)/2)

coeff_firwin = sc.signal.firwin2(cant_coeff_win, bands/(fs_ecg/2), desired, nfreqs=2**16 , window = "hamming")
freqs_fir_win, fir_win = sc.signal.freqz(coeff_firwin, a=1, worN = 2**20, fs = fs_ecg)

plt.figure(16)
plt.title("Respuesta en frecuencia FIR Hamming Window")
plt.xlabel("Frecuencias[Hz]")
plt.ylabel("Amplitud [dB]")
plt.plot(freqs_fir_win, 20*np.log10(np.abs(fir_win)) ) 
plt.grid()
plt.show()

plt.figure(17)
plt.title("Respuesta en frecuencia FIR Hamming Window")
plt.xlabel("Frecuencias[Hz]")
plt.ylabel("Angulo [deg]")
plt.plot(freqs_fir_win, np.angle(fir_lse)*180/np.pi ) 
plt.grid()
plt.show()

ecg_fir_win = sc.signal.lfilter(coeff_firwin,  1, ecg_one_lead_cut)

plt.figure(18)
plt.title("ECG Filtrado con FIR Hamming")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.plot(t[:-demora_win], ecg_one_lead_cut[:-demora_win], label = "Senal sin filtrar")
plt.plot(t[:-demora_win], ecg_fir_win[demora_win:], color = "orange", label = "Senal filtrada") 
plt.grid()
plt.legend()
plt.show()

# %%
## Filtrado no lineal

## Primer metodo: Filtro de medianas

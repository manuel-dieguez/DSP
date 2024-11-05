#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:20:35 2024

@author: manuel
"""

import numpy as np
import matplotlib.pyplot as plt

def bartlett_periodogram(x, fs, segment_size):
    """
    Calcula el periodograma de Bartlett para una señal x.

    Parámetros:
    - x: Señal de entrada (1D)
    - fs: Frecuencia de muestreo
    - segment_size: Tamaño de cada segmento
    
    Retorna:
    - f: Vector de frecuencias
    - Pxx: PSD estimada
    """
    # Número de segmentos
    num_segments = len(x) // segment_size

    # Inicializamos el acumulador de PSD
    Pxx = np.zeros(segment_size)

    # Calculamos el periodograma para cada segmento
    for i in range(num_segments):
        segment = x[i * segment_size:(i + 1) * segment_size]
        X = np.fft.fft(segment)  # FFT del segmento
        Pxx_segment = (np.abs(X[:segment_size]) ** 2) / segment_size  # PSD del segmento
        Pxx += Pxx_segment  # Acumulamos la PSD

    # Promediamos la PSD sobre los segmentos
    Pxx /= num_segments

    # Frecuencias correspondientes
    f = np.fft.fftfreq(segment_size, d=1/fs)[:segment_size]

    return f, Pxx
# %%
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from itertools import accumulate
%matplotlib Qt

# Para listar las variables que hay en el archivo
sio.whosmat("ECG_TP4.mat")
mat_struct = sio.loadmat("ECG_TP4.mat")

ecg_one_lead = mat_struct["ecg_lead"]
ecg_one_lead = ecg_one_lead.flatten()
# %%
# Sampling parameters
fs = 1000  # Hz
nyq_frec = fs / 2

N = 10000
w = 5
# %%
ecg_one_lead_20 = ecg_one_lead[:N]
df_p = (fs/N)
df_w = df_p * w

# 1. Periodograma
signal_spectrum = (1/N)*np.fft.fft(ecg_one_lead_20)
Pxx_period_full = abs(signal_spectrum[:int(N/2)])**2
Pxx_period = Pxx_period_full[:int(N/2)]
f_period = np.linspace(0, (N - 1) * df_p, N)

# 2. Welch
f_welch, Pxx_welch = sig.welch(ecg_one_lead_20, fs=fs, nperseg=(N/w))

# %%
# Plotting the results to compare them in the same graph
plt.figure(figsize=(12, 6))

# Plot both Periodogram and Welch's method in the same graph
plt.plot(f_period[:int(N/2)], Pxx_period, label="Periodogram")
plt.plot(f_welch,  Pxx_welch, label="Welch", color="orange")

# Set titles and labels
plt.title("PSD Estimate Comparison: Periodogram vs Welch")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency [dB/Hz]")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
# %%

# %%
norm_p = (Pxx_period.sum())
Pxx_period_normalized = Pxx_period / norm_p

norm_w = (Pxx_welch.sum())
Pxx_welch_normalized = Pxx_welch / norm_w

# %%
# Plotting the results to compare them in the same graph
plt.figure(figsize=(12, 6))

# Plot both Periodogram and Welch's method in the same graph
plt.plot(f_period[:int(N/2)], Pxx_period_normalized, label="Periodogram Normalized")
plt.plot(f_welch,  Pxx_welch_normalized, label="Welch Normalized", color="orange")

# Set titles and labels
plt.title("PSD Estimate Comparison: Periodogram vs Welch")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency ")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
# %%
Pxx_period_normalized.sum()
# %%
Pxx_welch_normalized.sum()

# %%
Pxx_period_normalized_db = 10 * np.log(Pxx_period_normalized)
Pxx_welch_normalized_db = 10 * np.log(Pxx_welch_normalized)
# %%
# Plotting the results to compare them in the same graph
plt.figure(figsize=(12, 6))

# Plot both Periodogram and Welch's method in the same graph
plt.plot(f_period[:int(N/2)], Pxx_period_normalized_db, label="Periodogram Normalized")
plt.plot(f_welch,  Pxx_welch_normalized_db, label="Welch Normalized", color="orange")

# Set titles and labels
plt.title("PSD Estimate Comparison: Periodogram vs Welch")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency [dB/Hz]")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
# %%
p_bw = 0.5

p_cumulative = np.array(list(accumulate(Pxx_period_normalized)))
p_index = np.argmax(p_cumulative >= p_bw) 

w_bw = 0.5

w_cumulative = np.array(list(accumulate(Pxx_welch_normalized)))
w_index = np.argmax(w_cumulative >= w_bw) 
# %%
# Plotting the results to compare them in the same graph
plt.figure(figsize=(12, 6))

# Plot both Periodogram and Welch's method in the same graph
plt.plot(f_period[:p_index], Pxx_period_normalized[:p_index], label="Periodogram filtered")

# Set titles and labels
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency [dB/Hz]")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
# %%
# Plotting the results to compare them in the same graph
plt.figure(figsize=(12, 6))

# Plot both Periodogram and Welch's method in the same graph
plt.plot(f_welch[:w_index], Pxx_welch_normalized[:w_index], label="Periodogram filtered")

# Set titles and labels
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency [dB/Hz]")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
# %%

p_filter = np.zeros(len(signal_spectrum))
p_filter[:p_index] = 1

filtered_ecg_spectrum = signal_spectrum * p_filter
# %%
# Plotting the results to compare them in the same graph
plt.figure(figsize=(12, 6))

# Plot both Periodogram and Welch's method in the same graph
plt.plot(f_period, filtered_ecg_spectrum, label="filtered signal")
#plt.plot(ecg_one_lead_20, label=" signal")


# Set titles and labels
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency [dB/Hz]")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
# %%
filtered_signal = N * np.fft.ifft(filtered_ecg_spectrum)
# %%
# Plotting the results to compare them in the same graph
plt.figure(figsize=(12, 6))

# Plot both Periodogram and Welch's method in the same graph
plt.plot(filtered_signal, label="filtered signal")
plt.plot(ecg_one_lead_20, label=" signal")


# Set titles and labels
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency [dB/Hz]")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# %%

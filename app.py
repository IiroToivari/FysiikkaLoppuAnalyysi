import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from pathlib import Path
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    b, a = butter(order, cutoff/nyq, btype="low")
    return filtfilt(b, a, data)

st.title("Lopputyön analysointi")

ACC_URL = "https://raw.githubusercontent.com/IiroToivari/FysiikkaLoppuAnalyysi/main/data/Linear%20Acceleration.csv"
GPS_URL = "https://raw.githubusercontent.com/IiroToivari/FysiikkaLoppuAnalyysi/main/data/Location.csv"

acc_df = pd.read_csv(ACC_URL)
gps_df = pd.read_csv(GPS_URL)


time = acc_df["Time (s)"].to_numpy(dtype=float)
data = acc_df["Linear Acceleration y (m/s^2)"].to_numpy(dtype=float)

T_tot = time.max() - time.min()
N = len(time)
fs = N / T_tot
nyq = fs / 2
order = 3
cutoff = 1/0.4

data_filt = butter_lowpass_filter(data, cutoff, fs, nyq, order)

jaksot = 0.0
for i in range(N - 1):
    if data_filt[i] * data_filt[i+1] < 0:
        jaksot += 0.5

steps_filt = jaksot

fourier = np.fft.fft(data_filt, N)
psd = fourier * np.conj(fourier) / N

dt = np.max(time) / N
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, int(N/2))

f_max = float(freq[L][np.argmax(psd[L].real)])
steps_fourier = float(f_max * T_tot)

gps_df = gps_df[gps_df["Horizontal Accuracy (m)"] < 6].reset_index(drop=True)

t_gps = gps_df["Time (s)"].to_numpy(dtype=float)
v = gps_df["Velocity (m/s)"].to_numpy(dtype=float)

mean_speed = float(np.nanmean(v))

v = np.nan_to_num(v, nan=0.0)
distance_m = float(np.trapz(v, t_gps))
distance_km = distance_m / 1000.0

step_length_cm = (distance_m / steps_filt) * 100.0

st.write("Askelmäärä laskettuna suodatuksen avulla:", round(steps_filt, 1), "askelta")
st.write("Askelmäärä laskettuna Fourier-analyysin avulla:", round(steps_fourier, 1), "askelta")
st.write("Keskinopeus:", round(mean_speed, 2), "m/s")
st.write("Kokonaismatka:", round(distance_km, 2), "km")
st.write("Askelpituus on", int(round(step_length_cm)), "cm")

st.subheader("Suodatettu kiihtyvyysdata kokonaisuudessaan")
fig1 = plt.figure(figsize=(18, 5))
plt.plot(time, data_filt)
plt.xlabel("Aika [s]")
plt.ylabel("Suodatettu kiihtyvyys y (m/s^2)")
st.pyplot(fig1)

st.subheader("Suodatettu kiihtyvyysdata välillä 210–250 s")
mask = (time >= 210) & (time <= 250)
fig1b = plt.figure(figsize=(18, 5))
plt.plot(time[mask], data_filt[mask])
plt.xlabel("Aika [s]")
plt.ylabel("Suodatettu kiihtyvyys y (m/s^2)")
st.pyplot(fig1b)

st.subheader("Tehospektri")
fig2 = plt.figure(figsize=(15, 6))
plt.plot(freq[L], psd[L].real)
plt.xlabel("Taajuus [Hz]")
plt.ylabel("Teho")
plt.axis([0, 10, 0, float(psd[L].real.max()) * 1.05])
st.pyplot(fig2)

st.subheader("Karttakuva")
start_lat = float(gps_df["Latitude (°)"].mean())
start_long = float(gps_df["Longitude (°)"].mean())
my_map = folium.Map(location=[start_lat, start_long], zoom_start=14)
folium.PolyLine(gps_df[["Latitude (°)", "Longitude (°)"]], color="red", weight=3.5, opacity=1).add_to(my_map)
st_folium(my_map, width=900, height=650)

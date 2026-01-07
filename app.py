import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Ballistic Expert Elite v9.0", layout="wide")

# --- БАЗА ДАНИХ (Розширена) ---
AMMO_DB = {
    "Високоточні": {
        "6.5 Creedmoor ELD-M": {"v0": 825, "bc": 0.313, "model": "G7", "weight_gr": 140},
        ".308 Win SMK 175": {"v0": 790, "bc": 0.243, "model": "G7", "weight_gr": 175},
        ".338 Lapua Scenar": {"v0": 900, "bc": 0.322, "model": "G7", "weight_gr": 250},
    },
    "Армійські": {
        "5.45x39 7N6": {"v0": 880, "bc": 0.330, "model": "G1", "weight_gr": 53},
        "7.62x54R LPS": {"v0": 820, "bc": 0.420, "model": "G1", "weight_gr": 148},
        "5.56x45 M855": {"v0": 915, "bc": 0.304, "model": "G1", "weight_gr": 62},
    }
}

# --- МАТЕМАТИКА ---
def run_simulation(p):
    v0_corr = p['v0'] + (p['temp'] - 15) * p['t_coeff']
    rho = (p['pressure'] * 100) / (287.05 * (p['temp'] + 273.15))
    k_drag = 0.5 * rho * (1/p['bc']) * 0.00052
    if p['model'] == "G7": k_drag *= 0.91

    results = []
    g = 9.80665
    for d in range(0, p['max_dist'] + 1, 10):
        t = d / (v0_corr * math.exp(-k_drag * d / 2)) if d > 0 else 0
        drop = 0.5 * g * (t**2)
        t_zero = p['zero_dist'] / (v0_corr * math.exp(-k_drag * p['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        y_m = -(drop - (drop_zero + p['sh']/100) * (d / p['zero_dist']) + p['sh']/100)
        
        wind_rad = math.radians(p['w_dir'] * 30)
        wind_drift = (p['w_speed'] * math.sin(wind_rad)) * (t - (d/v0_corr)) if d > 0 else 0
        
        results.append({"Range": d, "Drop_cm": y_m * 100, "Wind_cm": wind_drift * 100, "V": v0_corr * math.exp(-k_drag * d)})
    return pd.DataFrame(results)

# --- SIDEBAR ---
st.sidebar.title("🛠️ Ballistic Core")
tab1, tab2 = st.sidebar.tabs(["Набій", "Умови"])

with tab1:
    mode = st.radio("Режим:", ["База", "Custom"])
    if mode == "База":
        cat = st.selectbox("Категорія", list(AMMO_DB.keys()))
        ammo = st.selectbox("Набій", list(AMMO_DB[cat].keys()))
        b = AMMO_DB[cat][ammo]
        v0, bc, model, weight = b['v0'], b['bc'], b['model'], b['weight_gr']
    else:
        v0 = st.number_input("V0", 820); bc = st.number_input("BC", 0.450); model = "G1"; weight = 150
    sh = st.number_input("Висота прицілу (см)", 5.0)
    zero_d = st.number_input("Пристрілка (м)", 100)

with tab2:
    temp = st.slider("Температура (°C)", -20, 45, 15)
    press = st.number_input("Тиск (hPa)", 1013)
    w_speed = st.slider("Вітер (м/с)", 0, 15, 4)
    w_dir = st.slider("Напрямок (год)", 1, 12, 3)
    max_d = st.slider("Макс. дистанція (м)", 100, 1500, 800)

# Розрахунок основний
p = {'v0': v0, 'bc': bc, 'model': model, 'weight_gr': weight, 'temp': temp, 
     'pressure': press, 'w_speed': w_speed, 'w_dir': w_dir, 'zero_dist': zero_d, 
     'max_dist': max_d, 'sh': sh, 't_coeff': 0.2}

df = run_simulation(p)

# Розрахунок помилки вітру (+1 м/с)
p_error = p.copy(); p_error['w_speed'] += 1
df_error = run_simulation(p_error)

# --- ІНТЕРФЕЙС ---
st.title("🎯 Ballistic Master Pro v9.0")

col_main, col_sens = st.columns([2, 1])

with col_main:
    # Графік траєкторії
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Range'], y=df['Drop_cm'], name="Траєкторія", line=dict(color='lime', width=3)))
    fig.update_layout(template="plotly_dark", title="Падіння кулі (см)", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_sens:
    st.subheader("⚠️ Аналіз помилки")
    last_main = df.iloc[-1]
    last_err = df_error.iloc[-1]
    
    wind_error_cm = abs(last_err['Wind_cm'] - last_main['Wind_cm'])
    st.error(f"Помилка у вітрі на 1 м/с змістить кулю на **{round(wind_error_cm, 1)} см**")
    
    # Сітка прицілу з зоною розсіювання
    reticle = go.Figure()
    reticle.add_shape(type="circle", x0=-wind_error_cm/10, y0=last_main['Drop_cm']/10-1, x1=wind_error_cm/10, y1=last_main['Drop_cm']/10+1, 
                      line_color="yellow", fillcolor="rgba(255, 255, 0, 0.2)")
    reticle.add_trace(go.Scatter(x=[last_main['Wind_cm']/10], y=[last_main['Drop_cm']/10], mode="markers", marker=dict(color="red", size=10)))
    reticle.update_layout(template="plotly_dark", width=300, height=300, title="Зона невизначеності (MRAD)",
                          xaxis=dict(range=[-3, 3]), yaxis=dict(range=[-15, 2]))
    st.plotly_chart(reticle)

# Таблиця
st.subheader("📋 Картка вогню")
st.dataframe(df[df['Range'] % 100 == 0], use_container_width=True)

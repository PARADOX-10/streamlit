import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Magelan242 Ballistic v12.0", layout="wide")

# --- РОЗШИРЕНЕ МАТЕМАТИЧНЕ ЯДРО ---
def run_simulation(p):
    # Температурна стабільність пороху
    v0_corr = p['v0'] + (p['temp'] - 15) * p['t_coeff']
    
    # Модель атмосфери ICAO
    tk = p['temp'] + 273.15
    rho = (p['pressure'] * 100) / (287.05 * tk)
    
    # Розрахунок опору
    k_drag = 0.5 * rho * (1/p['bc']) * 0.00052
    if p['model'] == "G7": k_drag *= 0.91

    results = []
    g = 9.80665
    weight_kg = p['weight_gr'] * 0.0000647989
    angle_rad = math.radians(p['angle'])

    for d in range(0, p['max_dist'] + 1, 10):
        t = d / (v0_corr * math.exp(-k_drag * d / 2)) if d > 0 else 0
        
        # Вертикаль з урахуванням кута місця цілі
        drop = 0.5 * g * (t**2) * math.cos(angle_rad)
        t_zero = p['zero_dist'] / (v0_corr * math.exp(-k_drag * p['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        y_m = -(drop - (drop_zero + p['sh']/100) * (d / p['zero_dist']) + p['sh']/100)
        
        # Горизонталь: Вітер + Деривація
        wind_rad = math.radians(p['w_dir'] * 30)
        wind_drift = (p['w_speed'] * math.sin(wind_rad)) * (t - (d/v0_corr)) if d > 0 else 0
        derivation = 0.05 * (p['twist'] / 10) * (d / 100)**2 if d > 0 else 0
        
        v_curr = v0_corr * math.exp(-k_drag * d)
        energy = (weight_kg * v_curr**2) / 2
        
        results.append({
            "Range": d, "Drop_cm": y_m * 100, "Wind_cm": wind_drift * 100,
            "Deriv_cm": derivation * 100, "V": v_curr, "E": energy,
            "MRAD_V": (y_m * 100) / (d / 10) if d > 0 else 0,
            "MRAD_H": ((wind_drift + derivation) * 100) / (d / 10) if d > 0 else 0
        })
    return pd.DataFrame(results), v0_corr

# --- SIDEBAR: ПОВНЕ МЕНЮ НАЛАШТУВАНЬ ---
st.sidebar.title("🎮 Центр керування")

# Створюємо 4 основні секції
with st.sidebar.expander("🚀 ПАРАМЕТРИ НАБОЮ", expanded=True):
    v0 = st.number_input("Початкова швидкість V0 (м/с)", 200.0, 1500.0, 893.0)
    bc = st.number_input("Балістичний коефіцієнт (BC)", 0.01, 2.0, 0.584, format="%.3f")
    model = st.selectbox("Модель опору", ["G1", "G7"])
    weight = st.number_input("Вага кулі (gr/грани)", 1.0, 1000.0, 195.0)
    t_coeff = st.number_input("Термозалежність (м/с на 1°C)", 0.0, 5.0, 0.2)

with st.sidebar.expander("🔭 ПАРАМЕТРИ ЗБРОЇ"):
    sh = st.number_input("Висота прицілу (см)", 0.0, 20.0, 5.0)
    zero_dist = st.number_input("Дистанція пристрілки (м)", 1, 1000, 100)
    twist = st.number_input("Твіст ствола (дюйми)", 5.0, 20.0, 11.0)
    click_val = st.number_input("Ціна кліка (MRAD)", 0.01, 1.0, 0.1)

with st.sidebar.expander("🌍 СЕРЕДОВИЩЕ"):
    temp = st.slider("Температура (°C)", -40, 60, 15)
    pressure = st.number_input("Тиск (hPa / mbar)", 500, 1100, 1013)
    humidity = st.slider("Вологість (%)", 0, 100, 50)
    angle = st.slider("Кут місця цілі (°)", -60, 60, 0)

with st.sidebar.expander("🌬️ ВІТЕР"):
    w_speed = st.slider("Швидкість вітру (м/с)", 0.0, 25.0, 0.0)
    w_dir = st.slider("Напрямок вітру (год)", 1, 12, 12)
    max_dist = st.slider("Макс. дистанція розрахунку (м)", 100, 2500, 1000, 100)

# Розрахунок
p = {'v0': v0, 'bc': bc, 'model': model, 'weight_gr': weight, 'temp': temp, 
     'pressure': pressure, 'w_speed': w_speed, 'w_dir': w_dir, 'angle': angle,
     'twist': twist, 'zero_dist': zero_dist, 'max_dist': max_dist, 'sh': sh, 't_coeff': t_coeff}

df, v0_final = run_simulation(p)

# --- ОСНОВНИЙ ІНТЕРФЕЙС ---
st.title("🏹🚀 Magelan242 Ballistic v12.0")

# Картки швидкого доступу
c1, c2, c3, c4 = st.columns(4)
res = df.iloc[-1]
c1.metric("V0 (Коригована)", f"{v0_final:.1f} м/с")
c2.metric("Вертикаль (MRAD)", round(abs(res['MRAD_V']), 2))
c3.metric("Горизонталь (MRAD)", round(abs(res['MRAD_H']), 2))
c4.metric("Кліки (Вертикаль)", int(abs(res['MRAD_V'] / click_val)))

# Графіки
fig = make_subplots(rows=2, cols=2, subplot_titles=("Траєкторія (см)", "Знесення (Вітер+Дер, см)", "Швидкість (м/с)", "Енергія (Дж)"))
fig.add_trace(go.Scatter(x=df['Range'], y=df['Drop_cm'], name="Drop", line=dict(color='lime')), 1, 1)
fig.add_trace(go.Scatter(x=df['Range'], y=df['Wind_cm']+df['Deriv_cm'], name="Windage", line=dict(color='cyan')), 1, 2)
fig.add_trace(go.Scatter(x=df['Range'], y=df['V'], name="Velocity", line=dict(color='orange')), 2, 1)
fig.add_trace(go.Scatter(x=df['Range'], y=df['E'], name="Energy", line=dict(color='red')), 2, 2)
fig.update_layout(height=700, template="plotly_dark", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Професійна таблиця
st.subheader("📋 Детальна балістична таблиця")
st.dataframe(df[df['Range'] % 100 == 0], use_container_width=True)

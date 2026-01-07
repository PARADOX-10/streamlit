import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Налаштування інтерфейсу
st.set_page_config(page_title="Magelan242 Ballistics v15.5", layout="wide")

# --- МАТЕМАТИЧНЕ ЯДРО ---
def run_simulation(p):
    v0_corr = p['v0'] + (p['temp'] - 15) * p['t_coeff']
    tk = p['temp'] + 273.15
    rho = (p['pressure'] * 100) / (287.05 * tk)
    
    k_drag = 0.5 * rho * (1/p['bc']) * 0.00052
    if p['model'] == "G7": k_drag *= 0.91

    results = []
    g = 9.80665
    weight_kg = p['weight_gr'] * 0.0000647989
    angle_rad = math.radians(p['angle'])
    
    # Крок розрахунку 1 метр
    for d in range(0, p['max_dist'] + 1, 1):
        t = d / (v0_corr * math.exp(-k_drag * d / 2)) if d > 0 else 0
        
        # Падіння
        drop = 0.5 * g * (t**2) * math.cos(angle_rad)
        t_zero = p['zero_dist'] / (v0_corr * math.exp(-k_drag * p['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        y_m = -(drop - (drop_zero + p['sh']/100) * (d / p['zero_dist']) + p['sh']/100)
        
        # Вітер та Деривація
        wind_rad = math.radians(p['w_dir'] * 30)
        wind_drift = (p['w_speed'] * math.sin(wind_rad)) * (t - (d/v0_corr)) if d > 0 else 0
        derivation = 0.05 * (p['twist'] / 10) * (d / 100)**2 if d > 0 else 0
        
        v_curr = v0_corr * math.exp(-k_drag * d)
        energy = (weight_kg * v_curr**2) / 2
        
        # Розрахунок поправки в MRAD та Кліках (1 клік = 0.1 MRAD)
        mrad_v = (y_m * 100) / (d / 10) if d > 0 else 0
        clicks_v = round(abs(mrad_v) / 0.1, 1) if d > 0 else 0
        
        if d % 5 == 0 or d == p['max_dist']:
            results.append({
                "Дистанція (м)": d,
                "Падіння (см)": round(y_m * 100, 2),
                "Поправка MRAD": round(abs(mrad_v), 2),
                "Кліки (0.1)": clicks_v,
                "Знесення (см)": round((wind_drift + derivation) * 100, 2),
                "Швидкість (м/с)": round(v_curr, 1),
                "Енергія (Дж)": int(energy)
            })
            
    return pd.DataFrame(results), v0_corr

# --- БОКОВЕ МЕНЮ ---
st.sidebar.title("🛡️ Magelan242 Ballistics")
st.sidebar.markdown("**Стандарт: 1 клік = 1 см / 100 м**")

with st.sidebar.expander("🚀 НАБІЙ", expanded=True):
    v0 = st.number_input("V0 (м/с)", 200.0, 1500.0, 825.0, step=1.0)
    weight = st.number_input("Вага кулі (гран)", 1.0, 1000.0, 168.0)
    bc = st.number_input("Бал. коефіцієнт (BC)", 0.01, 2.0, 0.450, format="%.3f")
    model = st.selectbox("Модель", ["G7", "G1"])
    t_coeff = st.number_input("Термозалежність (м/с на 1°C)", 0.0, 2.0, 0.2)

with st.sidebar.expander("🔭 ЗБРОЯ"):
    sh = st.number_input("Висота прицілу (см)", 0.0, 30.0, 5.0)
    zero_dist = st.number_input("Пристрілка (м)", 1, 1000, 100)
    twist = st.number_input("Твіст ствола (дюйми)", 5.0, 20.0, 10.0)

with st.sidebar.expander("🌍 УМОВИ"):
    temp = st.slider("Температура (°C)", -40, 60, 15)
    press = st.number_input("Тиск (hPa)", 500, 1100, 1013)
    w_speed = st.slider("Вітер (м/с)", 0.0, 30.0, 3.0)
    w_dir = st.slider("Напрямок (год)", 1, 12, 3)
    max_d = st.number_input("Макс. дистанція (м)", 10, 5000, 1000, step=1)
    angle = st.slider("Кут стрільби (°)", -80, 80, 0)

# Розрахунок
params = {'v0': v0, 'bc': bc, 'model': model, 'weight_gr': weight, 'temp': temp, 
          'pressure': press, 'w_speed': w_speed, 'w_dir': w_dir, 'angle': angle,
          'twist': twist, 'zero_dist': zero_dist, 'max_dist': max_d, 'sh': sh, 't_coeff': t_coeff}

try:
    df, v0_final = run_simulation(params)
    res = df.iloc[-1]

    # --- ІНТЕРФЕЙС ---
    st.header(f"🎯 Картка вогню Magelan242 (ELR {max_d}м)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("V0 Коригована", f"{v0_final:.1f} м/с")
    c2.metric("Поправка (MRAD)", res['Поправка MRAD'])
    c3.metric("Кліки (0.1 MRAD)", int(res['Кліки (0.1)']))
    c4.metric("Енергія (Дж)", res['Енергія (Дж)'])

    # Графік
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Падіння (см)'], name="Траєкторія", line=dict(color='lime', width=3)))
    fig.update_layout(template="plotly_dark", title="Падіння кулі (см)", height=450, xaxis_title="Дистанція (м)", yaxis_title="см")
    st.plotly_chart(fig, use_container_width=True)

    # Таблиця
    st.subheader("📋 Детальна таблиця поправок")
    step = st.selectbox("Крок таблиці:", [1, 5, 10, 25, 50, 100], index=4)
    st.dataframe(df[df['Дистанція (м)'] % step == 0], use_container_width=True)

except Exception as e:
    st.error(f"Помилка: {e}")

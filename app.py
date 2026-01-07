import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Налаштування інтерфейсу
st.set_page_config(page_title="Балістичний калькулятор Magelan242 v14.0", layout="wide")

# --- МАТЕМАТИЧНЕ ЯДРО ---
def run_simulation(p):
    # Корекція швидкості на температуру
    v0_corr = p['v0'] + (p['temp'] - 15) * p['t_coeff']
    
    # Модель атмосфери
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
        
        results.append({
            "Дистанція (м)": d,
            "Падіння (см)": round(y_m * 100, 1),
            "Знесення (см)": round((wind_drift + derivation) * 100, 1),
            "MRAD Верт": round((y_m * 100) / (d / 10), 2) if d > 0 else 0,
            "Швидкість (м/с)": round(v_curr, 1),
            "Енергія (Дж)": int(energy)
        })
    return pd.DataFrame(results), v0_corr

# --- БОКОВЕ МЕНЮ ---
st.sidebar.title("🛡️ Magelan242 Ballistics")

tab_ammo, tab_rifle, tab_env = st.sidebar.tabs(["🚀 Набій", "🔭 Зброя", "🌍 Умови"])

with tab_ammo:
    v0 = st.number_input("Початкова швидкість V0 (м/с)", 200.0, 1500.0, 893.0)
    weight = st.number_input("Вага кулі (гран)", 1.0, 800.0, 195.0)
    
    # Розрахунок енергії
    weight_kg_calc = weight * 0.0000647989
    theoretical_energy = int((weight_kg_calc * v0**2) / 2)
    input_energy = st.number_input("Енергія набою (Дж)", value=theoretical_energy)
    
    bc = st.number_input("Бал. коефіцієнт (BC)", 0.01, 1.5, 0.584, format="%.3f")
    model = st.selectbox("Модель опору", ["G1", "G7"])
    t_coeff = st.number_input("Термозалежність (м/с на 1°C)", 0.0, 2.0, 0.2)

with tab_rifle:
    sh = st.number_input("Висота прицілу (см)", 0.0, 15.0, 5.0)
    zero_dist = st.number_input("Пристрілка (м)", 1, 1000, 300)
    twist = st.number_input("Твіст ствола (дюйми)", 5.0, 20.0, 11.0)
    click_val = st.number_input("Ціна кліка (MRAD)", 0.01, 1.0, 0.1)

with tab_env:
    temp = st.slider("Температура (°C)", -35, 50, 15)
    press = st.number_input("Тиск (hPa)", 800, 1100, 1013)
    w_speed = st.slider("Швидкість вітру (м/с)", 0.0, 20.0, 0.0)
    w_dir = st.slider("Напрямок (год)", 1, 12, 12)
    angle = st.slider("Кут стрільби (°)", -60, 60, 0)
    max_d = st.slider("Макс. дистанція (м)", 100, 2000, 1000, 1)

# Розрахунок
params = {'v0': v0, 'bc': bc, 'model': model, 'weight_gr': weight, 'temp': temp, 
          'pressure': press, 'w_speed': w_speed, 'w_dir': w_dir, 'angle': angle,
          'twist': twist, 'zero_dist': zero_dist, 'max_dist': max_d, 'sh': sh, 't_coeff': t_coeff}

try:
    df, v0_final = run_simulation(params)

    # --- ІНТЕРФЕЙС ---
    st.header("🎯 Балістичний калькулятор Magelan242 v14.0")

    c1, c2, c3, c4 = st.columns(4)
    res = df.iloc[-1]
    c1.metric("V0 Коригована", f"{v0_final:.1f} м/с")
    c2.metric("Поправка MRAD", abs(res['MRAD Верт']))
    c3.metric("Кліки", int(abs(res['MRAD Верт'] / click_val)))
    c4.metric("Енергія в цілі", f"{res['Енергія (Дж)']} Дж")

    # Графіки
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Траєкторія (см)", "Енергія (Дж)"))
    fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Падіння (см)'], name="Падіння", line=dict(color='lime')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Енергія (Дж)'], name="Енергія", fill='tozeroy', line=dict(color='red')), row=2, col=1)

    targets = {"Мала ціль (400Дж)": 400, "Середня ціль (1000Дж)": 1000, "Велика ціль (2000Дж)": 2000}
    for name, val in targets.items():
        if input_energy > val:
            fig.add_hline(y=val, line_dash="dot", annotation_text=name, row=2, col=1)

    fig.update_layout(height=700, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Аналіз ефективності
    st.subheader("📊 Дистанція ефективного ураження")
    cols = st.columns(len(targets))
    for i, (name, val) in enumerate(targets.items()):
        eff_dist = df[df['Енергія (Дж)'] >= val]['Дистанція (м)'].max()
        if pd.isna(eff_dist): eff_dist = 0
        cols[i].info(f"**{name.split(' (')[0]}**\n\nДо: **{int(eff_dist)} м**")

    # Таблиця
    st.subheader("📋 Таблиця поправок")
    st.dataframe(df[df['Дистанція (м)'] % 100 == 0], use_container_width=True)

except Exception as e:
    st.error(f"Виникла помилка в розрахунках: {e}")

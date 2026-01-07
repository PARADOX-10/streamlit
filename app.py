import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Настройка интерфейса
st.set_page_config(page_title="Magelan242 Ballistics v14.0", layout="wide")

# --- МАТЕМАТИЧЕСКОЕ ЯДРО ---
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

    for d in range(0, p['max_dist'] + 1, 10):
        t = d / (v0_corr * math.exp(-k_drag * d / 2)) if d > 0 else 0
        drop = 0.5 * g * (t**2) * math.cos(angle_rad)
        t_zero = p['zero_dist'] / (v0_corr * math.exp(-k_drag * p['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        y_m = -(drop - (drop_zero + p['sh']/100) * (d / p['zero_dist']) + p['sh']/100)
        
        wind_rad = math.radians(p['w_dir'] * 30)
        wind_drift = (p['w_speed'] * math.sin(wind_rad)) * (t - (d/v0_corr)) if d > 0 else 0
        derivation = 0.05 * (p['twist'] / 10) * (d / 100)**2 if d > 0 else 0
        
        v_curr = v0_corr * math.exp(-k_drag * d)
        energy = (weight_kg * v_curr**2) / 2
        
        results.append({
            "Дистанция (м)": d,
            "Падение (см)": y_m * 100,
            "Снос (см)": (wind_drift + derivation) * 100,
            "MRAD Верт": (y_m * 100) / (d / 10) if d > 0 else 0,
            "Скорость (м/с)": v_curr,
            "Энергия (Дж)": int(energy)
        })
    return pd.DataFrame(results), v0_corr

# --- БОКОВОЕ МЕНЮ ---
st.sidebar.title("🛡️ Magelan242 Ballistics")

tab_ammo, tab_rifle, tab_env = st.sidebar.tabs(["🚀 Набой", "🔭 Оружие", "🌍 Среда"])

with tab_ammo:
    st.subheader("Характеристики пули")
    v0 = st.number_input("Начальная скорость V0 (м/с)", 200.0, 1500.0, 825.0)
    weight = st.number_input("Вес пули (gr)", 1.0, 800.0, 168.0)
    
    # Расчет энергии
    weight_kg_calc = weight * 0.0000647989
    theoretical_energy = int((weight_kg_calc * v0**2) / 2)
    
    input_energy = st.number_input("Энергия набоя (Дж)", value=theoretical_energy)
    
    bc = st.number_input("Бал. коэффициент (BC)", 0.01, 1.5, 0.450, format="%.3f")
    model = st.selectbox("Модель сопротивления", ["G1", "G7"])
    t_coeff = st.number_input("Термозависимость (м/с на 1°C)", 0.0, 2.0, 0.2)

with tab_rifle:
    sh = st.number_input("Высота прицела (см)", 0.0, 15.0, 5.0)
    zero_dist = st.number_input("Пристрелка (м)", 1, 1000, 100)
    twist = st.number_input("Твист ствола (дюймы)", 5.0, 20.0, 10.0)
    click_val = st.number_input("Цена клика (MRAD)", 0.01, 1.0, 0.1)

with tab_env:
    temp = st.slider("Температура (°C)", -35, 50, 15)
    press = st.number_input("Давление (hPa)", 800, 1100, 1013)
    w_speed = st.slider("Скорость ветра (м/с)", 0.0, 20.0, 3.0)
    w_dir = st.slider("Направление (час)", 1, 12, 3)
    angle = st.slider("Угол стрельбы (°)", -60, 60, 0)
    max_d = st.slider("Макс. дистанция (м)", 100, 2000, 1000, 100)

# Расчет
params = {'v0': v0, 'bc': bc, 'model': model, 'weight_gr': weight, 'temp': temp, 
          'pressure': press, 'w_speed': w_speed, 'w_dir': w_dir, 'angle': angle,
          'twist': twist, 'zero_dist': zero_dist, 'max_dist': max_d, 'sh': sh, 't_coeff': t_coeff}

df, v0_final = run_simulation(params)

# --- ОСНОВНОЙ ИНТЕРФЕЙС ---
st.header("🎯 Аналитический центр Magelan242")

# Метрики
c1, c2, c3, c4 = st.columns(4)
res = df.iloc[-1]
c1.metric("V0 Коррект.", f"{v0_final:.1f} м/с")
c2.metric("Поправка MRAD", round(abs(res['MRAD Верт']), 2))
c3.metric("Клики", int(abs(res['MRAD Верт'] / click_val)))
c4.metric("Энергия у цели", f"{res['Энергія (Дж)']} Дж")

# Графики
fig = make_subplots(rows=2, cols=1, subplot_titles=("Траектория (см)", "Энергия (Дж) и Эффективность"))
fig.add_trace(go.Scatter(x=df['Дистанция (м)'], y=df['Падение (см)'], name="Падение", line=dict(color='lime')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Дистанция (м)'], y=df['Энергия (Дж)'], name="Энергия", fill='tozeroy', line=dict(color='red')), row=2, col=1)

# Линии порогов энергии
targets = {"Мелкая дичь (400Дж)": 400, "Средняя дичь (1000Дж)": 1000, "Крупная дичь (2000Дж)": 2000}
for name, val in targets.items():
    if theoretical_energy > val:
        fig.add_hline(y=val, line_dash="dot", annotation_text=name, row=2, col=1)

fig.update_layout(height=700, template="plotly_dark", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Анализ дистанции эффективности
st.subheader("📊 Анализ дистанции эффективного выстрела")
cols = st.columns(len(targets))
for i, (name, val) in enumerate(targets.items()):
    eff_dist = df[df['Энергия (Дж)'] >= val]['Дистанция (м)'].max()
    if pd.isna(eff_dist): eff_dist = 0
    cols[i].info(f"**{name.split(' (')[0]}**\n\nДо: **{eff_dist} м**")

# Таблица
st.subheader("📋 Таблица поправок")
st.dataframe(df[df['Дистанция (м)'] % 100 == 0], use_container_width=True)

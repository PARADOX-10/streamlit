
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Налаштування інтерфейсу
st.set_page_config(page_title="Magelan242 Ballistics v13.0", layout="wide")

# --- МАТЕМАТИЧНЕ ЯДРО ---
def run_simulation(p):
    # Термозалежність швидкості (зміна V0 від температури)
    v0_corr = p['v0'] + (p['temp'] - 15) * p['t_coeff']
    
    # Розрахунок щільності повітря (модель ICAO)
    tk = p['temp'] + 273.15
    rho = (p['pressure'] * 100) / (287.05 * tk)
    
    # Розрахунок коефіцієнта опору
    k_drag = 0.5 * rho * (1/p['bc']) * 0.00052
    if p['model'] == "G7": k_drag *= 0.91

    results = []
    g = 9.80665
    weight_kg = p['weight_gr'] * 0.0000647989
    angle_rad = math.radians(p['angle'])

    for d in range(0, p['max_dist'] + 1, 10):
        t = d / (v0_corr * math.exp(-k_drag * d / 2)) if d > 0 else 0
        
        # Вертикальне падіння з урахуванням кута нахилу
        drop = 0.5 * g * (t**2) * math.cos(angle_rad)
        t_zero = p['zero_dist'] / (v0_corr * math.exp(-k_drag * p['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        y_m = -(drop - (drop_zero + p['sh']/100) * (d / p['zero_dist']) + p['sh']/100)
        
        # Горизонтальні фактори (Вітер + Деривація)
        wind_rad = math.radians(p['w_dir'] * 30)
        wind_drift = (p['w_speed'] * math.sin(wind_rad)) * (t - (d/v0_corr)) if d > 0 else 0
        derivation = 0.05 * (p['twist'] / 10) * (d / 100)**2 if d > 0 else 0
        
        v_curr = v0_corr * math.exp(-k_drag * d)
        energy = (weight_kg * v_curr**2) / 2
        
        mrad_v = (y_m * 100) / (d / 10) if d > 0 else 0
        mrad_h = ((wind_drift + derivation) * 100) / (d / 10) if d > 0 else 0
        
        results.append({
            "Дистанція (м)": d,
            "Падіння (см)": y_m * 100,
            "Вітер+Дер (см)": (wind_drift + derivation) * 100,
            "MRAD Верт": mrad_v,
            "MRAD Гориз": mrad_h,
            "Швидкість (м/с)": v_curr,
            "Енергія (Дж)": int(energy)
        })
    return pd.DataFrame(results), v0_corr

# --- БОКОВЕ МЕНЮ ---
st.sidebar.title("🛡️ Magelan242 Ballistics")

tab_ammo, tab_rifle, tab_env = st.sidebar.tabs(["🚀 Набій", "🔭 Зброя", "🌍 Середовище"])

with tab_ammo:
    v0 = st.number_input("Початкова швидкість V0 (м/с)", 200.0, 1500.0, 825.0)
    bc = st.number_input("Бал. коефіцієнт (BC)", 0.01, 1.5, 0.450, format="%.3f")
    model = st.selectbox("Модель опору", ["G1", "G7"])
    weight = st.number_input("Вага кулі (gr)", 1.0, 800.0, 168.0)
    t_coeff = st.number_input("Термозалежність (м/с на 1°C)", 0.0, 2.0, 0.2)

with tab_rifle:
    sh = st.number_input("Висота прицілу (см)", 0.0, 15.0, 5.0)
    zero_dist = st.number_input("Пристрілка (м)", 1, 1000, 100)
    twist = st.number_input("Твіст ствола (дюйми)", 5.0, 20.0, 10.0)
    click_val = st.number_input("Ціна кліка (MRAD)", 0.01, 1.0, 0.1)

with tab_env:
    temp = st.slider("Температура повітря (°C)", -35, 50, 15)
    press = st.number_input("Тиск (hPa)", 800, 1100, 1013)
    w_speed = st.slider("Швидкість вітру (м/с)", 0.0, 20.0, 3.0)
    w_dir = st.slider("Напрямок (год)", 1, 12, 3)
    angle = st.slider("Кут стрільби (°)", -60, 60, 0)
    max_d = st.slider("Макс. дистанція (м)", 100, 2000, 1000, 100)

# Розрахунок
params = {'v0': v0, 'bc': bc, 'model': model, 'weight_gr': weight, 'temp': temp, 
          'pressure': press, 'w_speed': w_speed, 'w_dir': w_dir, 'angle': angle,
          'twist': twist, 'zero_dist': zero_dist, 'max_dist': max_d, 'sh': sh, 't_coeff': t_coeff}

df, v0_final = run_simulation(params)

# --- ОСНОВНИЙ ІНТЕРФЕЙС ---
st.header("🎯 Аналітичний центр Magelan242")

# Метрики
c1, c2, c3, c4 = st.columns(4)
res = df.iloc[-1]
c1.metric("V0 Коригована", f"{v0_final:.1f} м/с")
c2.metric("Поправка MRAD", round(abs(res['MRAD Верт']), 2))
c3.metric("Кліки", int(abs(res['MRAD Верт'] / click_val)))
c4.metric("Енергія (ціль)", f"{res['Енергія (Дж)']} Дж")

# Графіки
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=("Траєкторія падіння кулі (см)", "Швидкість та перехід у дозвук (м/с)"))

fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Падіння (см)'], name="Падіння", line=dict(color='#00ff00', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Швидкість (м/с)'], name="Швидкість", line=dict(color='#ffa500', width=3)), row=2, col=1)
fig.add_hline(y=340, line_dash="dash", line_color="red", row=2, col=1, annotation_text="340 м/с (Дозвук)")

fig.update_layout(height=700, template="plotly_dark", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Картка вогню
st.subheader("📋 Робоча таблиця поправок")
table_df = df[df['Дистанція (м)'] % 100 == 0].copy()
st.dataframe(table_df.style.format(precision=2), use_container_width=True)

st.download_button("📥 Завантажити CSV для друку", df.to_csv(index=False), "Magelan242_RangeCard.csv")

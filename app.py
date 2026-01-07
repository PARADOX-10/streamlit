import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Конфігурація сторінки
st.set_page_config(page_title="Балістичний Командний Центр v12.5", layout="wide")

# --- РОЗШИРЕНЕ МАТЕМАТИЧНЕ ЯДРО ---
def run_simulation(p):
    # Температурна корекція швидкості пороху
    v0_corr = p['v0'] + (p['temp'] - 15) * p['t_coeff']
    
    # Модель атмосфери (щільність повітря)
    tk = p['temp'] + 273.15
    rho = (p['pressure'] * 100) / (287.05 * tk)
    
    # Коефіцієнт опору
    k_drag = 0.5 * rho * (1/p['bc']) * 0.00052
    if p['model'] == "G7": k_drag *= 0.91

    results = []
    g = 9.80665
    weight_kg = p['weight_gr'] * 0.0000647989
    angle_rad = math.radians(p['angle'])

    for d in range(0, p['max_dist'] + 1, 10):
        t = d / (v0_corr * math.exp(-k_drag * d / 2)) if d > 0 else 0
        
        # Розрахунок вертикального падіння (з урахуванням кута нахилу)
        drop = 0.5 * g * (t**2) * math.cos(angle_rad)
        t_zero = p['zero_dist'] / (v0_corr * math.exp(-k_drag * p['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        y_m = -(drop - (drop_zero + p['sh']/100) * (d / p['zero_dist']) + p['sh']/100)
        
        # Горизонтальні фактори: Вітер та Деривація (обертання)
        wind_rad = math.radians(p['w_dir'] * 30)
        wind_drift = (p['w_speed'] * math.sin(wind_rad)) * (t - (d/v0_corr)) if d > 0 else 0
        derivation = 0.05 * (p['twist'] / 10) * (d / 100)**2 if d > 0 else 0
        
        v_curr = v0_corr * math.exp(-k_drag * d)
        energy = (weight_kg * v_curr**2) / 2
        
        results.append({
            "Дистанція (м)": d, 
            "Падіння (см)": y_m * 100, 
            "Знесення (см)": (wind_drift + derivation) * 100,
            "Швидкість (м/с)": v_curr, 
            "Енергія (Дж)": energy,
            "Вертикаль (MRAD)": (y_m * 100) / (d / 10) if d > 0 else 0,
            "Горизонталь (MRAD)": ((wind_drift + derivation) * 100) / (d / 10) if d > 0 else 0
        })
    return pd.DataFrame(results), v0_corr

# --- БОКОВЕ МЕНЮ: ВСІ НАЛАШТУВАННЯ ---
st.sidebar.title("🎮 Налаштування системи")

with st.sidebar.expander("🚀 ХАРАКТЕРИСТИКИ НАБОЮ", expanded=True):
    v0 = st.number_input("Початкова швидкість V0 (м/с)", 200.0, 1500.0, 820.0)
    bc = st.number_input("Балістичний коефіцієнт (BC)", 0.01, 2.0, 0.450, format="%.3f")
    model = st.selectbox("Балістична модель кулі", ["G1", "G7"])
    weight = st.number_input("Вага кулі (грани / gr)", 1.0, 1000.0, 168.0)
    t_coeff = st.number_input("Термозалежність пороху (м/с на 1°C)", 0.0, 5.0, 0.2)

with st.sidebar.expander("🔭 ПАРАМЕТРИ ЗБРОЇ"):
    sh = st.number_input("Висота осі прицілу (см)", 0.0, 20.0, 5.0)
    zero_dist = st.number_input("Дистанція пристрілки (м)", 1, 1000, 100)
    twist = st.number_input("Крок нарізів / Твіст (дюйми)", 5.0, 20.0, 10.0)
    click_val = st.number_input("Ціна кліка барабана (MRAD)", 0.01, 1.0, 0.1)

with st.sidebar.expander("🌍 АТМОСФЕРА ТА ЛАНДШАФТ"):
    temp = st.slider("Температура повітря (°C)", -40, 60, 15)
    pressure = st.number_input("Атмосферний тиск (hPa)", 500, 1100, 1013)
    angle = st.slider("Кут стрільби вгору/вниз (°)", -60, 60, 0)

with st.sidebar.expander("🌬️ ВІТЕР ТА ДИСТАНЦІЯ"):
    w_speed = st.slider("Швидкість вітру (м/с)", 0.0, 25.0, 3.0)
    w_dir = st.slider("Напрямок вітру (год)", 1, 12, 3)
    max_dist = st.slider("Максимальна дистанція (м)", 100, 2500, 1000, 100)

# Проведення розрахунків
params = {'v0': v0, 'bc': bc, 'model': model, 'weight_gr': weight, 'temp': temp, 
          'pressure': pressure, 'w_speed': w_speed, 'w_dir': w_dir, 'angle': angle,
          'twist': twist, 'zero_dist': zero_dist, 'max_dist': max_dist, 'sh': sh, 't_coeff': t_coeff}

df, v0_real = run_simulation(params)

# --- ОСНОВНИЙ ЕКРАН ---
st.title("🏹 Балістичний Master Pro v12.5")

# Панель головних показників
c1, c2, c3, c4 = st.columns(4)
res_end = df.iloc[-1]
c1.metric("V0 з корекцією", f"{v0_real:.1f} м/с")
c2.metric("Вертикаль (MRAD)", round(abs(res_end['Вертикаль (MRAD)']), 2))
c3.metric("Горизонталь (MRAD)", round(abs(res_end['Горизонталь (MRAD)']), 2))
c4.metric("Кліки барабана", int(abs(res_end['Вертикаль (MRAD)'] / click_val)))

# Графічний аналіз
fig = make_subplots(rows=2, cols=2, 
                    subplot_titles=("Траєкторія (Падіння, см)", "Знесення (Вітер+Деривація, см)", 
                                    "Швидкість (м/с)", "Енергія кулі (Дж)"))

fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Падіння (см)'], name="Падіння", line=dict(color='lime')), 1, 1)
fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Знесення (см)'], name="Знесення", line=dict(color='cyan')), 1, 2)
fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Швидкість (м/с)'], name="Швидкість", line=dict(color='orange')), 2, 1)
fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Енергія (Дж)'], name="Енергія", line=dict(color='red')), 2, 2)

# Додавання лінії звукового бар'єру
fig.add_hline(y=340, line_dash="dash", line_color="white", row=2, col=1, annotation_text="Звук")

fig.update_layout(height=750, template="plotly_dark", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Секція таблиці
st.subheader("📋 Таблиця поправок (Range Card)")
# Форматування для виводу
formatted_df = df[df['Дистанція (м)'] % 100 == 0].copy()
st.dataframe(formatted_df.style.format(precision=2), use_container_width=True)

# Кнопка для завантаження
st.download_button("📥 Завантажити результати (CSV)", df.to_csv(index=False), "ballistics_report.csv", "text/csv")

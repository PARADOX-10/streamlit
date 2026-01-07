import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# Налаштування сторінки
st.set_page_config(page_title="Ballistic Expert Pro v3.0", layout="wide")

# --- БАЗА ДАНИХ НАБОЇВ ---
AMMO_DB = {
    "Custom (Ручне введення)": {"v0": 800, "bc": 0.450, "model": "G1", "weight": 10.0},
    ".223 Rem (55 gr FMJ)": {"v0": 980, "bc": 0.243, "model": "G1", "weight": 3.56},
    ".308 Win (168 gr BTHP)": {"v0": 820, "bc": 0.450, "model": "G1", "weight": 10.89},
    ".308 Win (175 gr SMK G7)": {"v0": 790, "bc": 0.243, "model": "G7", "weight": 11.34},
    ".300 Win Mag (190 gr)": {"v0": 890, "bc": 0.530, "model": "G1", "weight": 12.31},
    ".338 Lapua Mag (250 gr)": {"v0": 900, "bc": 0.625, "model": "G1", "weight": 16.20},
    "6.5 Creedmoor (140 gr ELD)": {"v0": 825, "bc": 0.326, "model": "G7", "weight": 9.07},
    "7.62x39 (123 gr FMJ)": {"v0": 715, "bc": 0.275, "model": "G1", "weight": 8.0},
    ".50 BMG (655 gr)": {"v0": 920, "bc": 0.700, "model": "G1", "weight": 42.44}
}

# --- МАТЕМАТИЧНІ ФУНКЦІЇ ---
def get_air_density(temp, pressure, humidity):
    tk = temp + 273.15
    p_pa = pressure * 100
    # Спрощений облік вологості через щільність
    rho = p_pa / (287.05 * tk) * (1 - 0.378 * (humidity/100 * 6.112 * math.exp(17.62*temp/(243.12+temp))/pressure))
    return rho

def run_simulation(params):
    v0 = params['v0'] + (params['temp'] - 15) * params['t_coeff'] # Термозалежність
    rho = get_air_density(params['temp'], params['pressure'], params['humidity'])
    
    # Базовий коефіцієнт опору
    k_drag = 0.5 * rho * (1/params['bc']) * 0.00052
    if params['model'] == "G7": k_drag *= 0.91 # Корекція форми

    results = []
    g = 9.80665
    angle_rad = math.radians(params['angle'])
    
    for d in range(0, params['max_dist'] + 1, 10):
        # Час польоту (ітеративно для точності)
        t = d / (v0 * math.exp(-k_drag * d / 2)) if d > 0 else 0
        
        # Падіння (вертикаль)
        drop = 0.5 * g * (t**2) * math.cos(angle_rad)
        t_zero = params['zero_dist'] / (v0 * math.exp(-k_drag * params['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        
        y_m = -(drop - (drop_zero + params['sh']/100) * (d / params['zero_dist']) + params['sh']/100)
        
        # Вітер
        wind_rad = math.radians(params['w_dir'] * 30)
        wind_drift = (params['w_speed'] * math.sin(wind_rad)) * (t - (d/v0)) if d > 0 else 0
        
        # Деривація
        derivation = 0.05 * (params['twist'] / 10) * (d / 100)**2 if d > 0 else 0
        
        # Швидкість та Енергія
        v_current = v0 * math.exp(-k_drag * d)
        energy = (params['weight'] / 1000 * v_current**2) / 2
        
        mrad = (y_m * 100) / (d / 10) if d > 0 else 0
        moa = mrad * 3.438
        
        results.append({
            "Дистанція (м)": d,
            "Падіння (см)": round(y_m * 100, 1),
            "MRAD": round(mrad, 2),
            "MOA": round(moa, 2),
            "Вітер (см)": round((wind_drift + derivation) * 100, 1),
            "Швидкість (м/с)": round(v_current, 1),
            "Енергія (Дж)": int(energy)
        })
    return pd.DataFrame(results), v0

# --- СТРУКТУРА ІНТЕРФЕЙСУ ---
st.sidebar.title("🛠️ Налаштування")

with st.sidebar.expander("📦 Вибір набою", expanded=True):
    ammo_choice = st.selectbox("Пресет", list(AMMO_DB.keys()))
    data = AMMO_DB[ammo_choice]
    
    v0_in = st.number_input("V0 (м/с)", value=data['v0'])
    bc_in = st.number_input("BC", value=data['bc'], format="%.3f")
    model_in = st.selectbox("Модель", ["G1", "G7"], index=0 if data['model']=="G1" else 1)
    weight_in = st.number_input("Вага кулі (г)", value=data['weight'])
    t_coeff = st.slider("Термозалежність (м/с на °C)", 0.0, 1.0, 0.2)

with st.sidebar.expander("🌍 Атмосфера та Стрільба"):
    temp = st.slider("Температура (°C)", -20, 45, 15)
    pressure = st.number_input("Тиск (hPa)", value=1013)
    humidity = st.slider("Вологість (%)", 0, 100, 50)
    angle = st.slider("Кут нахилу (°)", -45, 45, 0)
    twist = st.number_input("Твіст (дюйми)", value=10.0)

with st.sidebar.expander("💨 Вітер"):
    w_speed = st.number_input("Швидкість (м/с)", value=0.0)
    w_dir = st.selectbox("Напрямок (год)", list(range(1, 13)), index=2)

with st.sidebar.expander("🎯 Дистанція"):
    zero_dist = st.number_input("Пристрілка (м)", value=100)
    max_dist = st.slider("Макс. дистанція (м)", 100, 1500, 800, step=50)
    sh = st.number_input("Висота прицілу (см)", value=5.0)

# Розрахунок
params = {
    'v0': v0_in, 'bc': bc_in, 'model': model_in, 'weight': weight_in,
    'temp': temp, 'pressure': pressure, 'humidity': humidity,
    'angle': angle, 'twist': twist, 'w_speed': w_speed, 'w_dir': w_dir,
    'zero_dist': zero_dist, 'max_dist': max_dist, 'sh': sh, 't_coeff': t_coeff
}

df, real_v0 = run_simulation(params)

# --- ВІЗУАЛІЗАЦІЯ ---
st.title("🏹 Ballistic Expert Pro v3.0")

# Верхні показники для обраної дистанції (макс)
last_row = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Поправка MRAD", abs(last_row['MRAD']))
c2.metric("Поправка MOA", abs(last_row['MOA']))
c3.metric("Швидкість у цілі", f"{last_row['Швидкість (м/с)']} м/с")
c4.metric("Енергія", f"{last_row['Енергія (Дж)']} Дж")

# Графік
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Падіння (см)'], 
                         name="Траєкторія", line=dict(color='#00ff00', width=3),
                         hovertemplate="Дист: %{x}м<br>Падіння: %{y}см"))
fig.add_hline(y=0, line_dash="dash", line_color="red")
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20),
                  xaxis_title="Дистанція (м)", yaxis_title="Падіння (см)")
st.plotly_chart(fig, use_container_width=True)

# Картка вогню
st.subheader("📋 Картка вогню (Крок 100м)")
show_df = df[df['Дистанція (м)'] % 100 == 0].copy()
st.dataframe(show_df, use_container_width=True, hide_index=True)

# Попередження про звук
if last_row['Швидкість (м/с)'] < 340:
    st.warning(f"⚠️ На дистанції {max_dist}м куля перейшла у дозвуковий режим ({last_row['Швидкість (м/с)']} м/с). Точність може бути нестабільною.")

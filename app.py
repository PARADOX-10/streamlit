import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# Налаштування сторінки
st.set_page_config(page_title="Magelan242 Ballistic Expert Pro v5.5", layout="wide")

# --- БАЗА ДАНИХ НАБОЇВ ---
AMMO_DB = {
    "Військові / Тактичні": {
        "5.45x39 7N6 (PS)": {"v0": 880, "bc": 0.330, "model": "G1", "weight_gr": 53},
        "7.62x39 (123 gr FMJ)": {"v0": 715, "bc": 0.275, "model": "G1", "weight_gr": 123},
        "5.56x45 M855 (SS109)": {"v0": 915, "bc": 0.304, "model": "G1", "weight_gr": 62},
        "7.62x54R (148 gr LPS)": {"v0": 830, "bc": 0.420, "model": "G1", "weight_gr": 148},
        ".308 Win M80 (147 gr)": {"v0": 850, "bc": 0.395, "model": "G1", "weight_gr": 147},
    },
    "Високоточні (Match/Sniper)": {
        "6.5 Creedmoor (140 gr ELD-M)": {"v0": 825, "bc": 0.313, "model": "G7", "weight_gr": 140},
        ".308 Win (175 gr SMK)": {"v0": 790, "bc": 0.243, "model": "G7", "weight_gr": 175},
        ".338 Lapua (250 gr Scenar)": {"v0": 900, "bc": 0.322, "model": "G7", "weight_gr": 250},
        ".375 CheyTac (350 gr)": {"v0": 930, "bc": 0.410, "model": "G7", "weight_gr": 350},
    },
    "Мисливські / Малокаліберні": {
        ".22 LR (40 gr RN)": {"v0": 330, "bc": 0.120, "model": "G1", "weight_gr": 40},
        ".243 Win (95 gr SST)": {"v0": 920, "bc": 0.355, "model": "G1", "weight_gr": 95},
        ".30-06 Spring (180 gr SP)": {"v0": 820, "bc": 0.425, "model": "G1", "weight_gr": 180},
    }
}

# --- МАТЕМАТИЧНЕ ЯДРО ---
def get_air_density(temp, pressure):
    tk = temp + 273.15
    return (pressure * 100) / (287.05 * tk)

def run_simulation(p):
    # Температурна корекція початкової швидкості
    v0 = p['v0'] + (p['temp'] - 15) * p['t_coeff']
    rho = get_air_density(p['temp'], p['pressure'])
    
    # Коефіцієнт опору повітря
    k_drag = 0.5 * rho * (1/p['bc']) * 0.00052
    if p['model'] == "G7": k_drag *= 0.91

    results = []
    g = 9.80665
    angle_rad = math.radians(p['angle'])
    weight_kg = p['weight_gr'] * 0.0000647989

    for d in range(0, p['max_dist'] + 1, 10):
        # Час польоту
        t = d / (v0 * math.exp(-k_drag * d / 2)) if d > 0 else 0
        
        # Вертикальне падіння
        drop = 0.5 * g * (t**2) * math.cos(angle_rad)
        t_zero = p['zero_dist'] / (v0 * math.exp(-k_drag * p['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        
        y_m = -(drop - (drop_zero + p['sh']/100) * (d / p['zero_dist']) + p['sh']/100)
        
        # Вітер та Деривація
        wind_rad = math.radians(p['w_dir'] * 30)
        wind_drift = (p['w_speed'] * math.sin(wind_rad)) * (t - (d/v0)) if d > 0 else 0
        derivation = 0.05 * (p['twist'] / 10) * (d / 100)**2 if d > 0 else 0
        
        # Швидкість та Енергія
        v_curr = v0 * math.exp(-k_drag * d)
        energy = (weight_kg * v_curr**2) / 2
        
        mrad = (y_m * 100) / (d / 10) if d > 0 else 0
        moa = mrad * 3.438
        
        results.append({
            "Дистанція (м)": d,
            "Падіння (см)": round(y_m * 100, 1),
            "MRAD": round(mrad, 2),
            "MOA": round(moa, 2),
            "Вітер+Дер (см)": round((wind_drift + derivation) * 100, 1),
            "Швидкість (м/с)": round(v_curr, 1),
            "Енергія (Дж)": int(energy)
        })
    return pd.DataFrame(results), v0

# --- ПАНЕЛЬ КЕРУВАННЯ ---
st.sidebar.title("🛡️ Налаштування")

mode = st.sidebar.radio("Вибір набою:", ["З бази даних", "Свій набій (Custom)"])

if mode == "З бази даних":
    cat = st.sidebar.selectbox("Категорія", list(AMMO_DB.keys()))
    ammo = st.sidebar.selectbox("Набій", list(AMMO_DB[cat].keys()))
    base = AMMO_DB[cat][ammo]
    
    v0_val = base['v0']
    bc_val = base['bc']
    model_val = base['model']
    weight_val = base['weight_gr']
    display_name = ammo
else:
    display_name = st.sidebar.text_input("Назва набою", "Custom Load")
    v0_val = st.sidebar.number_input("V0 (м/с)", value=800)
    model_val = st.sidebar.selectbox("Модель", ["G1", "G7"])
    bc_val = st.sidebar.number_input("BC", value=0.400, format="%.3f")
    weight_val = st.sidebar.number_input("Вага (gr)", value=150)

with st.sidebar.expander("📝 Додаткові параметри"):
    t_coeff = st.slider("Термозалежність (м/с на °C)", 0.0, 1.0, 0.2)
    sh = st.number_input("Висота прицілу (см)", value=5.0)
    twist = st.number_input("Твіст (дюйми)", value=10.0)

with st.sidebar.expander("🌍 Умови середовища"):
    temp = st.slider("Температура (°C)", -25, 50, 15)
    pressure = st.number_input("Тиск (hPa)", value=1013)
    w_speed = st.number_input("Вітер (м/с)", value=0.0)
    w_dir = st.selectbox("Напрямок вітру (год)", list(range(1, 13)), index=2)
    angle = st.slider("Кут нахилу (°)", -45, 45, 0)
    zero_dist = st.number_input("Пристрілка (м)", value=100)
    max_dist = st.slider("Макс. дистанція (м)", 100, 2000, 1000, step=100)

# Запуск розрахунку
sim_params = {
    'v0': v0_val, 'bc': bc_val, 'model': model_val, 'weight_gr': weight_val,
    'temp': temp, 'pressure': pressure, 'w_speed': w_speed, 'w_dir': w_dir,
    'angle': angle, 'twist': twist, 'zero_dist': zero_dist, 'max_dist': max_dist,
    'sh': sh, 't_coeff': t_coeff
}
df, final_v0 = run_simulation(sim_params)

# --- ВІЗУАЛІЗАЦІЯ ---
st.title(f"🎯 {display_name}")

c1, c2, c3, c4 = st.columns(4)
target_row = df.iloc[-1]
c1.metric("Поправка MRAD", abs(target_row['MRAD']))
c2.metric("V0 коригована", f"{final_v0:.1f} м/с")
c3.metric("Енергія у цілі", f"{target_row['Енергія (Дж)']} Дж")
c4.metric("Швидкість у цілі", f"{target_row['Швидкість (м/с)']} м/с")

# Графік падіння
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Падіння (см)'], 
                         name="Падіння кулі", line=dict(color='#00ff00', width=3)))
fig.add_hline(y=0, line_dash="dash", line_color="red")
fig.update_layout(template="plotly_dark", title="Траєкторія (см)", xaxis_title="Метри", yaxis_title="Сантиметри")
st.plotly_chart(fig, use_container_width=True)

# Картка вогню
st.subheader("📋 Картка вогню")
st.dataframe(df[df['Дистанція (м)'] % 100 == 0], hide_index=True, use_container_width=True)

# Попередження про дозвук
if target_row['Швидкість (м/с)'] < 340:
    st.warning(f"⚠️ Куля перейшла у дозвуковий режим на цій дистанції. Можлива втрата стабільності.")

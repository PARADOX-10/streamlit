import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Налаштування сторінки
st.set_page_config(page_title="Ballistic Expert Ultimate v7.5", layout="wide")

# --- МАКСИМАЛЬНА БАЗА ДАНИХ ---
AMMO_DB = {
    "Військові (Схід)": {
        "5.45x39 7N6 (PS)": {"v0": 880, "bc": 0.330, "model": "G1", "weight_gr": 53},
        "7.62x39 FMJ": {"v0": 715, "bc": 0.275, "model": "G1", "weight_gr": 123},
        "7.62x54R 7N1": {"v0": 830, "bc": 0.411, "model": "G1", "weight_gr": 151},
        "12.7x108 B-32": {"v0": 820, "bc": 1.050, "model": "G1", "weight_gr": 745},
    },
    "Військові (NATO)": {
        "5.56x45 M855 (SS109)": {"v0": 915, "bc": 0.304, "model": "G1", "weight_gr": 62},
        "7.62x51 M118LR": {"v0": 785, "bc": 0.243, "model": "G7", "weight_gr": 175},
        ".50 BMG M2": {"v0": 890, "bc": 0.670, "model": "G1", "weight_gr": 647},
    },
    "Високоточні": {
        "6.5 Creedmoor ELD-M": {"v0": 825, "bc": 0.313, "model": "G7", "weight_gr": 140},
        ".300 Win Mag SMK": {"v0": 890, "bc": 0.533, "model": "G1", "weight_gr": 190},
        ".338 Lapua Scenar": {"v0": 900, "bc": 0.322, "model": "G7", "weight_gr": 250},
        ".375 CheyTac": {"v0": 930, "bc": 0.410, "model": "G7", "weight_gr": 350},
    },
    "Малокаліберні/Пістолетні": {
        ".22 LR Standard": {"v0": 325, "bc": 0.120, "model": "G1", "weight_gr": 40},
        "9x19 Luger": {"v0": 360, "bc": 0.147, "model": "G1", "weight_gr": 115},
    }
}

# --- МАТЕМАТИЧНЕ ЯДРО ---
def get_air_density(temp, pressure):
    tk = temp + 273.15
    return (pressure * 100) / (287.05 * tk)

def run_simulation(p):
    v0 = p['v0'] + (p['temp'] - 15) * p['t_coeff']
    rho = get_air_density(p['temp'], p['pressure'])
    k_drag = 0.5 * rho * (1/p['bc']) * 0.00052
    if p['model'] == "G7": k_drag *= 0.91

    results = []
    g = 9.80665
    weight_kg = p['weight_gr'] * 0.0000647989

    for d in range(0, p['max_dist'] + 1, 10):
        t = d / (v0 * math.exp(-k_drag * d / 2)) if d > 0 else 0
        drop = 0.5 * g * (t**2)
        t_zero = p['zero_dist'] / (v0 * math.exp(-k_drag * p['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        y_m = -(drop - (drop_zero + p['sh']/100) * (d / p['zero_dist']) + p['sh']/100)
        
        wind_rad = math.radians(p['w_dir'] * 30)
        wind_drift = (p['w_speed'] * math.sin(wind_rad)) * (t - (d/v0)) if d > 0 else 0
        
        v_curr = v0 * math.exp(-k_drag * d)
        energy = (weight_kg * v_curr**2) / 2
        
        mrad = (y_m * 100) / (d / 10) if d > 0 else 0
        # Кліки
        clicks = round(mrad / p['click_value'], 1) if d > 0 else 0
        
        results.append({
            "Дистанція (м)": d, "Падіння (см)": round(y_m * 100, 1),
            "MRAD": round(mrad, 2), "Кліки": clicks,
            "Вітер (см)": round(wind_drift * 100, 1),
            "Швидкість (м/с)": round(v_curr, 1), "Енергія (Дж)": int(energy)
        })
    return pd.DataFrame(results), v0

# --- ІНТЕРФЕЙС SIDEBAR ---
st.sidebar.title("🛠️ Налаштування")

tab_ammo, tab_optics, tab_env = st.sidebar.tabs(["📦 Набій", "🔭 Оптика", "🌍 Умови"])

with tab_ammo:
    mode = st.radio("Джерело:", ["База", "Custom"])
    if mode == "База":
        cat = st.selectbox("Категорія", list(AMMO_DB.keys()))
        ammo = st.selectbox("Набій", list(AMMO_DB[cat].keys()))
        base = AMMO_DB[cat][ammo]
        v0_in, bc_in, mod_in, w_in = base['v0'], base['bc'], base['model'], base['weight_gr']
    else:
        v0_in = st.number_input("V0 (м/с)", value=800)
        mod_in = st.selectbox("Модель", ["G1", "G7"])
        bc_in = st.number_input("BC", value=0.400, format="%.3f")
        w_in = st.number_input("Вага (gr)", value=150)
    t_coeff = st.slider("Термозалежність (м/с на °C)", 0.0, 1.0, 0.2)

with tab_optics:
    click_val = st.selectbox("Ціна кліка прицілу", 
                             options=[0.1, 0.05, 0.25], 
                             format_func=lambda x: f"{x} MRAD" if x < 0.2 else "1/4 MOA (0.07 MRAD)")
    if click_val == 0.25: click_val = 0.0727 # конвертація MOA в MRAD для розрахунків
    sh = st.number_input("Висота прицілу (см)", value=5.0)
    zero_dist = st.number_input("Дистанція пристрілки (м)", value=100)

with tab_env:
    temp = st.slider("Температура (°C)", -25, 45, 15)
    press = st.number_input("Тиск (hPa)", value=1013)
    w_speed = st.number_input("Вітер (м/с)", value=0.0)
    w_dir = st.selectbox("Напрямок вітру (год)", list(range(1, 13)), index=2)
    max_dist = st.slider("Макс. дистанція (м)", 100, 1500, 800, step=100)

# Розрахунок
params = {'v0': v0_in, 'bc': bc_in, 'model': mod_in, 'weight_gr': w_in,
          'temp': temp, 'pressure': press, 'w_speed': w_speed, 'w_dir': w_dir,
          'zero_dist': zero_dist, 'max_dist': max_dist, 'sh': sh, 
          't_coeff': t_coeff, 'click_value': click_val}

df, final_v0 = run_simulation(params)

# --- ГОЛОВНИЙ ЕКРАН ---
st.title("🏹 Ballistic Expert Ultimate Pro v7.5")

# Метрики
target = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("КЛІКИ (вертикаль)", abs(target['Кліки']))
c2.metric("Поправка MRAD", abs(target['MRAD']))
c3.metric("Енергія", f"{target['Енергія (Дж)']} Дж")
c4.metric("Швидкість", f"{target['Швидкість (м/с)']} м/с")

# Графіки
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=("Траєкторія (см)", "Енергія кулі (Дж)"))

fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Падіння (см)'], name="Падіння", line=dict(color='lime', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Енергія (Дж)'], name="Енергія", line=dict(color='orange', width=3)), row=2, col=1)

fig.update_layout(height=600, template="plotly_dark", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Таблиця
st.subheader("📋 Робоча таблиця поправок")
def style_table(row):
    return ['background-color: rgba(255, 0, 0, 0.2)' if row['Швидкість (м/с)'] < 340 else ''] * len(row)

st.dataframe(df[df['Дистанція (м)'] % 100 == 0].style.apply(style_table, axis=1), use_container_width=True)

# Кнопка збереження
st.download_button("📥 Завантажити таблицю (CSV)", df.to_csv(index=False), "range_card.csv", "text/csv")

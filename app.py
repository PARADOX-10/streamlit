import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Ballistic Comparator Pro v11.0", layout="wide")

# --- МАТЕМАТИЧНА МОДЕЛЬ ---
def run_simulation(p):
    v0_corr = p['v0'] + (p['temp'] - 15) * p['t_coeff']
    tk = p['temp'] + 273.15
    rho = (p['pressure'] * 100) / (287.05 * tk)
    k_drag = 0.5 * rho * (1/p['bc']) * 0.00052
    if p['model'] == "G7": k_drag *= 0.91

    results = []
    g = 9.80665
    weight_kg = p['weight_gr'] * 0.0000647989

    for d in range(0, p['max_dist'] + 1, 10):
        t = d / (v0_corr * math.exp(-k_drag * d / 2)) if d > 0 else 0
        drop = 0.5 * g * (t**2)
        t_zero = p['zero_dist'] / (v0_corr * math.exp(-k_drag * p['zero_dist'] / 2))
        drop_zero = 0.5 * g * (t_zero**2)
        y_m = -(drop - (drop_zero + p['sh']/100) * (d / p['zero_dist']) + p['sh']/100)
        
        wind_rad = math.radians(p['w_dir'] * 30)
        wind_drift = (p['w_speed'] * math.sin(wind_rad)) * (t - (d/v0_corr)) if d > 0 else 0
        
        v_curr = v0_corr * math.exp(-k_drag * d)
        energy = (weight_kg * v_curr**2) / 2
        
        results.append({
            "Дистанція": d, "Падіння_см": y_m * 100, 
            "Знесення_см": wind_drift * 100, "Швидкість": v_curr, "Енергія": energy
        })
    return pd.DataFrame(results)

# --- SIDEBAR: НАЛАШТУВАННЯ ДВОХ КОНФІГУРАЦІЙ ---
st.sidebar.title("🛠️ Порівняння систем")

def get_params(suffix):
    with st.sidebar.expander(f"⚙️ Конфігурація {suffix}", expanded=True):
        v0 = st.number_input(f"V0 (м/с) {suffix}", value=800.0, key=f"v0_{suffix}")
        bc = st.number_input(f"BC {suffix}", value=0.450 if suffix == "A" else 0.500, format="%.3f", key=f"bc_{suffix}")
        mod = st.selectbox(f"Модель {suffix}", ["G1", "G7"], index=1, key=f"mod_{suffix}")
        w_gr = st.number_input(f"Вага (gr) {suffix}", value=168.0, key=f"w_{suffix}")
        sh = st.number_input(f"Висота прицілу (см) {suffix}", value=5.0, key=f"sh_{suffix}")
    return {'v0': v0, 'bc': bc, 'model': mod, 'weight_gr': w_gr, 'sh': sh}

cfg_a = get_params("A")
cfg_b = get_params("B")

# Спільні умови
with st.sidebar.expander("🌍 Спільні умови середовища"):
    temp = st.slider("Температура (°C)", -20, 45, 15)
    press = st.number_input("Тиск (hPa)", 1013)
    w_spd = st.slider("Вітер (м/с)", 0.0, 15.0, 4.0)
    w_dir = st.slider("Напрямок (год)", 1, 12, 3)
    max_d = st.slider("Макс. дистанція (м)", 100, 1500, 1000, step=100)

# Загальні параметри для обох
common = {'temp': temp, 'pressure': press, 'w_speed': w_spd, 'w_dir': w_dir, 
          'zero_dist': 100, 'max_dist': max_d, 't_coeff': 0.2, 'twist': 10}

# Розрахунок
df_a = run_simulation({**cfg_a, **common})
df_b = run_simulation({**cfg_b, **common})

# --- ОСНОВНА ПАНЕЛЬ ---
st.title("🏹 Ballistic Comparator Pro")

# Порівняльні графіки
fig = make_subplots(rows=2, cols=2, 
                    subplot_titles=("Траєкторія (Падіння, см)", "Знесення вітром (см)", 
                                    "Швидкість (м/с)", "Енергія (Дж)"))

# Падіння
fig.add_trace(go.Scatter(x=df_a['Дистанція'], y=df_a['Падіння_см'], name="Система A", line=dict(color='lime')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_b['Дистанція'], y=df_b['Падіння_см'], name="Система B", line=dict(color='orange', dash='dash')), row=1, col=1)

# Вітер
fig.add_trace(go.Scatter(x=df_a['Дистанція'], y=df_a['Знесення_см'], showlegend=False, line=dict(color='lime')), row=1, col=2)
fig.add_trace(go.Scatter(x=df_b['Дистанція'], y=df_b['Знесення_см'], showlegend=False, line=dict(color='orange', dash='dash')), row=1, col=2)

# Швидкість
fig.add_trace(go.Scatter(x=df_a['Дистанція'], y=df_a['Швидкість'], showlegend=False, line=dict(color='lime')), row=2, col=1)
fig.add_trace(go.Scatter(x=df_b['Дистанція'], y=df_b['Швидкість'], showlegend=False, line=dict(color='orange', dash='dash')), row=2, col=1)
fig.add_hline(y=340, line_dash="dot", line_color="red", row=2, col=1)

# Енергія
fig.add_trace(go.Scatter(x=df_a['Дистанція'], y=df_a['Енергія'], showlegend=False, line=dict(color='lime')), row=2, col=2)
fig.add_trace(go.Scatter(x=df_b['Дистанція'], y=df_b['Енергія'], showlegend=False, line=dict(color='orange', dash='dash')), row=2, col=2)

fig.update_layout(height=800, template="plotly_dark", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Порівняльна таблиця (фінальна дистанція)
st.subheader("🏁 Порівняння на максимальній дистанції")
res_a = df_a.iloc[-1]
res_b = df_b.iloc[-1]

comp_data = {
    "Параметр": ["Падіння (см)", "Вітер (см)", "Швидкість (м/с)", "Енергія (Дж)"],
    "Система A": [res_a['Падіння_см'], res_a['Знесення_см'], res_a['Швидкість'], res_a['Енергія']],
    "Система B": [res_b['Падіння_см'], res_b['Знесення_см'], res_b['Швидкість'], res_b['Енергія']],
    "Різниця": [res_a['Падіння_см']-res_b['Падіння_см'], res_a['Знесення_см']-res_b['Знесення_см'], 
                res_a['Швидкість']-res_b['Швидкість'], res_a['Енергія']-res_b['Енергія']]
}
st.table(pd.DataFrame(comp_data))

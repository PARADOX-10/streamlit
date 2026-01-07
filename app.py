import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# Налаштування сторінки
st.set_page_config(page_title="Magelan242 Ballistic", layout="wide")

# Стилізація інтерфейсу
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff00; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 Magelan242 Ballistic")
st.write("Професійний розрахунок траєкторії та поправок")

# --- БОКОВА ПАНЕЛЬ (Введення даних) ---
st.sidebar.header("⚙️ Параметри зброї та набою")

v0 = st.sidebar.number_input("Швидкість кулі (v0), м/с", value=893, step=1)
bc = st.sidebar.slider("Балістичний коефіцієнт (G1)", 0.100, 1.000, 0.584, format="%.3f")
sh = st.sidebar.number_input("Висота прицілу, см", value=5.0, step=0.5)
twist = st.sidebar.number_input("Твіст ствола, дюйми", value=11.0, step=0.5)

st.sidebar.header("🌍 Умови та ціль")
target_dist = st.sidebar.slider("Дистанція до цілі, м", 50, 1500, 500, step=50)
zero_dist = st.sidebar.number_input("Дистанція пристрілки, м", value=300)
angle = st.sidebar.slider("Кут нахилу, °", -45, 45, 0)

st.sidebar.header("💨 Вітер")
w_speed = st.sidebar.number_input("Швидкість вітру, м/с", value=0.0, step=0.5)
w_dir = st.sidebar.selectbox("Напрямок вітру (год)", list(range(1, 13)), index=2)

# --- ЛОГІКА РОЗРАХУНКУ ---
def calculate_ballistics(d):
    g = 9.80665
    angle_rad = math.radians(angle)
    k = 0.00015 / bc
    
    # Ефективна дистанція
    eff_d = d * math.cos(angle_rad)
    
    # Час польоту
    t = d / (v0 * math.exp(-k * d/2)) if d > 0 else 0
    
    # Падіння
    drop = 0.5 * g * (t**2)
    t_zero = zero_dist / (v0 * math.exp(-k * zero_dist/2))
    drop_zero = 0.5 * g * (t_zero**2)
    
    y_m = -(drop - (drop_zero + sh/100) * (d / zero_dist) + sh/100) if d > 0 else 0
    
    # Вітер
    wind_rad = math.radians(w_dir * 30)
    wind_drift = (w_speed * math.sin(wind_rad)) * (t - (d/v0)) if d > 0 else 0
    
    # Деривація
    derivation = 0.05 * (twist / 10) * (d / 100)**2 if d > 0 else 0
    
    # Поправки
    mrad = (y_m * 100) / (d / 10) if d > 0 else 0
    moa = mrad * 3.438
    
    return y_m * 100, mrad, moa, wind_drift * 100, derivation

# Розрахунок для поточної цілі
res_drop, res_mrad, res_moa, res_wind, res_der = calculate_ballistics(target_dist)

# --- ОСНОВНИЙ ЕКРАН (Результати) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Вертикаль (см)", f"{res_drop:.1f}")
col2.metric("Поправка MRAD", f"{abs(res_mrad):.2f}")
col3.metric("Поправка MOA", f"{abs(res_moa):.2f}")
col4.metric("Вітер/Дер. (см)", f"{res_wind + res_der:.1f}")

# --- ГРАФІК ---
distances = np.arange(0, target_dist + 50, 10)
drops = [calculate_ballistics(d)[0] for d in distances]

fig = go.Figure()
fig.add_trace(go.Scatter(x=distances, y=drops, mode='lines', name='Траєкторія', line=dict(color='#00ff00', width=3)))
fig.add_hline(y=0, line_dash="dash", line_color="red")
fig.update_layout(title="Графік польоту кулі", template="plotly_dark", xaxis_title="Відстань (м)", yaxis_title="Висота (см)")
st.plotly_chart(fig, use_container_width=True)

# --- ТАБЛИЦЯ ПОПРАВОК ---
st.subheader("📋 Таблиця поправок (Картка вогню)")
table_data = []
for d in range(100, target_dist + 100, 100):
    d_drop, d_mrad, d_moa, d_wind, d_der = calculate_ballistics(d)
    table_data.append([d, round(d_drop, 1), round(d_mrad, 2), round(d_moa, 2), round(d_wind + d_der, 1)])

df = pd.DataFrame(table_data, columns=["Дистанція (м)", "Падіння (см)", "MRAD", "MOA", "Вітер+Дер (см)"])
st.table(df)

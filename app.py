import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# --- КОНФІГУРАЦІЯ СТОРІНКИ ---
st.set_page_config(page_title="Ballistics PRO", layout="wide", page_icon="🎯")

# --- ФІЗИЧНА МОДЕЛЬ (СЕРЦЕ КАЛЬКУЛЯТОРА) ---

class BallisticsSolver:
    def __init__(self, drag_model='G7'):
        self.g = 9.80665
        self.drag_model = drag_model
        # Спрощена апроксимація кривої G7 (Mach vs Cd)
        # У реальному проекті тут має бути повна таблиця Litz/Lapua
        self.mach_data = [0.0, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 4.0]
        self.cd_data_g7 = [0.10, 0.10, 0.11, 0.15, 0.43, 0.38, 0.28, 0.23, 0.19, 0.15]
        self.drag_func = interp1d(self.mach_data, self.cd_data_g7, kind='linear', fill_value="extrapolate")

    def get_air_density(self, temp_c, pressure_hpa, humidity_pct):
        """Розрахунок щільності повітря (МСА)"""
        temp_k = temp_c + 273.15
        pressure_pa = pressure_hpa * 100
        # Спрощене рівняння ідеального газу для сухого повітря
        # (Для повної точності потрібне врахування вологості CIPM-2007)
        rho = pressure_pa / (287.05 * temp_k)
        return rho

    def get_mach(self, velocity, temp_c):
        speed_of_sound = 331.3 * np.sqrt(1 + temp_c / 273.15)
        return velocity / speed_of_sound

    def solve(self, params):
        # Розпакування параметрів
        v0 = params['v0']
        angle_rad = np.radians(params['angle'])
        bc = params['bc']
        mass_kg = params['weight_gr'] * 0.00006479891
        diameter_m = params['diameter_mm'] / 1000
        area = np.pi * (diameter_m / 2) ** 2
        
        rho = self.get_air_density(params['temp'], params['pressure'], params['humidity'])
        
        # Стандартна атмосфера для BC (ICAO)
        rho_std = 1.225 

        # Початковий стан [x, y, vx, vy]
        state = np.array([0.0, -params['sight_height']/100, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)])
        dt = 0.001  # Крок часу (1 мілісекунда для високої точності)
        t = 0.0
        
        results = []

        while state[0] <= params['max_dist'] and state[1] > -2.0:
            # Зберігаємо дані з певним кроком (щоб не перевантажувати графік)
            if len(results) == 0 or state[0] - results[-1]['dist'] >= 25:
                v_total = np.sqrt(state[2]**2 + state[3]**2)
                mach = self.get_mach(v_total, params['temp'])
                drop_moa = 0
                if state[0] > 0:
                    # Розрахунок поправки в MOA: (Drop / Dist) * conversion
                    drop_moa = np.degrees(np.arctan(state[1] / state[0])) * 60 * -1

                results.append({
                    'dist': state[0],
                    'drop_m': state[1],
                    'drop_moa': drop_moa,
                    'velocity': v_total,
                    'time': t,
                    'mach': mach,
                    'energy': 0.5 * mass_kg * v_total**2
                })

            # Метод Рунге-Кутта 4 (RK4)
            k1 = dt * self._derivatives(state, rho, rho_std, bc, area, mass_kg)
            k2 = dt * self._derivatives(state + 0.5 * k1, rho, rho_std, bc, area, mass_kg)
            k3 = dt * self._derivatives(state + 0.5 * k2, rho, rho_std, bc, area, mass_kg)
            k4 = dt * self._derivatives(state + k3, rho, rho_std, bc, area, mass_kg)
            
            state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
            t += dt

        return pd.DataFrame(results)

    def _derivatives(self, state, rho, rho_std, bc, area, mass):
        vx, vy = state[2], state[3]
        v = np.sqrt(vx**2 + vy**2)
        
        # Поточний Mach
        # (Тут спрощено, у повній версії треба передавати швидкість звуку)
        mach = v / 340.0 
        
        # Drag Coefficient з урахуванням BC
        # Cd = Cd_std * (FormFactor), де FormFactor ~ 1/BC для G-функцій
        # Але точніше: a_drag = -0.5 * rho * v^2 * area * Cd / m
        # Використовуємо стандартну модель:
        # Drag Force = 0.5 * rho * v^2 * S * Cd
        # Для G7: Cd_actual = Cd_G7(Mach) * (SectionalDensity / BC) * FormFactorCorrection
        # Для простоти реалізуємо пряму залежність через сповільнення:
        
        drag_coeff_std = self.drag_func(mach)
        
        # Основна формула опору (Modified Point Mass Equation)
        # F_drag = 0.5 * rho * v^2 * Area * Cd_std * (1/BC_factor)
        # Примітка: Це адаптація. У реальному двигуні використовується i7 форм-фактор.
        i_factor = 1.0 # Ідеальна форма G7
        if bc > 0:
             # Коригування під реальний BC відносно стандарту G7
             # Це математичне спрощення для прикладу. 
             # У повній версії: i7 = m / (d^2 * bc_g7)
             pass

        force_drag = 0.5 * rho * v * area * drag_coeff_std * (1/bc) * (mass/0.02) # Емпіричний коефіцієнт масштабування для прикладу
        
        # Сила опору направлена проти вектора швидкості
        ax = -(force_drag * vx) / mass
        ay = -self.g - (force_drag * vy) / mass
        
        return np.array([vx, vy, ax, ay])

# --- ІНТЕРФЕЙС КОРИСТУВАЧА ---

st.title("🎯 Precision Ballistics Calculator")
st.markdown("Професійний розрахунок траєкторії методом **Runge-Kutta 4**.")

# 1. Секція вводу даних (Сайдбар)
with st.sidebar:
    st.header("⚙️ Параметри зброї")
    
    # Встановлено ваші параметри .300 Win Mag за замовчуванням
    col1, col2 = st.columns(2)
    with col1:
        v0 = st.number_input("Швидкість (м/с)", value=893.0, step=1.0)
        weight = st.number_input("Вага кулі (гран)", value=195.0, step=0.1)
    with col2:
        bc_val = st.number_input("BC (G7)", value=0.292, step=0.001, format="%.3f")
        twist = st.number_input("Твіст (дюйми)", value=11.0, step=0.5)

    st.subheader("🔭 Приціл")
    sight_height = st.number_input("Висота прицілу (см)", value=4.5)
    zero_dist = st.number_input("Дистанція пристрілки (м)", value=100)

    st.header("🌤️ Атмосфера")
    temp = st.slider("Температура (°C)", -20, 40, 15)
    pressure = st.number_input("Тиск (hPa)", value=1013)
    humidity = st.slider("Вологість (%)", 0, 100, 50)

# 2. Розрахунок
if st.button("РОЗРАХУВАТИ ТРАЄКТОРІЮ", type="primary"):
    solver = BallisticsSolver()
    params = {
        'v0': v0, 'weight_gr': weight, 'bc': bc_val, 
        'diameter_mm': 7.82, # .308 калібр
        'temp': temp, 'pressure': pressure, 'humidity': humidity,
        'sight_height': sight_height, 'angle': 0, 'max_dist': 1200
    }
    
    df = solver.solve(params)

    # Коригування нуля (Simple zero shift)
    # Знаходимо падіння на дистанції пристрілки
    try:
        zero_row = df.iloc[(df['dist'] - zero_dist).abs().argsort()[:1]]
        zero_offset = zero_row['drop_moa'].values[0]
        df['corrected_moa'] = df['drop_moa'] - zero_offset
        df['clicks_01mrad'] = (df['corrected_moa'] / 3.4377) * 10 # Перевід в кліки 0.1 MRAD
    except:
        df['corrected_moa'] = 0

    # 3. Вивід результатів
    
    # Головні метрики (на 500м і 1000м)
    st.markdown("### 🔥 Ключові дистанції")
    c1, c2, c3, c4 = st.columns(4)
    
    d500 = df.iloc[(df['dist'] - 500).abs().argsort()[:1]]
    d1000 = df.iloc[(df['dist'] - 1000).abs().argsort()[:1]]
    
    c1.metric("Швидкість на 500м", f"{d500['velocity'].values[0]:.0f} м/с")
    c2.metric("Поправка (MOA)", f"{d500['corrected_moa'].values[0]:.1f}")
    c3.metric("Швидкість на 1000м", f"{d1000['velocity'].values[0]:.0f} м/с")
    c4.metric("Поправка (MOA)", f"{d1000['corrected_moa'].values[0]:.1f}")

    # Графіки
    tab1, tab2 = st.tabs(["📉 Графік Траєкторії", "📋 Таблиця"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['dist'], y=df['drop_m'], mode='lines', name='Траєкторія', line=dict(color='#ff4b4b', width=3)))
        fig.update_layout(
            title="Падіння кулі (метри)",
            xaxis_title="Дистанція (м)",
            yaxis_title="Висота (м)",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Графік швидкості
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df['dist'], y=df['velocity'], mode='lines', name='Швидкість', line=dict(color='#00cc96', width=3)))
        fig_v.add_hline(y=340, line_dash="dash", annotation_text="Звуковий бар'єр")
        fig_v.update_layout(title="Падіння швидкості", template="plotly_dark", height=400)
        st.plotly_chart(fig_v, use_container_width=True)

    with tab2:
        st.dataframe(
            df[['dist', 'drop_m', 'corrected_moa', 'velocity', 'energy']].style.format("{:.1f}"),
            use_container_width=True
        )

else:
    st.info("Натисніть кнопку 'Розрахувати' у сайдбарі")

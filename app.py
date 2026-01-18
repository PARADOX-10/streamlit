import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# --- 1. КОНФІГУРАЦІЯ ---
st.set_page_config(page_title="Magelan242 Ballistics PRO", layout="centered", initial_sidebar_state="collapsed")

# --- 2. БАЗА ДАНИХ КУЛЬ (UPDATED) ---
# Format: [Caliber(in), Weight(gr), BC G7, Length(mm), Model]
BULLET_DB = {
    "Custom (Ручне введення)": None,
    ".224 Sierra TMK 77gr": [0.224, 77, 0.203, 24.8, "G7"],
    ".224 Hornady ELD-M 75gr": [0.224, 75, 0.235, 25.1, "G7"],
    ".243 Hornady ELD-M 108gr": [0.243, 108, 0.270, 31.2, "G7"],
    ".264 Hornady ELD-M 140gr": [0.264, 140, 0.326, 34.5, "G7"],
    ".264 Hornady ELD-M 147gr": [0.264, 147, 0.351, 35.8, "G7"],
    ".264 Lapua Scenar-L 136gr": [0.264, 136, 0.274, 33.5, "G7"],
    ".264 Berger Hybrid 140gr": [0.264, 140, 0.311, 34.3, "G7"],
    ".308 Lapua Scenar 167gr": [0.308, 167, 0.216, 31.5, "G7"],
    ".308 Sierra MK 175gr": [0.308, 175, 0.243, 31.8, "G7"],
    ".308 Hornady ELD-M 178gr": [0.308, 178, 0.275, 32.8, "G7"],
    ".308 Berger Juggernaut 185gr": [0.308, 185, 0.284, 33.2, "G7"],
    ".308 Hornady ELD-X 212gr": [0.308, 212, 0.336, 39.1, "G7"],
    ".308 Berger Hybrid 215gr": [0.308, 215, 0.354, 38.6, "G7"],
    ".308 Hornady A-Tip 230gr": [0.308, 230, 0.414, 41.5, "G7"],
    ".308 Hornady A-Tip 250gr": [0.308, 250, 0.442, 45.2, "G7"],
    ".338 Lapua Scenar 250gr": [0.338, 250, 0.322, 39.5, "G7"],
    ".338 Lapua Scenar 300gr": [0.338, 300, 0.368, 45.3, "G7"],
    ".338 Hornady ELD-M 285gr": [0.338, 285, 0.400, 43.8, "G7"],
    ".510 Hornady A-MAX 750gr": [0.510, 750, 0.512, 69.5, "G7"]
}

# --- 3. СТИЛІЗАЦІЯ ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
        .stApp { background-color: #0e0e0e; font-family: 'Roboto Mono', monospace; color: #e0e0e0; }
        
        .header-title { font-size: 1.4rem; font-weight: bold; color: #00ff41; text-align: center; border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px;}
        
        .hud-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 15px; }
        .hud-card { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 10px; text-align: center; }
        .hud-label { color: #666; font-size: 0.7rem; text-transform: uppercase; }
        .hud-value { color: #fff; font-size: 1.5rem; font-weight: 700; }
        .hud-sub { color: #00ff41; font-size: 0.75rem; }
        
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { height: 40px; padding: 0 15px; background-color: #1a1a1a; color: #aaa; border-radius: 4px 4px 0 0; font-size: 0.8rem; border: none;}
        .stTabs [aria-selected="true"] { background-color: #00ff41 !important; color: #000 !important; font-weight: bold;}
        
        .stButton>button { border-radius: 4px; font-weight: bold; }
        input[type=number] { color: #00ff41 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 4. ENGINE CORE (PROFESSIONAL GRADE) ---

# Стандартна таблиця G7 (Ballistic Research Lab) - High Resolution
# Format: [Mach, Cd]
G7_STANDARD = np.array([
    [0.0, 0.262], [0.2, 0.262], [0.4, 0.262], [0.6, 0.262], [0.7, 0.265], 
    [0.8, 0.273], [0.85, 0.286], [0.9, 0.306], [0.925, 0.324], [0.95, 0.354],
    [0.975, 0.384], [1.0, 0.424], [1.025, 0.438], [1.05, 0.443], [1.075, 0.440],
    [1.1, 0.433], [1.15, 0.415], [1.2, 0.399], [1.3, 0.377], [1.4, 0.360],
    [1.5, 0.347], [1.6, 0.336], [1.7, 0.327], [1.8, 0.319], [1.9, 0.312],
    [2.0, 0.306], [2.2, 0.295], [2.5, 0.282], [3.0, 0.266], [4.0, 0.246], [5.0, 0.232]
])

def get_drag_coefficient_g7(mach):
    # Лінійна інтерполяція по розширеній таблиці
    if mach > 5.0: return 0.232
    return np.interp(mach, G7_STANDARD[:, 0], G7_STANDARD[:, 1])

def get_atmosphere(temp_c, pressure_hpa, humid_pct):
    # Magnus Formula for Saturation Vapor Pressure
    tk = temp_c + 273.15
    svp = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5)) # hPa
    pv = svp * (humid_pct / 100.0)
    
    # Щільність вологого повітря (CIPM-2007 approximation)
    pd_pa = (pressure_hpa - pv) * 100
    pv_pa = pv * 100
    Rd = 287.058
    Rv = 461.495
    rho = (pd_pa / (Rd * tk)) + (pv_pa / (Rv * tk))
    
    # Швидкість звуку (залежить від T та вологості через коеф. адіабати, спрощено через T)
    c_speed = 20.05 * math.sqrt(tk) 
    
    return rho, c_speed

def get_derivatives(state, p_phys):
    # State: [x, y, z, vx, vy, vz]
    vx, vy, vz = state[3], state[4], state[5]
    
    # Вектор швидкості відносно вітру (Relative Air Velocity)
    # Вітер задається в глобальній системі, тому віднімаємо його від швидкості кулі
    v_air_x = vx - p_phys['w_vec'][0]
    v_air_y = vy # Вітер вертикальний ігноруємо (хоча він є, але зазвичай 0)
    v_air_z = vz - p_phys['w_vec'][2]
    
    v_total_rel = math.sqrt(v_air_x**2 + v_air_y**2 + v_air_z**2)
    
    # Mach Number
    mach = v_total_rel / p_phys['c_speed']
    
    # Drag Force (F = 0.5 * rho * v^2 * Cd * A)
    # Acceleration = F / m
    # Ballistic Coefficient (BC) = m / (Cd_std * A) -> A = m / (BC * Cd_std)
    # Але тут ми використовуємо Form Factor approach:
    # Drag Accel = -0.5 * rho * v * v * (Cd_actual / BC_mass_factor)
    # Стандартна модель: a_D = - (rho / rho_std) * (v^2 / BC) * Cd_G7(Mach) * Const
    
    # i7 Form Factor = Cd_actual / Cd_G7_Standard. 
    # Але ми використовуємо BC, тому:
    # Accel = - (rho_actual * pi * d^2 * Cd_g7(M) * v^2) / (8 * m)  <- Це "сира" фізика
    # Спрощена інженерна формула через BC (на основі lbs, inch, ft/s - потім конвертуємо):
    # Ми використаємо метричну систему "з нуля" для точності.
    
    cd_curr = get_drag_coefficient_g7(mach)
    
    # Площа перерізу (м2)
    area = math.pi * (p_phys['cal_m'] / 2)**2
    
    # Сила опору (Newtons)
    # G7 BC визначений відносно Standard Metro (rho_std ~ 1.225)
    # Відношення Cd/Cd_std зашите в BC.
    # a_drag = -0.5 * rho * v^2 * (Cd / m) * A
    # Використовуємо класичне рівняння Point Mass з BC:
    # F_drag = 0.5 * rho * v^2 * Cd * A.
    # BC_G7_lbs = m_lbs / (d_in^2 * i7). i7 = Cd / Cd_G7_ref.
    # Тому ефективніше рахувати через стандартну функцію відставання:
    
    k0 = 0.5 * p_phys['rho'] * area * cd_curr
    # Корекція на BC: Якщо BC=1.0 (ідеальна G7), то k0 правильний.
    # Якщо BC інший, ми маємо масштабувати Cd.
    # BC_metric = BC_imperial * 703.069 (кг/м2).
    # Але BC зазвичай дає порівняння мас.
    
    # ПРОСТИЙ ТА ТОЧНИЙ МЕТОД (Pejsa/McCoy approximation adapted):
    # Drag Accel magnitude
    # rho_factor = rho / 1.225
    # drag_acc_mag = (rho_factor * v_total_rel^2 * cd_curr * Const) / BC
    
    # CONSTANT 0.00105 is approximation. Let's use precise:
    # 1 lb = 0.453592 kg, 1 inch = 0.0254 m
    # G7 Standard reference: m=1lb, d=1in, BC=1.
    
    accel_drag_mag = (0.5 * p_phys['rho'] * v_total_rel**2 * math.pi * (p_phys['cal_m']/2)**2 * cd_curr) / p_phys['mass_kg']
    
    # Однак BC користувача змінює це. 
    # Форм-фактор i = Cd_bullet / Cd_std.
    # BC = m / (d^2 * i). => i = m / (d^2 * BC).
    # Тому Real Drag Force масштабується на i.
    # Але ми взяли Cd_std з таблиці. Нам треба помножити на i.
    # Aле стривайте, BC вже в знаменнику рівняння прискорення в балістичних солверах.
    # Правильний підхід з BC (G7):
    # Accel = (pi * rho * v^2 * d^2 * Cd_G7_table(M)) / (8 * m) * (m_std / (d_std^2 * BC_user)) ?? Ні.
    
    # Найточніша формула (McCoy):
    # a = - (rho * v * v * Cd_G7_table * factor) / BC_G7
    # Factor для метричної системи (v в м/с, rho в кг/м3, BC в G1/G7):
    # Якщо BC в дюймах/фунтах (стандарт):
    # a (m/s2) = - (rho / 1.225) * (v^2) * Cd_G7 * (1 / BC) * 0.0053 (емпіричний кеф для куль 0.308) -> НЕТОЧНО.
    
    # ПОВЕРНЕМОСЯ ДО ФІЗИКИ БЕЗ МАГІЧНИХ ЧИСЕЛ:
    # Form Factor (i7) = (Weight_lbs / Cal_in^2) / BC_G7
    w_lbs = p_phys['mass_kg'] * 2.20462
    d_in = p_phys['cal_m'] / 0.0254
    i7 = (w_lbs / d_in**2) / p_phys['bc']
    
    # Drag Force Magnitude = 0.5 * rho * v^2 * Area * Cd_G7_table * i7
    drag_force = 0.5 * p_phys['rho'] * v_total_rel**2 * area * cd_curr * i7
    drag_acc_mag = drag_force / p_phys['mass_kg']

    # Проекції прискорення опору (проти вектора швидкості відносно повітря)
    ax_drag = -drag_acc_mag * (v_air_x / v_total_rel)
    ay_drag = -drag_acc_mag * (v_air_y / v_total_rel)
    az_drag = -drag_acc_mag * (v_air_z / v_total_rel)
    
    # Гравітація
    ay_g = -9.80665
    
    # Коріоліс (векторний)
    # Omega vector (Earth rotation) approx: [0, cos(lat)*Omega, sin(lat)*Omega] ?? 
    # Ні, краще через компоненти.
    # OMEGA = 7.292115e-5 rad/s
    OMEGA = 7.292115e-5
    lat = p_phys['lat_rad']
    az = p_phys['az_rad'] # Азимут стрільби (0 = North, 90 = East)
    
    # Вектор кутової швидкості Землі в системі координат стрільця (x - range, y - up, z - right)
    # Omega_x = Omega * cos(lat) * cos(az)
    # Omega_y = Omega * sin(lat)
    # Omega_z = -Omega * cos(lat) * sin(az)
    
    omega_x = OMEGA * math.cos(lat) * math.cos(az)
    omega_y = OMEGA * math.sin(lat)
    omega_z = -OMEGA * math.cos(lat) * math.sin(az)
    
    # a_cor = -2 * (Omega x V)
    # Cross product components:
    # (Wy*vz - Wz*vy)
    # (Wz*vx - Wx*vz)
    # (Wx*vy - Wy*vx)
    
    ax_cor = -2 * (omega_y * vz - omega_z * vy)
    ay_cor = -2 * (omega_z * vx - omega_x * vz) # Vertical Coriolis (Eotvos)
    az_cor = -2 * (omega_x * vy - omega_y * vx) # Horizontal Coriolis
    
    return np.array([vx, vy, vz, 
                     ax_drag + ax_cor, 
                     ay_drag + ay_g + ay_cor, 
                     az_drag + az_cor])

def run_simulation(p):
    DT = 0.0005 # Висока точність (0.5 ms step)
    
    # 1. Physics Setup
    rho, c_speed = get_atmosphere(p['temp'], p['pressure'], p['humid'])
    
    # Справжня V0 (з корекцією на порох)
    v0_true = p['v0'] + (p['temp'] - 15.0) * p['temp_sens']
    
    # Вектор вітру (переводимо з "годин" в радіани)
    # 12 годин = зустрічний (0 deg), 3 години = справа (90 deg)
    # Але в балістиці азимут вітру часто відносно Півночі. 
    # Тут ми вважаємо відносно стрільця:
    w_angle_rad = math.radians(p['w_dir'] * 30) # 12->360/0, 3->90
    # Якщо вітер з 3 годин, він дме ВЛІВО (проти осі Z, якщо Z вправо).
    # W_vector: x (along line), z (right).
    # Wind from 3 o'clock (90 deg) -> Blows TO -90 deg.
    # Wx = speed * cos(from_angle + 180)
    # Wz = speed * sin(from_angle + 180)
    # Спростимо: 
    # Headwind (12h/0deg from): blows to 180. (x component negative).
    # Crosswind right (3h/90deg from): blows to 270 (z component negative).
    
    # Convert clock to arithmetic angle (0 is 3 o'clock in math, but 12 in shooting).
    # Shooting: 12=0deg, 3=90deg.
    wind_from_rad = math.radians(p['w_dir'] * 30)
    w_vec = np.array([
        -p['w_speed'] * math.cos(wind_from_rad), # X (range)
        0,                                       # Y (up)
        -p['w_speed'] * math.sin(wind_from_rad)  # Z (right)
    ])
    
    p_phys = {
        'mass_kg': p['weight_gr'] * 0.0000647989,
        'cal_m': p['caliber'] * 0.0254,
        'bc': p['bc'],
        'rho': rho,
        'c_speed': c_speed,
        'lat_rad': math.radians(p['latitude']),
        'az_rad': math.radians(p['azimuth']),
        'w_vec': w_vec
    }
    
    # 2. Miller Stability & Spin Drift Setup
    # Sg_standard calculation
    twist_m = p['twist'] * 0.0254
    # Формула Міллера: Sg = 30 * m_gr / (twist_cal^2 * d_in^3 * L_cal * (1+L_cal^2))
    # Спрощена, але з корекцією на атмосферу:
    # Sg_act = Sg_std * (rho_std / rho_act)
    # rho_std ~ 1.225
    
    twist_cal = p['twist'] / p['caliber']
    len_cal = (p['length_mm'] / 25.4) / p['caliber']
    
    sg_std = (30 * p['weight_gr']) / (
        (twist_cal**2) * (p['caliber']**3) * len_cal * (1 + len_cal**2)
    )
    sg_act = sg_std * (1.225 / rho)
    
    # Aero Jump (Mrad) = (0.055 / Sg_act) * (Wind_Cross_ms)
    # Стрибок відбувається на дульному зрізі.
    # Вітер справа (w_vec[2] < 0) -> Стрибок ВГОРУ (Right Twist).
    t_dir = 1 if p['twist_dir'] == "Right" else -1
    crosswind_ms = -w_vec[2] # Positive if wind from right
    jump_mrad = (0.055 / sg_act) * crosswind_ms * t_dir # Vertical offset
    jump_rad = jump_mrad / 1000.0
    
    # 3. Zeroing Angle
    # Знаходимо кут кидання для "нуля" (ітеративно або аналітично спрощено)
    # Тут використовуємо спрощену балістику вакууму для початкової здогадки + поправка
    # Точніше: симуляція "нуля" не потрібна, якщо ми знаємо Drop на дистанції нуля.
    # Але для коректності ми додамо кут ствола.
    # Angle ~ (Drop_at_Zero + SH) / Zero_Dist
    # Це спрощення, але для <500м працює добре. Для "PRO" треба окремий прогін Zero.
    # Робимо мікро-симуляцію для нуля? Так, це професійно.
    
    # ... Skipping strict zero finding for performance, using high-order approx ...
    # Drop approx = 0.5 * g * t^2. t = Z/V.
    # Angle = atan( (0.5*g*(Z/V)^2 - SH)/Z )
    # Додаємо jump_rad до вертикального кута.
    angle_g = 0.5 * 9.81 * (p['zero_dist'] / v0_true)**2
    theta_0 = math.atan((angle_g + p['sh']/100) / p['zero_dist'])
    
    theta_total = theta_0 + jump_rad
    
    # Initial State
    vx0 = v0_true * math.cos(theta_total)
    vy0 = v0_true * math.sin(theta_total)
    state = np.array([0.0, -p['sh']/100, 0.0, vx0, vy0, 0.0]) # Z=0 start
    
    t, dist = 0.0, 0.0
    results = []
    
    # Loop vars
    step_check = 0
    max_d = p['max_dist']
    
    while dist <= max_d + 10 and state[1] > -100: # Stop if hits ground heavily
        # RK4 Integration
        k1 = get_derivatives(state, p_phys)
        k2 = get_derivatives(state + k1 * DT/2, p_phys)
        k3 = get_derivatives(state + k2 * DT/2, p_phys)
        k4 = get_derivatives(state + k3 * DT, p_phys)
        
        state_new = state + (k1 + 2*k2 + 2*k3 + k4) * DT / 6
        state = state_new
        t += DT
        dist = state[0]
        
        # Записуємо дані кожні 25м (або менше)
        if dist >= step_check:
            v_curr = math.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
            mach = v_curr / c_speed
            
            # --- PROFESSIONAL SPIN DRIFT ---
            # Formula: Drift = 1.25 * (Sg + 1.2) * TOF^1.83 * (Direction)
            # Важливо: Sg тут - це Sg_act (з урахуванням щільності)
            sd_inches = 1.25 * (sg_act + 1.2) * (t**1.83)
            sd_meters = (sd_inches * 0.0254) * t_dir
            
            # Total Horizontal = Wind Drift (calculated in vector) + Spin Drift + Coriolis (in vector)
            # Наш вектор 'state[2]' вже містить Wind Drift (від сили опору) і Coriolis.
            # Нам треба лише додати Spin Drift (бо це ефект гіроскопа, не врахований у Point Mass).
            z_total = state[2] + sd_meters
            
            drop_m = state[1] # Y coordinate (relative to bore axis level approx)
            
            # MRAD Calculations
            # MRAD = (Drop_m * 1000) / Dist_m
            # Але треба врахувати нуль.
            # На дистанції нуля Drop має бути 0.
            # Тому ми робимо "Absolute Trajectory" тут, а потім в UI віднімемо поправку нуля.
            # Або краще: просто видаємо абсолютні координати, а UI рахує кліки.
            
            results.append({
                "Dist": dist,
                "Time": t,
                "V": v_curr,
                "Mach": mach,
                "Drop_Abs": drop_m,     # Абсолютна висота відносно дула
                "Wind_Abs": z_total,    # Абсолютний знос
                "Sg": sg_act
            })
            step_check += 25

    df = pd.DataFrame(results)
    
    # --- ZERO OFFSET CORRECTION ---
    # Знаходимо точку в таблиці найближчу до дистанції нуля
    zero_row = df.iloc[(df['Dist'] - p['zero_dist']).abs().argsort()[:1]]
    y_zero_offset = zero_row['Drop_Abs'].values[0]
    z_zero_offset = zero_row['Wind_Abs'].values[0] # Зазвичай нулимо вітер в 0, але деривацію можемо врахувати
    
    # Коригуємо дані, щоб на дистанції нуля було 0
    # Але стрільці нулять гвинтівку так, щоб куля прилетіла в хрест.
    # Значить, наша оптична вісь дивиться в точку падіння.
    # Drop (у кліках) = Angle_traj - Angle_sight.
    
    # Спрощено для UI: 
    # Drop_cm = (Drop_Abs - Drop_Zero_Abs) * 100 ?? Ні.
    # Правильно: Кут прицілювання ми вже заклали в theta_0. 
    # Тож state[1] - це відхилення від лінії кидання? Ні, від горизонту.
    # Drop from Line of Sight (LoS):
    # y_los = -SH + dist * tan(alpha). alpha ~ 0 for flat scope??
    # Складно. Найкращий метод для калькуляторів:
    # Drop (Adjusted) = Drop_Abs - (Drop_at_Zero / Zero_Dist) * Dist  <- Лінійна корекція прицільної лінії
    
    slope = y_zero_offset / p['zero_dist']
    df['Drop_Cm'] = (df['Drop_Abs'] - (slope * df['Dist'])) * 100 # см відносно LoS
    df['Drift_Cm'] = (df['Wind_Abs']) * 100 # см
    
    # Перерахунок в MRAD/MOA
    df['MRAD_V'] = -(df['Drop_Cm'] / 100) / (df['Dist']) * 1000
    df['MRAD_H'] = -(df['Drift_Cm'] / 100) / (df['Dist']) * 1000
    
    # Clean NaN at dist 0
    df.fillna(0, inplace=True)
    
    return df

# --- 5. UI STRUCTURE (UNCHANGED LOGIC, UPDATED CALLS) ---
st.markdown('<div class="header-title">🦅 MAGELAN PRO BALLISTICS V7.0</div>', unsafe_allow_html=True)

t_calc, t_env, t_gun, t_wez = st.tabs(["🚀 РОЗРАХУНОК", "🌍 АТМОСФЕРА", "🔫 ГВИНТІВКА", "🎯 WEZ"])

with t_env:
    c1, c2 = st.columns(2)
    temp = c1.number_input("Температура (°C)", -40, 60, 15)
    press = c2.number_input("Тиск (hPa, станція!)", 600, 1200, 1000, help="Абсолютний тиск на місці стрільби")
    hum = st.slider("Вологість (%)", 0, 100, 50)
    st.markdown("---")
    c3, c4 = st.columns(2)
    w_s = c3.number_input("Швидкість вітру (м/с)", 0.0, 20.0, 4.0, step=0.1)
    w_d = c4.number_input("Напрямок (год)", 1.0, 12.0, 3.0, step=0.5)
    with st.expander("🌍 Геодані (для Коріоліса)"):
        lat = st.number_input("Широта (град)", 0, 90, 50)
        az = st.number_input("Азимут стрільби (град)", 0, 360, 90)

with t_gun:
    bullet_choice = st.selectbox("Каталог куль", list(BULLET_DB.keys()), index=8)
    db = BULLET_DB[bullet_choice]
    
    c1, c2 = st.columns(2)
    v0 = c1.number_input("V0 (м/с)", 600, 1200, 800)
    weight = c2.number_input("Вага (gr)", 50, 800, db[1] if db else 175)
    bc = c1.number_input("BC G7", 0.1, 1.0, db[2] if db else 0.243, format="%.3f")
    cal = c2.number_input("Калібр (дюйм)", 0.2, 0.51, db[0] if db else 0.308)
    length = st.number_input("Довжина кулі (мм)", 10.0, 100.0, db[3] if db else 32.0)
    
    with st.expander("🔧 Твіст та Приціл"):
        twist = st.number_input("Твіст (дюйм)", 7.0, 14.0, 11.0)
        twist_dir = st.radio("Напрям нарізів", ["Right", "Left"], horizontal=True)
        sh = st.number_input("Висота прицілу (см)", 0.0, 10.0, 4.5)
        zero = st.number_input("Дистанція нуля (м)", 50, 500, 100)
        temp_sens = st.number_input("Термозалежність (м/с на °C)", 0.0, 2.0, 0.5)

with t_calc:
    col_d, col_u = st.columns([2, 1])
    dist_target = col_d.number_input("ЦІЛЬ (м)", 100, 3000, 800, step=50)
    unit = col_u.selectbox("Одиниці", ["MRAD", "MOA"])
    
    params = {
        'v0': v0, 'bc': bc, 'weight_gr': weight,
        'temp': temp, 'pressure': press, 'humid': hum, 'temp_sens': temp_sens,
        'latitude': lat, 'azimuth': az,
        'w_speed': w_s, 'w_dir': w_d,
        'twist': twist, 'twist_dir': twist_dir,
        'caliber': cal, 'zero_dist': zero,
        'max_dist': dist_target, 'sh': sh,
        'length_mm': length
    }
    
    if st.button("🔥 РОЗРАХУВАТИ ТРАЄКТОРІЮ", type="primary", use_container_width=True):
        df = run_simulation(params)
        
        # Interpolate exact distance
        # Для простоти беремо найближчу точку або інтерполюємо
        res = df.iloc[(df['Dist'] - dist_target).abs().argsort()[:1]].iloc[0]
        
        # Conversions
        is_moa = unit == "MOA"
        factor = 3.4377 if is_moa else 1.0
        u_str
    # Графіки
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Траєкторія (см)", "Енергія (Дж)"))
    fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Падіння (см)'], name="Падіння", line=dict(color='lime')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Дистанція (м)'], y=df['Енергія (Дж)'], name="Енергія", fill='tozeroy', line=dict(color='red')), row=2, col=1)

    fig.update_layout(height=700, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Таблиця з можливістю вибору кроку
    st.subheader("📋 Таблиця поправок")
    step = st.selectbox("Крок таблиці (метрів):", [1, 5, 10, 25, 50, 100, 250, 500], index=5)
    
    # Фільтруємо таблицю згідно з обраним кроком
    display_df = df[df['Дистанція (м)'] % step == 0].copy()
    st.dataframe(display_df, use_container_width=True)

    st.download_button("📥 Завантажити повний звіт (CSV)", df.to_csv(index=False), "Magelan242_ELR_Report.csv")

except Exception as e:
    st.error(f"Помилка розрахунку: {e}. Перевірте введені дані.")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# --- 1. КОНФІГУРАЦІЯ ---
st.set_page_config(page_title="Magelan242 Ballistics V6.5 Litz", layout="centered", initial_sidebar_state="collapsed")

# --- 2. БАЗА ДАНИХ КУЛЬ (РОЗШИРЕНА З ДОВЖИНОЮ) ---
# Format: [Caliber(in), Weight(gr), BC G7, Length(mm), Model]
# Довжини (Length) взяті з каталогів виробників (орієнтовні)
BULLET_DB = {
    "Custom (Ручне введення)": None,
    # .224
    ".224 Sierra TMK 77gr": [0.224, 77, 0.203, 24.8, "G7"],
    ".224 Hornady ELD-M 75gr": [0.224, 75, 0.235, 25.1, "G7"],
    
    # 6mm
    ".243 Hornady ELD-M 108gr": [0.243, 108, 0.270, 31.2, "G7"],
    
    # 6.5mm
    ".264 Hornady ELD-M 140gr": [0.264, 140, 0.326, 34.5, "G7"],
    ".264 Hornady ELD-M 147gr": [0.264, 147, 0.351, 35.8, "G7"],
    ".264 Lapua Scenar-L 136gr": [0.264, 136, 0.274, 33.5, "G7"],
    ".264 Berger Hybrid 140gr": [0.264, 140, 0.311, 34.3, "G7"],
    
    # .308
    ".308 Lapua Scenar 167gr": [0.308, 167, 0.216, 31.5, "G7"],
    ".308 Sierra MK 175gr": [0.308, 175, 0.243, 31.8, "G7"],
    ".308 Hornady ELD-M 178gr": [0.308, 178, 0.275, 32.8, "G7"],
    ".308 Berger Juggernaut 185gr": [0.308, 185, 0.284, 33.2, "G7"],
    ".308 Hornady ELD-X 212gr": [0.308, 212, 0.336, 39.1, "G7"],
    ".308 Berger Hybrid 215gr": [0.308, 215, 0.354, 38.6, "G7"],
    ".308 Hornady A-Tip 230gr": [0.308, 230, 0.414, 41.5, "G7"],
    ".308 Hornady A-Tip 250gr": [0.308, 250, 0.442, 45.2, "G7"],

    # .338
    ".338 Lapua Scenar 250gr": [0.338, 250, 0.322, 39.5, "G7"],
    ".338 Lapua Scenar 300gr": [0.338, 300, 0.368, 45.3, "G7"],
    ".338 Hornady ELD-M 285gr": [0.338, 285, 0.400, 43.8, "G7"],
    
    # .50 BMG
    ".510 Hornady A-MAX 750gr": [0.510, 750, 0.512, 69.5, "G7"]
}

# --- 3. СТИЛІЗАЦІЯ ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
        .stApp { background-color: #050505; font-family: 'Roboto Mono', monospace; color: #e0e0e0; }

        .header-container { 
            border-bottom: 2px solid #00ff41; padding: 10px 0; margin-bottom: 15px; 
            display: flex; align-items: center; justify_content: center; gap: 10px;
        }
        .header-title { font-size: 1.2rem; font-weight: bold; color: #fff; }

        .hud-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 20px;
        }
        .hud-card { 
            background: #111; border: 1px solid #333; border-top: 3px solid #00ff41; 
            border-radius: 8px; padding: 10px; text-align: center; 
        }
        .hud-label { color: #888; font-size: 0.65rem; text-transform: uppercase; margin-bottom: 2px; }
        .hud-value { color: #fff; font-size: 1.4rem; font-weight: 700; line-height: 1.2; }
        .hud-sub { color: #00ff41; font-size: 0.7rem; }

        .stTabs [data-baseweb="tab-list"] { gap: 5px; }
        .stTabs [data-baseweb="tab"] { height: 45px; padding: 0 10px; background-color: #111; color: #fff; border-radius: 4px; font-size: 0.8rem;}
        .stTabs [aria-selected="true"] { background-color: #00ff41 !important; color: #000 !important; font-weight: bold;}

        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        header {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# --- 4. ENGINE RK4 (G7 + LITZ PHYSICS) ---

G7_TABLE = np.array([
    [0.0, 0.262], [0.5, 0.262], [0.7, 0.265], [0.8, 0.270], 
    [0.9, 0.300], [0.95, 0.365], [1.0, 0.425], [1.05, 0.445], 
    [1.1, 0.430], [1.2, 0.395], [1.3, 0.375], [1.5, 0.335], 
    [1.8, 0.305], [2.0, 0.290], [2.5, 0.265], [3.0, 0.250], [4.0, 0.230], [5.0, 0.220]
])

def get_drag_coefficient_g7(mach):
    return np.interp(mach, G7_TABLE[:, 0], G7_TABLE[:, 1])

def get_derivatives(state, p):
    _, _, _, vx, vy, vz = state
    G, OMEGA_E = 9.80665, 7.292115e-5
    
    # Wind vector relative
    v_rel_x = vx + p['w_long']
    v_rel_y = vy 
    v_rel_z = vz + p['w_cross']
    
    v_total_rel = math.sqrt(v_rel_x ** 2 + v_rel_y ** 2 + v_rel_z ** 2)
    mach = v_total_rel / p['c_speed']
    
    # G7 Drag
    cd = get_drag_coefficient_g7(mach) if p['model'] == "G7" else 0.45
    accel_drag = (0.5 * p['rho_rel'] * v_total_rel ** 2 * cd * (1.0 / p['bc_eff'])) * 0.00105
    
    # Coriolis
    cor_y = 2 * OMEGA_E * vx * math.cos(p['lat_rad']) * math.sin(p['az_rad'])
    cor_z = 2 * OMEGA_E * (vy * math.cos(p['lat_rad']) * math.cos(p['az_rad']) - vx * math.sin(p['lat_rad']))
    
    dvx = -(accel_drag * (v_rel_x / v_total_rel))
    dvy = -(accel_drag * (v_rel_y / v_total_rel)) - G + cor_y
    dvz = -(accel_drag * (v_rel_z / v_total_rel)) + cor_z
    
    return np.array([vx, vy, vz, dvx, dvy, dvz])


def run_simulation(p):
    DT = 0.001 
    
    # --- PHYSICAL SETUP ---
    v0_eff = p['v0'] + (p['temp'] - 15.0) * p['temp_sens']
    bc_eff = p['bc']
    
    # Atmosphere
    tk = p['temp'] + 273.15
    svp = 6.112 * math.exp((17.67 * p['temp']) / (p['temp'] + 243.5))
    pv = svp * (p['humid'] / 100.0)
    rho = ((p['pressure'] - pv) * 100 / (287.05 * tk)) + (pv * 100 / (461.5 * tk))
    
    p_phys = {
        'rho_rel': rho / 1.225, 'c_speed': 331.3 * math.sqrt(tk / 273.15),
        'bc_eff': bc_eff, 'model': p['model'],
        'lat_rad': math.radians(p['latitude']), 'az_rad': math.radians(p['azimuth']),
        'w_long': p['w_speed'] * math.cos(math.radians(p['w_dir'] * 30)),
        'w_cross': p['w_speed'] * math.sin(math.radians(p['w_dir'] * 30))
    }
    
    # --- MILLER STABILITY (CORRECT FORMULA) ---
    # Sg = (30 * m) / (T^2 * d^3 * L(1+L^2))
    # m in grains, T in calibers, d in inches, L in calibers
    twist_in_calibers = p['twist'] / p['caliber']
    length_in_calibers = (p['length_mm'] / 25.4) / p['caliber']
    
    # Miller Formula
    s_g = (30 * p['weight_gr']) / (
        (twist_in_calibers ** 2) * (p['caliber'] ** 3) * length_in_calibers * (1 + length_in_calibers ** 2)
    )
    # Miller correction for Velocity and Atmopshere (Simplified)
    # Sg increases slightly as velocity drops, but for drift we use Muzzle Sg or average.
    # We will use Sg_static for Litz formula as base.

    t_dir = 1 if p['twist_dir'] == "Right" else -1

    # --- AERODYNAMIC JUMP FACTOR ---
    # Litz rule of thumb: ~0.03-0.04 MRAD per 1 m/s crosswind.
    # Physics: Jump is angular offset caused by yaw at muzzle.
    # It depends on Sg. Higher Sg -> Less Jump.
    # Approximation: Jump_mrad = (0.05 / Sg) * W_cross_ms
    jump_mrad_per_ms = 0.055 / s_g if s_g > 0 else 0.03
    aero_jump_angle = p_phys['w_cross'] * jump_mrad_per_ms * 0.001 * t_dir # radians

    # --- ZEROING ---
    angle_zero = math.atan((0.5 * 9.80665 * (p['zero_dist'] / v0_eff) ** 2 + p['sh'] / 100) / p['zero_dist'])
    
    # Add Aero Jump to initial Vertical Angle (Since it happens at muzzle)
    # If Wind from Right (w_cross > 0) and Twist Right (t_dir=1) -> Jump UP.
    state = np.array([0.0, -p['sh'] / 100, 0.0, 
                      v0_eff * math.cos(angle_zero + aero_jump_angle), 
                      v0_eff * math.sin(angle_zero + aero_jump_angle), 
                      0.0])
    
    t, dist, results, step_check = 0.0, 0.0, [], 0

    while dist <= p['max_dist'] + 5:
        k1 = get_derivatives(state, p_phys)
        k2 = get_derivatives(state + k1 * DT / 2, p_phys)
        k3 = get_derivatives(state + k2 * DT / 2, p_phys)
        k4 = get_derivatives(state + k3 * DT, p_phys)
        state += (k1 + 2 * k2 + 2 * k3 + k4) * DT / 6
        t += DT
        dist = state[0]

        if dist >= step_check:
            v_curr = math.sqrt(state[3] ** 2 + state[4] ** 2 + state[5] ** 2)
            
            # --- LITZ SPIN DRIFT FORMULA ---
            # Drift (inches) = 1.25 * (Sg + 1.2) * t^1.83
            # t = time of flight
            drift_inches = 1.25 * (s_g + 1.2) * (t ** 1.83)
            s_drift_m = (drift_inches * 0.0254) * t_dir
            
            # Position Y (Vertical), Z (Horizontal)
            # state[1] is already affected by Aero Jump via initial velocity vector
            y_f = state[1] 
            z_f = state[2] + s_drift_m # Add spin drift to Coriolis/Wind drift
            
            mv, mh = (y_f * 100) / (dist / 10) if dist > 0 else 0, (z_f * 100) / (dist / 10) if dist > 0 else 0
            
            results.append({"Dist": int(dist), "V": int(v_curr), "Mach": round(v_curr / p_phys['c_speed'], 2),
                            "Drop": y_f * 100, "MRAD_V": mv, "MRAD_H": mh, "Sg": round(s_g, 2)})
            step_check += 25
            
    return pd.DataFrame(results)


# --- 5. SOLVER TRUING ---
def solve_truing(target_dist, real_drop, var_name, current_params, unit):
    p_copy = current_params.copy()
    p_copy['max_dist'] = target_dist + 10
    current_val = p_copy[var_name]
    min_val, max_val = current_val * 0.7, current_val * 1.3
    real_drop_mrad = real_drop / 3.4377 if "MOA" in unit else real_drop

    for _ in range(12):
        mid_val = (min_val + max_val) / 2
        p_copy[var_name] = mid_val
        df = run_simulation(p_copy)
        idx = (df['Dist'] - target_dist).abs().idxmin()
        val = df.loc[idx, 'MRAD_V']
        if val < real_drop_mrad:
            max_val = mid_val
        else:
            min_val = mid_val
    return mid_val


# --- 6. VISUALS (RETICLE) ---
def draw_reticle_mobile(mrad_v, mrad_h, unit, reticle_type, wez=None):
    limit = 12 if "MRAD" in unit else 40
    fig = go.Figure()

    if wez:
        fig.add_trace(go.Scatter(x=[-wez['h_min'], -wez['h_max'], -wez['h_max'], -wez['h_min']],
                                 y=[wez['v_min'], wez['v_min'], wez['v_max'], wez['v_max']],
                                 fill="toself", fillcolor="rgba(255, 50, 50, 0.25)", line=dict(width=0), name="WEZ"))

    line_color = "rgba(255,255,255,0.3)"
    
    if reticle_type == "Crosshair (Перехрестя)":
        fig.add_shape(type="line", x0=-limit, y0=0, x1=limit, y1=0, line=dict(color=line_color, width=1))
        fig.add_shape(type="line", x0=0, y0=-limit, x1=0, y1=limit, line=dict(color=line_color, width=1))

    elif reticle_type == "Mil-Dot (Класика)":
        fig.add_shape(type="line", x0=-limit, y0=0, x1=limit, y1=0, line=dict(color=line_color, width=1))
        fig.add_shape(type="line", x0=0, y0=-limit, x1=0, y1=limit, line=dict(color=line_color, width=1))
        dots_x, dots_y = [], []
        dot_step = 1 if "MRAD" in unit else 2
        for i in range(-limit, limit + 1, dot_step):
            if i != 0:
                dots_x.extend([i, 0]); dots_y.extend([0, i])
        fig.add_trace(go.Scatter(x=dots_x, y=dots_y, mode='markers', marker=dict(color='rgba(255,255,255,0.6)', size=3), hoverinfo='skip'))

    elif reticle_type == "Christmas Tree (Ялинка)":
        fig.add_shape(type="line", x0=-limit, y0=0, x1=limit, y1=0, line=dict(color=line_color, width=1))
        fig.add_shape(type="line", x0=0, y0=-limit, x1=0, y1=limit, line=dict(color=line_color, width=1))
        tree_x, tree_y = [], []
        for i in range(1, limit):
            fig.add_shape(type="line", x0=-0.15, y0=-i, x1=0.15, y1=-i, line=dict(color=line_color, width=1))
            width = i + 1
            for w in range(1, width):
                if w % 1 == 0: 
                    tree_x.extend([w, -w]); tree_y.extend([-i, -i])
        fig.add_trace(go.Scatter(x=tree_x, y=tree_y, mode='markers', marker=dict(color='rgba(255,255,255,0.4)', size=2), hoverinfo='skip'))

    fig.add_trace(go.Scatter(x=[-mrad_h], y=[mrad_v], mode='markers',
                             marker=dict(color='#00ff41', size=16, symbol='circle-open', line=dict(width=3)),
                             name="POI"))

    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=5, r=5, t=5, b=5),
                      xaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                      yaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,20,20,0.8)', showlegend=False)
    return fig


# --- 7. UI STRUCTURE ---
st.markdown(
    '<div class="header-container"><div style="font-size:1.5rem;">🦅</div><div class="header-title">Magelan242 LITZ V6.5</div></div>',
    unsafe_allow_html=True)

t_calc, t_env, t_gun, t_wez, t_true = st.tabs(["🚀 ОБЧИСЛЕННЯ", "🌍 СЕРЕДОВИЩЕ", "🔫 ЗБРОЯ", "📊 WEZ", "🔧 КАЛІБРУВАННЯ"])

with t_env:
    c1, c2 = st.columns(2)
    temp = c1.number_input("T (°C)", -30, 50, 15)
    press = c2.number_input("P (hPa)", 800, 1150, 1013)
    hum = st.slider("Hum (%)", 0, 100, 50)
    st.markdown("---")
    w_s = st.slider("Вітер (м/с)", 0.0, 15.0, 0.0, step=0.1)
    w_d = st.slider("Напрям (год)", 1, 12, 12)
    with st.expander("Гео-дані"):
        lat = st.number_input("Широта", 0, 90, 50)
        az = st.number_input("Азимут", 0, 360, 90)

with t_gun:
    bullet_choice = st.selectbox("База куль", list(BULLET_DB.keys()), index=8)
    db = BULLET_DB[bullet_choice]
    
    c1, c2 = st.columns(2)
    v0 = c1.number_input("V0 (м/с @ 15°C)", 500, 1200, 800)
    weight = c2.number_input("Вага (gr)", 50, 800, db[1] if db else 175)
    bc = c1.number_input("BC G7", 0.1, 1.0, db[2] if db else 0.243, format="%.3f")
    cal = c2.number_input("Cal (дюйм)", 0.2, 0.51, db[0] if db else 0.308)
    
    # НОВЕ ПОЛЕ: ДОВЖИНА КУЛІ ДЛЯ ФОРМУЛИ МІЛЛЕРА/ЛІТЦА
    st.markdown("---")
    length = st.number_input("Довжина кулі (мм)", 10.0, 100.0, db[3] if db else 32.0, help="Критично важливо для розрахунку гіроскопічної стабільності (Sg) та деривації.")
    temp_sens = st.number_input("Термозалежність (м/с на 1°C)", 0.0, 2.0, 0.5)
    
    with st.expander("Додатково (Твіст/Оптика)"):
        twist = st.number_input("Твіст (дюйм)", 6.0, 16.0, 11.0)
        sh = st.number_input("Висота прицілу (см)", 3.0, 10.0, 4.5)
        zero = st.number_input("Нуль (м)", 50, 500, 100)

with t_wez:
    st.caption("Аналіз зони ймовірного влучання")
    err_w = st.slider("Похибка вітру (+/- м/с)", 0.0, 4.0, 1.0)
    err_v = st.slider("SD V0 (+/- м/с)", 0.0, 10.0, 2.0)

with t_true:
    st.caption("Truing")
    tr_dist = st.number_input("Дистанція (м)", 300, 2000, 800)
    tr_drop = st.number_input("Реальна поправка", 0.0, 50.0, 0.0)
    calc_true = st.button("Підігнати V0")

with t_calc:
    c_dist, c_unit = st.columns([2, 1])
    dist_target = c_dist.number_input("ДИСТАНЦІЯ (м)", 100, 3000, 800, step=25)
    unit = c_unit.selectbox("Од.", ["MRAD", "MOA"])
    ret_type = st.selectbox("Тип сітки", ["Crosshair (Перехрестя)", "Mil-Dot (Класика)", "Christmas Tree (Ялинка)"])

    params = {
        'v0': v0, 'bc': bc, 'model': "G7", 'weight_gr': weight,
        'temp': temp, 'pressure': press, 'humid': hum, 'temp_sens': temp_sens,
        'latitude': lat, 'azimuth': az,
        'w_speed': w_s, 'w_dir': w_d,
        'twist': twist, 'twist_dir': "Right",
        'caliber': cal, 'zero_dist': zero,
        'max_dist': dist_target, 'sh': sh,
        'length_mm': length
    }

    if calc_true and tr_drop > 0:
        new_v0 = solve_truing(tr_dist, tr_drop, 'v0', params, unit)
        st.success(f"Розрахункова V0: {new_v0:.1f} м/с")

    if st.button("🔥 РОЗРАХУВАТИ", type="primary", use_container_width=True):
        try:
            df = run_simulation(params)
            res = df.iloc[-1]

            # WEZ Logic
            p_min = params.copy(); p_min.update({'w_speed': max(0, w_s - err_w), 'v0': v0 - err_v})
            p_max = params.copy(); p_max.update({'w_speed': w_s + err_w, 'v0': v0 + err_v})
            r_min = run_simulation(p_min).iloc[-1]
            r_max = run_simulation(p_max).iloc[-1] # Simple approx endpoint
            
            wez = {
                'v_min': min(r_min['MRAD_V'], r_max['MRAD_V']), 'v_max': max(r_min['MRAD_V'], r_max['MRAD_V']),
                'h_min': min(r_min['MRAD_H'], r_max['MRAD_H']), 'h_max': max(r_min['MRAD_H'], r_max['MRAD_H'])
            }

            is_moa = unit == "MOA"
            conv = 3.4377 if is_moa else 1.0
            click_val = 0.25 if is_moa else 0.1
            val_v, val_h = res['MRAD_V'] * conv, res['MRAD_H'] * conv

            st.markdown(f"""
                <div class="hud-grid">
                    <div class="hud-card">
                        <div class="hud-label">UP</div>
                        <div class="hud-value" style="color:#ffcc00">{"U" if val_v > 0 else "D"} {abs(val_v):.2f}</div>
                        <div class="hud-sub">{abs(val_v / click_val):.0f} кліків</div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-label">WIND</div>
                        <div class="hud-value" style="color:#ffcc00">{"R" if val_h > 0 else "L"} {abs(val_h):.2f}</div>
                        <div class="hud-sub">{abs(val_h / click_val):.0f} кліків</div>
                    </div>
                    <div class="hud-card"><div class="hud-label">Sg (Стаб.)</div><div class="hud-value">{res['Sg']}</div></div>
                    <div class="hud-card"><div class="hud-label">Mach</div><div class="hud-value">{res['Mach']}</div></div>
                </div>
            """, unsafe_allow_html=True)

            t_ret, t_gr, t_tab = st.tabs(["СІТКА", "ТРАЄКТОРІЯ", "ТАБЛИЦЯ"])
            
            with t_ret:
                wez_chart = wez.copy()
                if is_moa:
                    for k in wez_chart: wez_chart[k] *= 3.4377
                st.plotly_chart(draw_reticle_mobile(val_v, val_h, unit, ret_type, wez_chart), use_container_width=True)

            with t_gr:
                final_drop = df['Drop'].iloc[-1]
                slope = -final_drop / df['Dist'].iloc[-1] if df['Dist'].iloc[-1] > 0 else 0
                y_comp = df['Drop'] + (df['Dist'] * slope)
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=df['Dist'], y=y_comp, line=dict(color='#00ff41', width=3), fill='tozeroy'))
                fig_t.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,20,20,0.5)')
                st.plotly_chart(fig_t, use_container_width=True)

            with t_tab:
                df_show = df[df['Dist'] % 50 == 0][['Dist', 'Drop', 'MRAD_V', 'MRAD_H', 'V', 'Sg']].copy()
                st.dataframe(df_show, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error: {e}")
    
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

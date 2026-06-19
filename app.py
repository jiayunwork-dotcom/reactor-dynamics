import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, fsolve
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

from reactor_models import (
    R_GAS,
    solve_cstr_steady,
    solve_cstr_steady_fast,
    get_steady_temp_fast,
    find_critical_cooling,
    run_dynamic_cstr,
    run_dynamic_pfr,
    run_dynamic_semibatch,
    run_dynamic_multicstr,
    reaction_rates,
    stoich_net
)

st.set_page_config(page_title="化工反应器动力学建模与温度控制优化", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

COMPONENTS = ['A', 'B', 'C', 'D', 'E']

DEFAULT_CASES = {
    "单一不可逆放热反应 A→B": [
        {
            'reactants': ['A'], 'reactant_coeffs': {'A': 1.0}, 'reactant_orders': [1],
            'products': ['B'], 'product_coeffs': {'B': 1.0},
            'A': 1e10, 'Ea': 80.0, 'dH': -100.0
        }
    ],
    "可逆反应 A⇌B": [
        {
            'reactants': ['A'], 'reactant_coeffs': {'A': 1.0}, 'reactant_orders': [1],
            'products': ['B'], 'product_coeffs': {'B': 1.0},
            'A': 1e9, 'Ea': 60.0, 'dH': -50.0
        },
        {
            'reactants': ['B'], 'reactant_coeffs': {'B': 1.0}, 'reactant_orders': [1],
            'products': ['A'], 'product_coeffs': {'A': 1.0},
            'A': 1e8, 'Ea': 40.0, 'dH': 50.0
        }
    ],
    "串联反应 A→B→C": [
        {
            'reactants': ['A'], 'reactant_coeffs': {'A': 1.0}, 'reactant_orders': [1],
            'products': ['B'], 'product_coeffs': {'B': 1.0},
            'A': 2e9, 'Ea': 70.0, 'dH': -60.0
        },
        {
            'reactants': ['B'], 'reactant_coeffs': {'B': 1.0}, 'reactant_orders': [1],
            'products': ['C'], 'product_coeffs': {'C': 1.0},
            'A': 5e8, 'Ea': 90.0, 'dH': -40.0
        }
    ]
}


def get_default_reaction(idx):
    return {
        'reactants': ['A'], 'reactant_coeffs': {'A': 1.0}, 'reactant_orders': [1],
        'products': ['B'], 'product_coeffs': {'B': 1.0},
        'A': 1e10, 'Ea': 80.0, 'dH': -80.0
    }


def reaction_equation_str(rxn):
    reactant_str = " + ".join([f"{int(c) if c == int(c) else c}{k}" for k, c in rxn['reactant_coeffs'].items()])
    product_str = " + ".join([f"{int(c) if c == int(c) else c}{k}" for k, c in rxn['product_coeffs'].items()])
    arrow = "⇌" if any(r['dH'] > 0 for r in st.session_state.reactions) else "→"
    return f"{reactant_str} {arrow} {product_str}"


def sidebar_reactor_config():
    st.sidebar.title("⚙️ 反应器配置")
    
    reactor_type = st.sidebar.selectbox(
        "选择反应器类型",
        ["连续搅拌釜式反应器 (CSTR)", "管式反应器 (PFR)", "半间歇反应器", "多级串联CSTR"],
        index=0
    )
    
    params = {}
    
    if reactor_type == "连续搅拌釜式反应器 (CSTR)":
        params['V'] = st.sidebar.number_input("反应器体积 V (m³)", 0.1, 10.0, 1.0, 0.1)
        params['F'] = st.sidebar.number_input("进料流量 F (m³/s)", 0.001, 1.0, 0.01, 0.001, format="%.3f")
        params['tau'] = params['V'] / params['F']
        st.sidebar.info(f"停留时间 τ = {params['tau']:.1f} s")
        
    elif reactor_type == "管式反应器 (PFR)":
        params['L'] = st.sidebar.number_input("管长 L (m)", 1.0, 50.0, 10.0, 1.0)
        params['A_cross'] = st.sidebar.number_input("横截面积 (m²)", 0.001, 0.1, 0.01, 0.001)
        params['u'] = st.sidebar.number_input("流速 u (m/s)", 0.01, 5.0, 0.5, 0.01)
        params['perimeter'] = st.sidebar.number_input("管周长 (m)", 0.05, 1.0, 0.35, 0.01)
        params['UA_per_L'] = st.sidebar.number_input("单位长度换热系数 (W/(m·K))", 100.0, 5000.0, 1000.0, 100.0)
        
    elif reactor_type == "半间歇反应器":
        params['V_max'] = st.sidebar.number_input("最大体积 V_max (m³)", 0.5, 10.0, 2.0, 0.1)
        params['F_in'] = st.sidebar.number_input("进料流量 F_in (m³/s)", 0.001, 0.1, 0.01, 0.001, format="%.3f")
        
    else:
        n_stages = st.sidebar.slider("级数", 2, 10, 3, 1)
        params['n_stages'] = n_stages
        equal_vol = st.sidebar.checkbox("各级体积相等", True)
        if equal_vol:
            V_total = st.sidebar.number_input("总体积 V_total (m³)", 0.5, 20.0, 3.0, 0.1)
            for i in range(n_stages):
                params[f'V_{i}'] = V_total / n_stages
        else:
            for i in range(n_stages):
                params[f'V_{i}'] = st.sidebar.number_input(f"第{i+1}级体积 (m³)", 0.1, 5.0, 1.0, 0.1)
        params['F'] = st.sidebar.number_input("进料流量 F (m³/s)", 0.001, 1.0, 0.01, 0.001, format="%.3f")
    
    params['rho_cp'] = st.sidebar.number_input("ρCp (J/(m³·K))", 1e3, 1e4, 4e3, 1e2)
    
    st.sidebar.subheader("进料条件")
    params['T_f'] = st.sidebar.slider("进料温度 T_f (K)", 250.0, 400.0, 300.0, 1.0)
    params['C_f'] = np.zeros(5)
    for i, comp in enumerate(COMPONENTS):
        params['C_f'][i] = st.sidebar.number_input(f"C_{comp},f (mol/m³)", 0.0, 1000.0, 100.0 if i == 0 else 0.0, 1.0)
    
    st.sidebar.subheader("冷却条件")
    if reactor_type == "多级串联CSTR":
        equal_cooling = st.sidebar.checkbox("各级冷却条件相同", True)
        if equal_cooling:
            params['T_c'] = st.sidebar.slider("冷却介质温度 T_c (K)", 250.0, 350.0, 290.0, 1.0)
            params['UA'] = st.sidebar.number_input("总传热系数*面积 UA (W/K)", 100.0, 10000.0, 2000.0, 100.0)
        else:
            for i in range(n_stages):
                params[f'T_c_{i}'] = st.sidebar.slider(f"T_c,{i+1} (K)", 250.0, 350.0, 290.0, 1.0)
                params[f'UA_{i}'] = st.sidebar.number_input(f"UA_{i+1} (W/K)", 100.0, 10000.0, 2000.0, 100.0)
    else:
        params['T_c'] = st.sidebar.slider("冷却介质温度 T_c (K)", 250.0, 350.0, 290.0, 1.0)
        params['UA'] = st.sidebar.number_input("总传热系数*面积 UA (W/K)", 100.0, 10000.0, 2000.0, 100.0)
    
    return reactor_type, params


def sidebar_reaction_config():
    st.sidebar.title("🧪 反应动力学配置")
    
    if 'reactions' not in st.session_state:
        st.session_state.reactions = DEFAULT_CASES["单一不可逆放热反应 A→B"].copy()
    
    for rxn in st.session_state.reactions:
        rxn['reactant_orders'] = [int(o) for o in rxn['reactant_orders']]
    
    case = st.sidebar.selectbox(
        "加载经典案例",
        ["自定义"] + list(DEFAULT_CASES.keys()),
        index=1
    )
    
    if case != "自定义":
        st.session_state.reactions = DEFAULT_CASES[case].copy()
        for rxn in st.session_state.reactions:
            rxn['reactant_orders'] = [int(o) for o in rxn['reactant_orders']]
    
    n_reactions = st.sidebar.slider("反应数量", 1, 3, len(st.session_state.reactions))
    
    while len(st.session_state.reactions) < n_reactions:
        st.session_state.reactions.append(get_default_reaction(len(st.session_state.reactions)))
    while len(st.session_state.reactions) > n_reactions:
        st.session_state.reactions.pop()
    
    reactions = []
    for i in range(n_reactions):
        with st.sidebar.expander(f"反应 {i+1}", expanded=True):
            rxn = st.session_state.reactions[i]
            
            st.markdown(f"**{reaction_equation_str(rxn)}**")
            
            col1, col2 = st.columns(2)
            with col1:
                n_reactants = st.number_input(f"反应物种类", 1, 3, len(rxn['reactants']), key=f'nr_{i}')
            with col2:
                n_products = st.number_input(f"产物种类", 1, 3, len(rxn['products']), key=f'np_{i}')
            
            reactants = []
            reactant_coeffs = {}
            reactant_orders = []
            for j in range(n_reactants):
                r_comp = st.selectbox(f"反应物{j+1}", COMPONENTS, 
                                     index=COMPONENTS.index(rxn['reactants'][j]) if j < len(rxn['reactants']) else 0,
                                     key=f'rc_{i}_{j}')
                r_coeff = st.number_input(f"系数 {r_comp}", 0.1, 5.0, 
                                         rxn['reactant_coeffs'].get(r_comp, 1.0), 0.1, key=f'rcoeff_{i}_{j}')
                default_order = int(rxn['reactant_orders'][j]) if j < len(rxn['reactant_orders']) else 1
                order_index = [0, 1, 2].index(default_order) if default_order in [0, 1, 2] else 1
                r_order = st.selectbox(f"反应级数 {r_comp}", [0, 1, 2], 
                                      index=order_index,
                                      key=f'ro_{i}_{j}')
                reactants.append(r_comp)
                reactant_coeffs[r_comp] = r_coeff
                reactant_orders.append(r_order)
            
            products = []
            product_coeffs = {}
            for j in range(n_products):
                p_comp = st.selectbox(f"产物{j+1}", COMPONENTS, 
                                     index=COMPONENTS.index(rxn['products'][j]) if j < len(rxn['products']) else 1,
                                     key=f'pc_{i}_{j}')
                p_coeff = st.number_input(f"系数 {p_comp}", 0.1, 5.0, 
                                         rxn['product_coeffs'].get(p_comp, 1.0), 0.1, key=f'pcoeff_{i}_{j}')
                products.append(p_comp)
                product_coeffs[p_comp] = p_coeff
            
            A = st.number_input(f"频率因子 A (s⁻¹)", 1e3, 1e15, rxn['A'], format="%.2e", key=f'A_{i}')
            Ea = st.slider(f"活化能 Ea (kJ/mol)", 20.0, 200.0, rxn['Ea'], 1.0, key=f'Ea_{i}')
            dH = st.slider(f"反应热 ΔH (kJ/mol)", -200.0, 200.0, rxn['dH'], 1.0, key=f'dH_{i}')
            
            reactions.append({
                'reactants': reactants, 'reactant_coeffs': reactant_coeffs, 
                'reactant_orders': [int(o) for o in reactant_orders],
                'products': products, 'product_coeffs': product_coeffs,
                'A': A, 'Ea': Ea, 'dH': dH
            })
    
    st.session_state.reactions = reactions
    return reactions


def page_steady_state_analysis(reactor_type, params, reactions):
    st.header("📊 CSTR稳态多重性分析")
    
    if "CSTR" not in reactor_type or "多级" in reactor_type:
        st.warning("稳态分析仅适用于单级CSTR反应器")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("参数调节")
        T_f_slider = st.slider("进料温度 T_f (K)", 250.0, 400.0, params['T_f'], 1.0, key='ss_Tf')
        T_c_slider = st.slider("冷却介质温度 T_c (K)", 250.0, 350.0, params['T_c'], 1.0, key='ss_Tc')
        UA_slider = st.slider("UA (W/K)", 100.0, 10000.0, params['UA'], 100.0, key='ss_UA')
        
        params_adj = params.copy()
        params_adj['T_f'] = T_f_slider
        params_adj['T_c'] = T_c_slider
        params_adj['UA'] = UA_slider
        
        if st.button("计算临界冷却温度"):
            T_c_range, n_steady, critical_points = find_critical_cooling(params_adj, reactions)
            st.success(f"找到 {len(critical_points)} 个临界点")
            for cp in critical_points:
                st.info(f"T_c = {cp['T_c']:.1f} K: 从 {cp['from']} 个稳态变为 {cp['to']} 个稳态")
    
    with col1:
        T_range, Q_gen, Q_rem, steady_states = solve_cstr_steady(params_adj, reactions)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(T_range, Q_gen / 1000, 'b-', label='热量产生速率 Q_gen', linewidth=2)
        ax.plot(T_range, Q_rem / 1000, 'r-', label='热量移除速率 Q_rem', linewidth=2)
        
        colors = ['g', 'orange', 'purple']
        for idx, ss in enumerate(steady_states):
            marker = 'o' if ss['stable'] else 's'
            label = f"稳态点{idx+1}: T={ss['T']:.1f}K, {'稳定' if ss['stable'] else '不稳定'}"
            ax.plot(ss['T'], np.interp(ss['T'], T_range, Q_gen) / 1000, 
                    marker, markersize=10, markeredgecolor='k', 
                    markerfacecolor=colors[idx % len(colors)], label=label)
        
        ax.set_xlabel('温度 T (K)', fontsize=12)
        ax.set_ylabel('热量速率 (kW)', fontsize=12)
        ax.set_title('CSTR热平衡曲线与稳态操作点', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    if steady_states:
        st.subheader("稳态点详情")
        ss_data = []
        for idx, ss in enumerate(steady_states):
            row = {
                '稳态点': idx + 1,
                '温度 (K)': f"{ss['T']:.1f}",
                'C_A (mol/m³)': f"{ss['C'][0]:.2f}",
                'C_B (mol/m³)': f"{ss['C'][1]:.2f}",
                'C_C (mol/m³)': f"{ss['C'][2]:.2f}",
                '转化率 (%)': f"{(1 - ss['C'][0] / max(params_adj['C_f'][0], 1e-10)) * 100:.1f}",
                '稳定性': '✅ 稳定' if ss['stable'] else '❌ 不稳定'
            }
            ss_data.append(row)
        st.table(ss_data)


def page_dynamic_simulation(reactor_type, params, reactions):
    st.header("⏱️ 动态仿真分析")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("仿真设置")
        t_final = st.number_input("仿真时间 (s)", 10.0, 10000.0, 500.0, 10.0)
        
        st.subheader("初始条件")
        use_steady = st.checkbox("使用稳态点作为初始条件", True)
        
        if not use_steady:
            C0 = np.zeros(5)
            for i, comp in enumerate(COMPONENTS):
                C0[i] = st.number_input(f"C_{comp}(0) (mol/m³)", 0.0, 1000.0, params['C_f'][i], 1.0, key=f'dyn_C0_{i}')
            T0 = st.number_input("T(0) (K)", 250.0, 500.0, params['T_f'], 1.0, key='dyn_T0')
            if reactor_type == "半间歇反应器":
                V0 = st.number_input("V(0) (m³)", 0.1, params['V_max'], 0.5, 0.1, key='dyn_V0')
        else:
            if "CSTR" in reactor_type and "多级" not in reactor_type:
                _, _, _, steady_states = solve_cstr_steady(params, reactions)
                if steady_states:
                    ss_idx = st.selectbox("选择稳态点", list(range(len(steady_states))), 
                                          format_func=lambda x: f"稳态点{x+1}: T={steady_states[x]['T']:.1f}K")
                    C0 = steady_states[ss_idx]['C']
                    T0 = steady_states[ss_idx]['T']
                else:
                    C0 = params['C_f']
                    T0 = params['T_f']
            else:
                C0 = params['C_f']
                T0 = params['T_f']
        
        st.subheader("阶跃扰动")
        n_disturbances = st.number_input("扰动数量", 0, 3, 0, 1)
        disturbances = []
        for i in range(n_disturbances):
            with st.expander(f"扰动 {i+1}"):
                dist_time = st.number_input("施加时间 (s)", 0.0, t_final, t_final * 0.4, 1.0, key=f'dist_t_{i}')
                dist_type = st.selectbox("扰动类型", ['T_f', 'F', 'C_f', 'F_in'], key=f'dist_type_{i}')
                if dist_type == 'C_f':
                    dist_val = np.zeros(5)
                    for j, comp in enumerate(COMPONENTS):
                        dist_val[j] = st.number_input(f"新 C_{comp},f", 0.0, 1000.0, params['C_f'][j] * 1.2, 1.0, key=f'dist_cf_{i}_{j}')
                else:
                    base_val = params.get(dist_type, params.get('T_f', 300.0))
                    dist_val = st.number_input("新值", 0.0, 1000.0, base_val * 1.2, 1.0, key=f'dist_val_{i}')
                disturbances.append({'time': dist_time, 'type': dist_type, 'value': dist_val})
    
    with col2:
        if st.button("🚀 开始仿真", type="primary", use_container_width=True):
            with st.spinner("正在进行仿真计算..."):
                if reactor_type == "连续搅拌釜式反应器 (CSTR)":
                    y0 = np.zeros(6)
                    y0[:5] = C0
                    y0[5] = T0
                    t, y = run_dynamic_cstr(params, reactions, y0, [0, t_final], disturbances)
                    plot_dynamic_results(t, y, ['CSTR'], params, disturbances)
                
                elif reactor_type == "管式反应器 (PFR)":
                    t_span = [0, t_final]
                    
                    def pfr_dynamic_ode(t, y, params, reactions):
                        n_points = 50
                        dydt = np.zeros(6 * n_points)
                        
                        for k in range(n_points):
                            idx = k * 6
                            C = np.clip(y[idx:idx+5], 0, None)
                            T = y[idx+5]
                            
                            u = params['u']
                            dz = params['L'] / (n_points - 1)
                            
                            rates = reaction_rates(C, T, reactions)
                            net = stoich_net(reactions)
                            
                            if k < n_points - 1:
                                next_idx = (k + 1) * 6
                                C_next = y[next_idx:next_idx+5]
                                T_next = y[next_idx+5]
                                dCdz = (C_next - C) / dz
                                dTdz = (T_next - T) / dz
                            else:
                                dCdz = np.zeros(5)
                                dTdz = 0
                            
                            dCdt = -u * dCdz
                            for j, comp in enumerate(COMPONENTS):
                                dCdt[j] += net[comp] * sum(rates)
                            
                            Q_gen = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates))
                            Q_rem = params['UA_per_L'] * (T - params['T_c']) * params['perimeter'] / params['A_cross']
                            dTdt = (-u * dTdz + (Q_gen - Q_rem) / params['rho_cp'])
                            
                            dydt[idx:idx+5] = dCdt
                            dydt[idx+5] = dTdt
                        
                        return dydt
                    
                    n_points = 50
                    y0 = np.zeros(6 * n_points)
                    for k in range(n_points):
                        idx = k * 6
                        y0[idx:idx+5] = C0
                        y0[idx+5] = T0
                    y0[:5] = params['C_f']
                    y0[5] = params['T_f']
                    
                    t_eval = np.linspace(0, t_final, 100)
                    sol = solve_ivp(
                        lambda t, y: pfr_dynamic_ode(t, y, params, reactions),
                        [0, t_final], y0, t_eval=t_eval, method='LSODA',
                        rtol=1e-6, atol=1e-8, max_step=5.0
                    )
                    t = sol.t
                    y = sol.y
                    
                    plot_pfr_dynamic_results(t, y, params, n_points)
                
                elif reactor_type == "半间歇反应器":
                    y0 = np.zeros(7)
                    y0[:5] = C0
                    y0[5] = T0
                    y0[6] = V0 if 'V0' in locals() else 0.5
                    t, y = run_dynamic_semibatch(params, reactions, y0, [0, t_final], disturbances)
                    plot_dynamic_results(t, y, ['半间歇'], params, disturbances, has_volume=True)
                
                else:
                    n_stages = params['n_stages']
                    y0 = np.zeros(6 * n_stages)
                    for i in range(n_stages):
                        idx = i * 6
                        y0[idx:idx+5] = C0
                        y0[idx+5] = T0
                    t, y = run_dynamic_multicstr(params, reactions, y0, [0, t_final], n_stages, disturbances)
                    plot_multi_cstr_results(t, y, n_stages, params, disturbances)


def plot_dynamic_results(t, y, labels, params, disturbances, has_volume=False):
    n_plots = 4 if has_volume else 3
    fig = make_subplots(rows=n_plots, cols=1, shared_xaxes=True,
                       subplot_titles=('各组分浓度变化', '反应器温度变化', '转化率变化', '反应器体积变化') if has_volume else 
                                       ('各组分浓度变化', '反应器温度变化', '转化率变化'))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, comp in enumerate(COMPONENTS):
        if params['C_f'][i] > 0 or np.max(y[i]) > 0.01:
            fig.add_trace(go.Scatter(x=t, y=y[i], name=f'C_{comp}', line=dict(color=colors[i])), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=t, y=y[5], name='温度', line=dict(color='red', width=2)), row=2, col=1)
    
    if params['C_f'][0] > 0:
        conversion = (1 - y[0] / params['C_f'][0]) * 100
        fig.add_trace(go.Scatter(x=t, y=conversion, name='转化率', line=dict(color='green', width=2)), row=3, col=1)
    
    if has_volume:
        fig.add_trace(go.Scatter(x=t, y=y[6], name='体积', line=dict(color='purple', width=2)), row=4, col=1)
    
    for dist in disturbances:
        for row in range(1, n_plots + 1):
            fig.add_vline(x=dist['time'], line_dash="dash", line_color="red", 
                         annotation_text=f"{dist['type']}扰动", annotation_position="top right",
                         row=row, col=1)
    
    fig.update_xaxes(title_text="时间 (s)", row=n_plots, col=1)
    fig.update_yaxes(title_text="浓度 (mol/m³)", row=1, col=1)
    fig.update_yaxes(title_text="温度 (K)", row=2, col=1)
    fig.update_yaxes(title_text="转化率 (%)", row=3, col=1)
    if has_volume:
        fig.update_yaxes(title_text="体积 (m³)", row=4, col=1)
    
    fig.update_layout(height=300 * n_plots, title_text="动态仿真结果", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_pfr_dynamic_results(t, y, params, n_points):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       subplot_titles=('出口浓度变化', '出口温度变化', '沿管长分布剖面图'))
    
    outlet_idx = (n_points - 1) * 6
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, comp in enumerate(COMPONENTS):
        if params['C_f'][i] > 0 or np.max(y[outlet_idx + i]) > 0.01:
            fig.add_trace(go.Scatter(x=t, y=y[outlet_idx + i], name=f'C_{comp} 出口', 
                                    line=dict(color=colors[i])), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=t, y=y[outlet_idx + 5], name='出口温度', 
                            line=dict(color='red', width=2)), row=2, col=1)
    
    z = np.linspace(0, params['L'], n_points)
    final_idx = -1
    C_final = np.zeros(n_points)
    T_final = np.zeros(n_points)
    for k in range(n_points):
        idx = k * 6
        C_final[k] = y[idx, final_idx]
        T_final[k] = y[idx + 5, final_idx]
    
    fig.add_trace(go.Scatter(x=z, y=C_final, name='C_A 分布', line=dict(color='#1f77b4')), row=3, col=1)
    fig.add_trace(go.Scatter(x=z, y=T_final, name='温度分布', line=dict(color='red'), yaxis='y2'), row=3, col=1)
    
    fig.update_layout(height=900, title_text="PFR动态仿真结果")
    st.plotly_chart(fig, use_container_width=True)


def plot_multi_cstr_results(t, y, n_stages, params, disturbances):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       subplot_titles=('各级温度变化', '各级A浓度变化', '各级B浓度变化'))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_stages))
    
    for i in range(n_stages):
        idx = i * 6
        fig.add_trace(go.Scatter(x=t, y=y[idx + 5], name=f'T_{i+1}', 
                                line=dict(color=f'rgb({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)})')), 
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=y[idx], name=f'C_A,{i+1}', 
                                line=dict(color=f'rgb({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)})')), 
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=y[idx + 1], name=f'C_B,{i+1}', 
                                line=dict(color=f'rgb({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)})')), 
                     row=3, col=1)
    
    for dist in disturbances:
        for row in range(1, 4):
            fig.add_vline(x=dist['time'], line_dash="dash", line_color="red", row=row, col=1)
    
    fig.update_xaxes(title_text="时间 (s)", row=3, col=1)
    fig.update_yaxes(title_text="温度 (K)", row=1, col=1)
    fig.update_yaxes(title_text="C_A (mol/m³)", row=2, col=1)
    fig.update_yaxes(title_text="C_B (mol/m³)", row=3, col=1)
    fig.update_layout(height=900, title_text="多级串联CSTR动态仿真结果")
    st.plotly_chart(fig, use_container_width=True)


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, output_min=0, output_max=10000, dt=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_min = output_min
        self.output_max = output_max
        self.dt = dt
        self.integral = 0
        self.prev_error = 0
        self.prev_time = 0
    
    def update(self, t, measurement):
        error = measurement - self.setpoint  # 反作用控制：T>SP时error正，增大UA冷却
        dt = max(t - self.prev_time, self.dt)
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, 
                               self.output_min / max(self.Ki, 1e-10) * 0.5,
                               self.output_max / max(self.Ki, 1e-10) * 0.5)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = np.clip(output, self.output_min, self.output_max)
        
        self.prev_error = error
        self.prev_time = t
        
        return output


def trapezoid_compat(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


def ziegler_nichols_tuning(params, reactions, y0):
    Ku = None
    Tu = None
    
    Kp_candidates = np.logspace(0.5, 2.5, 10)
    
    for Kp_test in Kp_candidates:
        pid = PIDController(Kp_test, 0, 0, params['T_f'] + 20, 
                           output_min=params['UA'] * 0.05, output_max=params['UA'] * 3)
        
        def pid_func(t, T):
            return pid.update(t, T)
        
        t_sim, y_sim = run_dynamic_cstr(params, reactions, y0, [0, 200], pid_func=pid_func)
        
        T_data = y_sim[5]
        T_final = T_data[-1]
        T_set = params['T_f'] + 20
        
        if len(T_data) > 30:
            T_std = np.std(T_data[-50:]) if len(T_data) > 50 else np.std(T_data[-20:])
            if T_std > 2.0 and T_final < 600:
                window = 15
                peaks = []
                start_i = max(window, len(T_data) // 3)
                for i in range(start_i, len(T_data) - window):
                    seg_before = T_data[i-window:i]
                    seg_after = T_data[i+1:i+window+1]
                    if len(seg_before) > 0 and len(seg_after) > 0:
                        if T_data[i] > seg_before.max() and T_data[i] > seg_after.max():
                            peaks.append(i)
                if len(peaks) >= 2:
                    Ku = Kp_test
                    Tu = np.mean(np.diff(t_sim[peaks]))
                    break
    
    if Ku is None:
        tau = params.get('tau', params['V'] / params['F'])
        steady_states = solve_cstr_steady_fast(params, reactions)
        if steady_states:
            T_ss = steady_states[0]['T']
            delta_T = params['T_f'] + 20 - T_ss
            if delta_T > 5:
                Kp = max(10.0, params['UA'] * 0.02 / delta_T * tau)
                Kp = min(Kp, params['UA'] * 0.15)
                Ki = Kp / (tau * 0.5)
                Kd = Kp * tau * 0.08
            else:
                Kp = params['UA'] * 0.03
                Ki = Kp / tau
                Kd = Kp * tau * 0.05
        else:
            Kp = params['UA'] * 0.05
            Ki = Kp / max(tau, 10.0)
            Kd = Kp * max(tau, 10.0) * 0.05
        return Kp, Ki, Kd
    
    Kp = 0.6 * Ku
    Ki = 2 * Kp / max(Tu, 1.0)
    Kd = Kp * Tu / 8
    
    return Kp, Ki, Kd


def page_temperature_control(reactor_type, params, reactions):
    st.header("🌡️ 温度控制模块")
    
    if "CSTR" not in reactor_type or "多级" in reactor_type:
        st.warning("温度控制模块目前仅适用于单级CSTR反应器")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("PID参数设置")
        auto_tune = st.checkbox("使用Ziegler-Nichols自动整定", True)
        
        if auto_tune:
            try:
                T_ss, C_ss = get_steady_temp_fast(params, reactions)
                y0 = np.zeros(6)
                y0[:5] = C_ss
                y0[5] = T_ss
            except:
                y0 = np.zeros(6)
                y0[:5] = params['C_f']
                y0[5] = params['T_f']
            Kp_auto, Ki_auto, Kd_auto = ziegler_nichols_tuning(params, reactions, y0)
            st.info(f"自动整定结果: Kp={Kp_auto:.2f}, Ki={Ki_auto:.2f}, Kd={Kd_auto:.3f}")
            Kp, Ki, Kd = Kp_auto, Ki_auto, Kd_auto
        else:
            Kp = st.number_input("比例系数 Kp", 0.0, 1000.0, 50.0, 1.0)
            Ki = st.number_input("积分系数 Ki", 0.0, 1000.0, 10.0, 0.1)
            Kd = st.number_input("微分系数 Kd", 0.0, 1000.0, 5.0, 0.1)
        
        st.subheader("控制设置")
        T_set = st.number_input("设定温度 (K)", 280.0, 500.0, params['T_f'] + 30, 1.0)
        T_step_time = st.number_input("设定值阶跃时间 (s)", 0.0, 500.0, 50.0, 1.0)
        T_step_value = st.number_input("阶跃后温度 (K)", 280.0, 500.0, T_set + 20, 1.0)
        t_final = st.number_input("仿真时间 (s)", 100.0, 2000.0, 500.0, 10.0)
    
    with col2:
        if st.button("🎯 开始控制仿真", type="primary", use_container_width=True):
            with st.spinner("正在进行控制仿真..."):
                try:
                    T_ss, C_ss = get_steady_temp_fast(params, reactions)
                    y0 = np.zeros(6)
                    y0[:5] = C_ss
                    y0[5] = T_ss
                except:
                    y0 = np.zeros(6)
                    y0[:5] = params['C_f']
                    y0[5] = params['T_f']
                
                pid = PIDController(Kp, Ki, Kd, T_set, 
                                   output_min=params['UA'] * 0.1, output_max=params['UA'] * 3)
                
                def pid_func(t, T):
                    if t >= T_step_time:
                        pid.setpoint = T_step_value
                    UA_val = pid.update(t, T)
                    return UA_val
                
                t, y = run_dynamic_cstr(params, reactions, y0, [0, t_final], pid_func=pid_func)
                
                T = y[5]
                setpoint_trace = np.where(t < T_step_time, T_set, T_step_value)
                
                UA_output = np.zeros_like(t)
                pid2 = PIDController(Kp, Ki, Kd, T_set, 
                                    output_min=params['UA'] * 0.1, output_max=params['UA'] * 3)
                for i, ti in enumerate(t):
                    if ti >= T_step_time:
                        pid2.setpoint = T_step_value
                    UA_output[i] = pid2.update(ti, T[i])
                
                error = T - setpoint_trace
                IAE = trapezoid_compat(np.abs(error), t)
                
                T_after_step = T[t >= T_step_time]
                t_after_step = t[t >= T_step_time]
                SP_after = T_step_value
                T_before_step = T[t < T_step_time]
                T_baseline = T_before_step[-1] if len(T_before_step) > 0 else params['T_f']
                
                step_size = SP_after - T_baseline
                if step_size > 0:
                    overshoot = (np.max(T_after_step) - SP_after) / max(step_size, 1e-6) * 100
                elif step_size < 0:
                    overshoot = (SP_after - np.min(T_after_step)) / max(abs(step_size), 1e-6) * 100
                else:
                    overshoot = 0.0
                overshoot = max(0.0, overshoot)
                
                target_band_abs = max(0.02 * abs(SP_after - T_baseline), 2.0)
                within_band = np.abs(T_after_step - SP_after) <= target_band_abs
                settling_time = None
                if len(within_band) > 0:
                    for i in range(len(within_band)):
                        if np.all(within_band[i:]):
                            settling_time = t_after_step[i] - T_step_time
                            break
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=('温度控制响应', '控制器输出 (UA)'))
                
                fig.add_trace(go.Scatter(x=t, y=T, name='反应器温度', line=dict(color='red', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=setpoint_trace, name='设定值', 
                                        line=dict(color='black', dash='dash')), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=t, y=UA_output, name='UA输出', 
                                        line=dict(color='blue', width=2)), row=2, col=1)
                
                fig.add_vline(x=T_step_time, line_dash="dash", line_color="gray", 
                             annotation_text="设定值阶跃", row=1, col=1)
                fig.add_vline(x=T_step_time, line_dash="dash", line_color="gray", row=2, col=1)
                
                fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
                fig.update_yaxes(title_text="温度 (K)", row=1, col=1)
                fig.update_yaxes(title_text="UA (W/K)", row=2, col=1)
                fig.update_layout(height=600, title_text="PID温度控制响应")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("控制性能指标")
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("超调量", f"{overshoot:.1f}%")
                with metric_cols[1]:
                    st.metric("调节时间", f"{settling_time:.1f}s" if settling_time else "未进入稳态")
                with metric_cols[2]:
                    st.metric("IAE积分绝对误差", f"{IAE:.1f}")


def check_thermal_runaway(params, reactions, T_f, T_c, UA):
    test_params = params.copy()
    test_params['T_f'] = T_f
    test_params['T_c'] = T_c
    test_params['UA'] = UA
    
    try:
        T_ss, C_ss = get_steady_temp_fast(test_params, reactions)
        if T_ss > 500 or T_ss > T_f + 200:
            return True
        V = test_params.get('V', 1.0)
        F = test_params.get('F', 1.0)
        tau = V / F
        rho_cp = test_params.get('rho_cp', 4.0e3)
        
        eps = 0.01
        T_hi = T_ss + eps
        C_hi = np.copy(test_params['C_f'])
        for _ in range(50):
            rates = reaction_rates(C_hi, T_hi, reactions)
            net = stoich_net(reactions)
            C_new = np.zeros(5)
            for j, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
                C_new[j] = test_params['C_f'][j] / (1 + tau * abs(net[comp]) * sum(rates) / max(test_params['C_f'][j], 1e-15))
                C_new[j] = max(C_new[j], 0)
            if np.max(np.abs(C_new - C_hi)) < 1e-6:
                break
            C_hi = C_new
        rates_hi = reaction_rates(C_hi, T_hi, reactions)
        Q_gen_hi = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates_hi)) * V
        Q_rem_hi = rho_cp * F * (T_hi - T_f) + UA * (T_hi - T_c)
        
        T_lo = T_ss - eps
        C_lo = np.copy(test_params['C_f'])
        for _ in range(50):
            rates = reaction_rates(C_lo, T_lo, reactions)
            net = stoich_net(reactions)
            C_new = np.zeros(5)
            for j, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
                C_new[j] = test_params['C_f'][j] / (1 + tau * abs(net[comp]) * sum(rates) / max(test_params['C_f'][j], 1e-15))
                C_new[j] = max(C_new[j], 0)
            if np.max(np.abs(C_new - C_lo)) < 1e-6:
                break
            C_lo = C_new
        rates_lo = reaction_rates(C_lo, T_lo, reactions)
        Q_gen_lo = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates_lo)) * V
        Q_rem_lo = rho_cp * F * (T_lo - T_f) + UA * (T_lo - T_c)
        
        dQgen_dT = (Q_gen_hi - Q_gen_lo) / (2 * eps)
        dQrem_dT = (Q_rem_hi - Q_rem_lo) / (2 * eps)
        
        if dQgen_dT > dQrem_dT and T_ss > T_f + 30:
            return True
        return False
    except:
        return False


def find_runaway_boundary(params, reactions):
    N_Tc = 15
    N_Cf = 15
    T_c_range = np.linspace(265, 335, N_Tc)
    C_f_range = np.linspace(20, 450, N_Cf)
    
    Tc_grid, Cf_grid = np.meshgrid(T_c_range, C_f_range)
    T_ss_grid = np.zeros_like(Tc_grid)
    runaway_grid = np.zeros_like(Tc_grid, dtype=bool)
    
    V = params.get('V', 1.0)
    F = params.get('F', 1.0)
    tau = V / F
    rho_cp = params.get('rho_cp', 4.0e3)
    T_f = params['T_f']
    
    def quick_check_runaway(T_c, C_f0):
        C_f_test = np.zeros(5)
        C_f_test[0] = C_f0
        T_test = 450.0
        C = C_f_test.copy()
        for _ in range(15):
            rates = reaction_rates(C, T_test, reactions)
            net = stoich_net(reactions)
            C_new = np.zeros(5)
            for j, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
                C_new[j] = C_f_test[j] / (1 + tau * abs(net[comp]) * sum(rates) / max(C_f_test[j], 1e-15))
                C_new[j] = max(C_new[j], 0)
            if np.max(np.abs(C_new - C)) < 1e-4:
                break
            C = C_new
        rates = reaction_rates(C, T_test, reactions)
        Q_gen = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V
        Q_rem = rho_cp * F * (T_test - T_f) + params['UA'] * (T_test - T_c)
        return Q_gen > Q_rem
    
    for i in range(N_Cf):
        for j in range(N_Tc):
            test_params = params.copy()
            test_params['T_c'] = Tc_grid[i, j]
            test_params['C_f'] = np.zeros(5)
            test_params['C_f'][0] = Cf_grid[i, j]
            
            try:
                if quick_check_runaway(Tc_grid[i, j], Cf_grid[i, j]):
                    runaway_grid[i, j] = True
                    T_ss_grid[i, j] = 500.0
                else:
                    T_ss, C_ss = get_steady_temp_fast(test_params, reactions)
                    T_ss_grid[i, j] = T_ss
                    if T_ss > 490:
                        runaway_grid[i, j] = True
            except:
                T_ss_grid[i, j] = np.nan
                runaway_grid[i, j] = True
    
    return T_c_range, C_f_range, T_ss_grid, runaway_grid


def page_safety_analysis(reactor_type, params, reactions):
    st.header("⚠️ 安全分析模块 - 飞温检测")
    
    if "CSTR" not in reactor_type or "多级" in reactor_type:
        st.warning("安全分析模块目前仅适用于单级CSTR反应器")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("飞温检测")
        st.info("Semenov判据: 热产生曲线与热移除直线相切的条件为临界点")
        
        T_f_test = st.slider("测试进料温度 T_f (K)", 250.0, 400.0, params['T_f'], 5.0)
        UA_test = st.slider("测试冷却强度 UA (W/K)", 100.0, 5000.0, params['UA'], 100.0)
        
        if st.button("🔍 检测飞温临界条件"):
            T_c_lo, T_c_hi = 260.0, 340.0
            runaway_lo = check_thermal_runaway(params, reactions, T_f_test, T_c_lo, UA_test)
            runaway_hi = check_thermal_runaway(params, reactions, T_f_test, T_c_hi, UA_test)
            
            T_c_critical = None
            if runaway_lo != runaway_hi:
                for _ in range(20):
                    T_c_mid = (T_c_lo + T_c_hi) / 2
                    runaway_mid = check_thermal_runaway(params, reactions, T_f_test, T_c_mid, UA_test)
                    if runaway_mid == runaway_lo:
                        T_c_lo = T_c_mid
                        runaway_lo = runaway_mid
                    else:
                        T_c_hi = T_c_mid
                        runaway_hi = runaway_mid
                T_c_critical = (T_c_lo + T_c_hi) / 2
                
                if check_thermal_runaway(params, reactions, T_f_test, 260.0, UA_test):
                    st.warning(f"⚠️ 临界冷却水温度: {T_c_critical:.1f} K")
                    st.info(f"当冷却水温度低于 {T_c_critical:.1f} K 时，可能发生飞温！")
                else:
                    st.warning(f"⚠️ 临界冷却水温度: {T_c_critical:.1f} K")
                    st.info(f"当冷却水温度高于 {T_c_critical:.1f} K 时，可能发生飞温！")
            else:
                if runaway_lo:
                    st.warning("⚠️ 整个冷却温度范围内均存在飞温风险！")
                else:
                    st.success("✅ 在测试条件下未检测到飞温风险")
    
    with col2:
        if st.button("🗺️ 生成安全边界图", type="primary"):
            with st.spinner("正在计算安全边界..."):
                T_c_range, C_f_range, T_ss_grid, runaway_grid = find_runaway_boundary(params, reactions)
                
                fig, ax = plt.subplots(figsize=(10, 7))
                
                contour = ax.contourf(T_c_range, C_f_range, T_ss_grid, 
                                     levels=np.linspace(280, 500, 15), cmap='RdYlBu_r', alpha=0.8)
                cbar = plt.colorbar(contour, ax=ax)
                cbar.set_label('稳态操作温度 (K)')
                
                ax.contour(T_c_range, C_f_range, runaway_grid.astype(int), 
                          levels=[0.5], colors='red', linestyles='--', linewidths=3)
                
                ax.set_xlabel('冷却水温度 T_c (K)', fontsize=12)
                ax.set_ylabel('进料浓度 C_A,f (mol/m³)', fontsize=12)
                ax.set_title('飞温安全边界图', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', alpha=0.3, label='危险区 (飞温)'),
                    Patch(facecolor='blue', alpha=0.3, label='安全区'),
                    plt.Line2D([], [], color='red', linestyle='--', label='飞温边界')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                st.pyplot(fig)


def calculate_performance(params, reactions, T_f, tau, T_c, UA, n_stages=1):
    if n_stages == 1:
        test_params = params.copy()
        test_params['T_f'] = T_f
        test_params['T_c'] = T_c
        test_params['UA'] = UA
        test_params['tau'] = tau
        test_params['V'] = test_params['F'] * tau
        
        steady_states = solve_cstr_steady_fast(test_params, reactions)
        if not steady_states:
            T_out, C_out = get_steady_temp_fast(test_params, reactions)
        else:
            best_ss = max(steady_states, key=lambda x: x['C'][1] if len(x['C']) > 1 else x['T'])
            C_out = best_ss['C']
            T_out = best_ss['T']
    else:
        y0 = np.zeros(6 * n_stages)
        for i in range(n_stages):
            idx = i * 6
            y0[idx:idx+5] = params['C_f']
            y0[idx+5] = T_f[i] if isinstance(T_f, list) else T_f
        
        test_params = params.copy()
        if isinstance(T_f, list):
            for i in range(n_stages):
                test_params[f'T_c_{i}'] = T_c[i] if isinstance(T_c, list) else T_c
                test_params[f'V_{i}'] = tau[i] * test_params['F'] if isinstance(tau, list) else tau * test_params['F']
        
        t, y = run_dynamic_multicstr(test_params, reactions, y0, [0, 300], n_stages)
        final_idx = -1
        C_out = y[(n_stages-1)*6:(n_stages-1)*6+5, final_idx]
        T_out = y[(n_stages-1)*6+5, final_idx]
    
    C_f0 = params['C_f'][0]
    conversion = (C_f0 - C_out[0]) / max(C_f0, 1e-10)
    selectivity = C_out[1] / max(C_f0 - C_out[0], 1e-10) if len(reactions) >= 2 else 1.0
    yield_CB = C_out[1] / max(C_f0, 1e-10)
    
    Q_cooling = UA * (T_out - T_c) if not isinstance(T_c, list) else UA[0] * (T_out - T_c[0])
    energy_consumption = abs(Q_cooling) / 1000
    
    T_safe = 473.15
    safety_margin = (T_safe - T_out) / T_safe * 100
    
    return {
        'conversion': conversion,
        'selectivity': selectivity,
        'yield': yield_CB,
        'energy': energy_consumption,
        'safety': safety_margin,
        'T_out': T_out,
        'C_out': C_out
    }


def page_optimization(reactor_type, params, reactions):
    st.header("📈 优化求解模块")
    
    if "CSTR" not in reactor_type:
        st.warning("优化模块目前仅适用于CSTR类反应器")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("优化设置")
        
        objective = st.selectbox("优化目标", 
                                ["最大化主产物收率", "最小化副产物生成量", "最大化转化率"],
                                index=0)
        
        st.subheader("约束条件")
        T_max = st.number_input("最高允许温度 (K)", 350.0, 600.0, 473.15, 1.0)
        X_min = st.slider("最低转化率要求 (%)", 0.0, 95.0, 30.0, 1.0)
        UA_max = st.number_input("最大冷却水用量 (UA上限, W/K)", 500.0, 20000.0, params['UA'] * 2, 100.0)
        
        st.subheader("优化变量范围")
        T_f_bounds = st.slider("进料温度范围 (K)", 250.0, 400.0, (280.0, 380.0), 1.0)
        tau_bounds = st.slider("停留时间范围 (s)", 10.0, 500.0, (50.0, 300.0), 1.0)
        T_c_bounds = st.slider("冷却水温度范围 (K)", 250.0, 350.0, (260.0, 330.0), 1.0)
    
    with col2:
        n_stages = params.get('n_stages', 1) if "多级" in reactor_type else 1
        
        if st.button("⚡ 开始优化", type="primary", use_container_width=True):
            with st.spinner("正在进行优化求解..."):
                base_perf = calculate_performance(params, reactions, params['T_f'], 
                                                 params.get('tau', params['V']/params['F']), 
                                                 params['T_c'], params['UA'], n_stages)
                
                def objective_func(x):
                    if n_stages == 1:
                        T_f, tau, T_c = x
                        UA = params['UA']
                    else:
                        T_f = x[0]
                        tau = x[1:n_stages+1]
                        T_c = x[n_stages+1:2*n_stages+1]
                        UA = params['UA']
                    
                    perf = calculate_performance(params, reactions, T_f, tau, T_c, UA, n_stages)
                    if perf is None:
                        return 1e6
                    
                    if perf['T_out'] > T_max:
                        return 1e6 + (perf['T_out'] - T_max) * 100
                    if perf['conversion'] < X_min / 100:
                        return 1e6 + (X_min / 100 - perf['conversion']) * 1000
                    
                    if objective == "最大化主产物收率":
                        return -perf['yield']
                    elif objective == "最小化副产物生成量":
                        return perf['C_out'][2] if len(perf['C_out']) > 2 else -perf['yield']
                    else:
                        return -perf['conversion']
                
                if n_stages == 1:
                    bounds = [T_f_bounds, tau_bounds, T_c_bounds]
                    x0 = [params['T_f'], params.get('tau', 100), params['T_c']]
                    
                    if st.checkbox("启用初始点网格搜索（更准但更慢）", False):
                        n_grid = 5
                        T_f_grid = np.linspace(*T_f_bounds, n_grid)
                        tau_grid = np.linspace(*tau_bounds, n_grid)
                        best_obj = 1e10
                        for Tf in T_f_grid:
                            for ta in tau_grid:
                                obj_val = objective_func([Tf, ta, params['T_c']])
                                if obj_val < best_obj:
                                    best_obj = obj_val
                                    x0 = [Tf, ta, params['T_c']]
                else:
                    bounds = [T_f_bounds]
                    x0 = [params['T_f']]
                    for i in range(n_stages):
                        bounds.append(tau_bounds)
                        x0.append(params.get(f'V_{i}', 1.0) / params['F'])
                    for i in range(n_stages):
                        bounds.append(T_c_bounds)
                        x0.append(params.get(f'T_c_{i}', params['T_c']))
                
                result = minimize(objective_func, x0, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 40, 'ftol': 1e-5, 'maxfun': 80, 'eps': 1e-3})
                
                if n_stages == 1:
                    T_f_opt, tau_opt, T_c_opt = result.x
                    opt_perf = calculate_performance(params, reactions, T_f_opt, tau_opt, 
                                                    T_c_opt, params['UA'])
                else:
                    T_f_opt = result.x[0]
                    tau_opt = result.x[1:n_stages+1]
                    T_c_opt = result.x[n_stages+1:2*n_stages+1]
                    opt_perf = calculate_performance(params, reactions, T_f_opt, tau_opt,
                                                    T_c_opt, params['UA'], n_stages)
                
                st.subheader("优化结果")
                if n_stages == 1:
                    opt_data = [
                        {'参数': '进料温度 T_f', '优化前': f"{params['T_f']:.1f} K", '优化后': f"{T_f_opt:.1f} K"},
                        {'参数': '停留时间 τ', '优化前': f"{params.get('tau', 100):.1f} s", '优化后': f"{tau_opt:.1f} s"},
                        {'参数': '冷却水温度 T_c', '优化前': f"{params['T_c']:.1f} K", '优化后': f"{T_c_opt:.1f} K"},
                        {'参数': '反应器温度', '优化前': f"{base_perf['T_out']:.1f} K", '优化后': f"{opt_perf['T_out']:.1f} K"},
                    ]
                else:
                    opt_data = [{'参数': '进料温度 T_f', '优化前': f"{params['T_f']:.1f} K", '优化后': f"{T_f_opt:.1f} K"}]
                    for i in range(n_stages):
                        opt_data.append({
                            '参数': f'第{i+1}级停留时间', 
                            '优化前': f"{params.get(f'V_{i}', 1.0)/params['F']:.1f} s", 
                            '优化后': f"{tau_opt[i]:.1f} s"
                        })
                        opt_data.append({
                            '参数': f'第{i+1}级冷却水温', 
                            '优化前': f"{params.get(f'T_c_{i}', params['T_c']):.1f} K", 
                            '优化后': f"{T_c_opt[i]:.1f} K"
                        })
                
                st.table(opt_data)
                
                st.subheader("性能指标对比")
                metrics_data = [
                    {'指标': '转化率 (%)', 
                     '优化前': f"{base_perf['conversion']*100:.1f}", 
                     '优化后': f"{opt_perf['conversion']*100:.1f}",
                     '变化': f"{(opt_perf['conversion']-base_perf['conversion'])*100:+.1f}"},
                    {'指标': '选择性 (%)', 
                     '优化前': f"{base_perf['selectivity']*100:.1f}", 
                     '优化后': f"{opt_perf['selectivity']*100:.1f}",
                     '变化': f"{(opt_perf['selectivity']-base_perf['selectivity'])*100:+.1f}"},
                    {'指标': '收率 (%)', 
                     '优化前': f"{base_perf['yield']*100:.1f}", 
                     '优化后': f"{opt_perf['yield']*100:.1f}",
                     '变化': f"{(opt_perf['yield']-base_perf['yield'])*100:+.1f}"},
                    {'指标': '能耗 (kW)', 
                     '优化前': f"{base_perf['energy']:.1f}", 
                     '优化后': f"{opt_perf['energy']:.1f}",
                     '变化': f"{(opt_perf['energy']-base_perf['energy']):+.1f}"},
                    {'指标': '安全裕度 (%)', 
                     '优化前': f"{base_perf['safety']:.1f}", 
                     '优化后': f"{opt_perf['safety']:.1f}",
                     '变化': f"{(opt_perf['safety']-base_perf['safety']):+.1f}"},
                ]
                st.table(metrics_data)
                
                categories = ['转化率', '选择性', '收率', '安全裕度']
                base_vals = [base_perf['conversion']*100, base_perf['selectivity']*100, 
                            base_perf['yield']*100, base_perf['safety']]
                opt_vals = [opt_perf['conversion']*100, opt_perf['selectivity']*100, 
                           opt_perf['yield']*100, opt_perf['safety']]
                
                base_vals_norm = [v / max(base_vals[i], opt_vals[i], 1) * 100 for i, v in enumerate(base_vals)]
                opt_vals_norm = [v / max(base_vals[i], opt_vals[i], 1) * 100 for i, v in enumerate(opt_vals)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=base_vals_norm, theta=categories, fill='toself', name='优化前',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatterpolar(
                    r=opt_vals_norm, theta=categories, fill='toself', name='优化后',
                    line=dict(color='red')
                ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                 title="优化前后性能对比雷达图", height=500)
                st.plotly_chart(fig, use_container_width=True)


def page_sensitivity_analysis(reactor_type, params, reactions):
    st.header("📊 参数灵敏度分析")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("灵敏度设置")
        
        input_params = [
            ('活化能 Ea', 'Ea'),
            ('频率因子 A', 'A'),
            ('反应热 ΔH', 'dH'),
            ('冷却系数 UA', 'UA'),
            ('进料温度 T_f', 'T_f'),
            ('进料浓度 C_A,f', 'C_f'),
            ('停留时间 τ', 'tau'),
        ]
        
        input_param = st.selectbox("选择输入参数", input_params, 
                                  format_func=lambda x: x[0])
        
        output_metrics = [
            ('转化率', 'conversion'),
            ('最高温度', 'T_max'),
            ('主产物收率', 'yield'),
            ('选择性', 'selectivity'),
        ]
        
        output_metric = st.selectbox("选择输出指标", output_metrics,
                                    format_func=lambda x: x[0])
        
        param_range = st.slider("参数变化范围 (%)", 10, 100, 50, 5)
        n_points = 20
        
        if st.button("📐 计算灵敏度", type="primary", use_container_width=True):
            with st.spinner("正在计算灵敏度..."):
                param_key = input_param[1]
                metric_key = output_metric[1]
                
                if param_key in ['Ea', 'A', 'dH']:
                    base_val = reactions[0][param_key]
                    if param_key == 'A':
                        vals = np.logspace(np.log10(base_val * (1 - param_range/100)), 
                                          np.log10(base_val * (1 + param_range/100)), n_points)
                    else:
                        vals = np.linspace(base_val * (1 - param_range/100), 
                                          base_val * (1 + param_range/100), n_points)
                elif param_key == 'tau':
                    base_val = params.get('tau', params['V'] / params['F'])
                    vals = np.linspace(base_val * (1 - param_range/100), 
                                      base_val * (1 + param_range/100), n_points)
                elif param_key == 'C_f':
                    base_val = params['C_f'][0]
                    vals = np.linspace(base_val * (1 - param_range/100), 
                                      base_val * (1 + param_range/100), n_points)
                else:
                    base_val = params[param_key]
                    vals = np.linspace(base_val * (1 - param_range/100), 
                                      base_val * (1 + param_range/100), n_points)
                
                outputs = []
                for val in vals:
                    test_params = params.copy()
                    test_reactions = [r.copy() for r in reactions]
                    
                    if param_key in ['Ea', 'A', 'dH']:
                        for i_rxn in range(len(test_reactions)):
                            test_reactions[i_rxn][param_key] = val
                    elif param_key == 'tau':
                        test_params['V'] = test_params['F'] * val
                        test_params['tau'] = val
                    elif param_key == 'C_f':
                        test_params['C_f'] = np.array(test_params['C_f'], copy=True)
                        test_params['C_f'][0] = val
                    else:
                        test_params[param_key] = val
                    
                    if "CSTR" in reactor_type and "多级" not in reactor_type:
                        perf = calculate_performance(test_params, test_reactions, 
                                                   test_params['T_f'], 
                                                   test_params.get('tau', test_params['V']/test_params['F']),
                                                   test_params['T_c'], test_params['UA'])
                        if perf is not None:
                            if metric_key == 'conversion':
                                outputs.append(perf['conversion'])
                            elif metric_key == 'T_max':
                                outputs.append(perf['T_out'])
                            elif metric_key == 'yield':
                                outputs.append(perf['yield'])
                            elif metric_key == 'selectivity':
                                outputs.append(perf['selectivity'])
                            continue
                    outputs.append(np.nan)
                
                outputs = np.array(outputs)
                valid_mask = ~np.isnan(outputs)
                vals_valid = vals[valid_mask]
                outputs_valid = outputs[valid_mask]
                
                if len(vals_valid) < 2:
                    st.error("无法计算灵敏度，请检查参数范围")
                    return
                
                S = np.zeros_like(vals_valid)
                for i in range(1, len(vals_valid)):
                    delta_p = (vals_valid[i] - base_val) / base_val * 100
                    delta_o = (outputs_valid[i] - outputs_valid[0]) / outputs_valid[0] * 100
                    if abs(delta_p) > 1e-6:
                        S[i] = delta_o / delta_p
                    else:
                        S[i] = 0
                S[0] = S[1] if len(S) > 1 else 0
                
                sensitive_mask = np.abs(S) > 2
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                ax1.plot(vals_valid, outputs_valid, 'b-o', linewidth=2, markersize=4)
                ax1.axvline(base_val, color='red', linestyle='--', label=f'基准值: {base_val:.2e}')
                
                if np.any(sensitive_mask):
                    ax1.fill_between(vals_valid, outputs_valid.min(), outputs_valid.max(),
                                    where=sensitive_mask, color='red', alpha=0.2, label='参数敏感区')
                
                ax1.set_xlabel(input_param[0], fontsize=12)
                ax1.set_ylabel(output_metric[0], fontsize=12)
                ax1.set_title(f'{output_metric[0]} 随 {input_param[0]} 的变化', fontsize=14)
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(vals_valid, S, 'g-o', linewidth=2, markersize=4)
                ax2.axhline(2, color='red', linestyle='--', label='|S| > 2 阈值')
                ax2.axhline(-2, color='red', linestyle='--')
                ax2.axvline(base_val, color='red', linestyle='--')
                ax2.fill_between(vals_valid, -2, 2, where=np.abs(S) > 2, 
                                color='red', alpha=0.2, interpolate=True)
                
                ax2.set_xlabel(input_param[0], fontsize=12)
                ax2.set_ylabel('归一化灵敏度系数 S', fontsize=12)
                ax2.set_title('归一化灵敏度系数曲线', fontsize=14)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("灵敏度分析结果")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("最大灵敏度", f"{np.max(np.abs(S)):.2f}")
                with col_s2:
                    st.metric("平均灵敏度", f"{np.mean(np.abs(S)):.2f}")
                with col_s3:
                    st.metric("敏感区占比", f"{np.sum(sensitive_mask)/len(sensitive_mask)*100:.1f}%")
                
                if np.any(sensitive_mask):
                    st.warning(f"⚠️ 检测到参数敏感区！当 {input_param[0]} 在 "
                              f"{vals_valid[sensitive_mask].min():.2e} ~ {vals_valid[sensitive_mask].max():.2e} 范围内时，"
                              f"{output_metric[0]} 对参数变化高度敏感。")


def main():
    if 'reactions' in st.session_state:
        for rxn in st.session_state.reactions:
            rxn['reactant_orders'] = [int(o) for o in rxn['reactant_orders']]
    
    st.title("🏭 化工反应器动力学建模与温度控制优化工具")
    st.markdown("---")
    
    reactor_type, params = sidebar_reactor_config()
    reactions = sidebar_reaction_config()
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio("📋 功能模块", 
        ["📊 稳态分析", "⏱️ 动态仿真", "🌡️ 温度控制", 
         "⚠️ 安全分析", "📈 优化求解", "📊 参数灵敏度"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("面向化工工艺工程师的专业仿真工具")
    
    if page == "📊 稳态分析":
        page_steady_state_analysis(reactor_type, params, reactions)
    elif page == "⏱️ 动态仿真":
        page_dynamic_simulation(reactor_type, params, reactions)
    elif page == "🌡️ 温度控制":
        page_temperature_control(reactor_type, params, reactions)
    elif page == "⚠️ 安全分析":
        page_safety_analysis(reactor_type, params, reactions)
    elif page == "📈 优化求解":
        page_optimization(reactor_type, params, reactions)
    elif page == "📊 参数灵敏度":
        page_sensitivity_analysis(reactor_type, params, reactions)


if __name__ == "__main__":
    main()


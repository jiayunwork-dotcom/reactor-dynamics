import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, fsolve
from scipy.integrate import solve_ivp
import pandas as pd
from datetime import datetime
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
from reactor_network import (
    ReactorNetwork, ReactorNode, Connection,
    REACTOR_TYPES, REACTOR_TYPE_CN, COMPONENTS
)
from parameter_estimation import (
    fit_parameters,
    generate_validation_data,
    compute_confidence_ellipse,
    compute_sensitivity_curves,
    validate_experimental_data,
    get_cell_error_mask,
    model_predict
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


def calculate_key_metrics(t, y, params, reactor_type, n_stages=None):
    if reactor_type == "连续搅拌釜式反应器 (CSTR)":
        T_idx = 5
        C_A_idx = 0
        T = y[T_idx]
        C_A = y[C_A_idx]
    elif reactor_type == "管式反应器 (PFR)":
        n_points = 50
        outlet_idx = (n_points - 1) * 6
        T = y[outlet_idx + 5]
        C_A = y[outlet_idx]
    elif reactor_type == "半间歇反应器":
        T = y[5]
        C_A = y[0]
    else:
        n_stages = n_stages or params.get('n_stages', 1)
        last_stage_idx = (n_stages - 1) * 6
        T = y[last_stage_idx + 5]
        C_A = y[last_stage_idx]

    C_f0 = params['C_f'][0]
    conversion = (1 - C_A[-1] / max(C_f0, 1e-10)) * 100 if C_f0 > 0 else 0

    T_max = np.max(T)

    if len(T) > 20:
        T_final = T[-1]
        tol = max(abs(T_final) * 0.005, 0.5)
        steady_idx = None
        for i in range(len(T) - 10, -1, -1):
            if np.max(np.abs(T[i:i+10] - T_final)) <= tol:
                steady_idx = i
            else:
                break
        settling_time = t[steady_idx] if steady_idx is not None else t[-1]
    else:
        settling_time = t[-1]

    avg_rates = []
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        dC = C_A[i+1] - C_A[i]
        if dt > 0:
            rate = -dC / dt
            avg_rates.append(rate)
    avg_reaction_rate = np.mean(avg_rates) if avg_rates else 0

    return {
        'final_conversion': float(conversion),
        'max_temperature': float(T_max),
        'settling_time': float(settling_time),
        'avg_reaction_rate': float(avg_reaction_rate)
    }


def init_saved_simulations():
    if 'saved_simulations' not in st.session_state:
        st.session_state.saved_simulations = []


def save_simulation(label, t, y, params, reactor_type, metrics, n_stages=None):
    init_saved_simulations()
    if len(st.session_state.saved_simulations) >= 5:
        st.warning("已达到最大保存数量（5组），请先删除部分记录。")
        return False

    params_simple = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            params_simple[k] = v.tolist()
        else:
            params_simple[k] = v

    sim_data = {
        'id': datetime.now().timestamp(),
        'label': label,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        't': t.tolist(),
        'y': y.tolist(),
        'params': params_simple,
        'reactor_type': reactor_type,
        'metrics': metrics,
        'n_stages': n_stages
    }

    st.session_state.saved_simulations.append(sim_data)
    return True


def delete_simulation(sim_id):
    init_saved_simulations()
    st.session_state.saved_simulations = [
        s for s in st.session_state.saved_simulations if s['id'] != sim_id
    ]


def get_temperature_data(sim_data):
    reactor_type = sim_data['reactor_type']
    y = np.array(sim_data['y'])
    t = np.array(sim_data['t'])

    if reactor_type == "连续搅拌釜式反应器 (CSTR)":
        T = y[5]
    elif reactor_type == "管式反应器 (PFR)":
        n_points = 50
        outlet_idx = (n_points - 1) * 6
        T = y[outlet_idx + 5]
    elif reactor_type == "半间歇反应器":
        T = y[5]
    else:
        n_stages = sim_data.get('n_stages', 1)
        last_stage_idx = (n_stages - 1) * 6
        T = y[last_stage_idx + 5]

    return t, T


def export_comparison_csv():
    init_saved_simulations()
    sims = st.session_state.saved_simulations

    if not sims:
        return None

    params_data = []
    metrics_data = []

    for sim in sims:
        params_row = {'标签': sim['label'], '保存时间': sim['timestamp']}
        for k, v in sim['params'].items():
            if isinstance(v, list):
                params_row[k] = str(v)
            else:
                params_row[k] = v
        params_data.append(params_row)

        metrics_row = {
            '标签': sim['label'],
            '保存时间': sim['timestamp'],
            '最终转化率 (%)': f"{sim['metrics']['final_conversion']:.2f}",
            '最高温度 (K)': f"{sim['metrics']['max_temperature']:.2f}",
            '稳态时间 (s)': f"{sim['metrics']['settling_time']:.2f}",
            '平均反应速率 (mol/m³/s)': f"{sim['metrics']['avg_reaction_rate']:.4f}"
        }
        metrics_data.append(metrics_row)

    params_df = pd.DataFrame(params_data)
    metrics_df = pd.DataFrame(metrics_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_comparison_{timestamp}.csv"

    csv_content = "=== 参数汇总表 ===\n"
    csv_content += params_df.to_csv(index=False, encoding='utf-8-sig')
    csv_content += "\n=== 指标对比表 ===\n"
    csv_content += metrics_df.to_csv(index=False, encoding='utf-8-sig')

    return csv_content, filename


BATCH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def page_batch_comparison():
    st.header("📊 批次对比分析")
    init_saved_simulations()
    sims = st.session_state.saved_simulations

    if not sims:
        st.info("暂无保存的仿真结果。请先运行动态仿真并点击\"保存当前仿真结果\"按钮。")
        return

    col_export, col_clear = st.columns([1, 1])
    with col_export:
        csv_result = export_comparison_csv()
        if csv_result:
            csv_content, filename = csv_result
            st.download_button(
                label="📥 导出对比报告 (CSV)",
                data=csv_content.encode('utf-8-sig'),
                file_name=filename,
                mime='text/csv',
                use_container_width=True
            )

    with col_clear:
        if st.button("🗑️ 清空所有记录", use_container_width=True):
            st.session_state.saved_simulations = []
            st.rerun()

    st.subheader("📋 关键指标对比")

    table_data = []
    for i, sim in enumerate(sims):
        color = BATCH_COLORS[i % len(BATCH_COLORS)]
        row = {
            '序号': i + 1,
            '颜色': f'<span style="color:{color}; font-size: 20px;">●</span>',
            '标签': sim['label'],
            '保存时间': sim['timestamp'],
            '最终转化率 (%)': f"{sim['metrics']['final_conversion']:.2f}",
            '最高温度 (K)': f"{sim['metrics']['max_temperature']:.2f}",
            '稳态时间 (s)': f"{sim['metrics']['settling_time']:.2f}",
            '平均反应速率 (mol/m³/s)': f"{sim['metrics']['avg_reaction_rate']:.4f}"
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    col_table, col_actions = st.columns([4, 1])
    with col_table:
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    with col_actions:
        st.markdown("**操作**")
        for i, sim in enumerate(sims):
            if st.button(f"删除 #{i+1}", key=f"del_{sim['id']}"):
                delete_simulation(sim['id'])
                st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 温度-时间曲线对比")
        fig_temp = go.Figure()
        for i, sim in enumerate(sims):
            t, T = get_temperature_data(sim)
            color = BATCH_COLORS[i % len(BATCH_COLORS)]
            fig_temp.add_trace(go.Scatter(
                x=t, y=T,
                name=sim['label'],
                line=dict(color=color, width=2),
                mode='lines'
            ))

        fig_temp.update_layout(
            xaxis_title='时间 (s)',
            yaxis_title='温度 (K)',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        st.subheader("📊 转化率与最高温度对比")
        fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

        labels = [sim['label'] for sim in sims]
        conversions = [sim['metrics']['final_conversion'] for sim in sims]
        max_temps = [sim['metrics']['max_temperature'] for sim in sims]
        bar_colors = [BATCH_COLORS[i % len(BATCH_COLORS)] for i in range(len(sims))]

        fig_bar.add_trace(go.Bar(
            x=labels, y=conversions,
            name='最终转化率 (%)',
            marker_color=bar_colors,
            opacity=0.7,
            text=[f"{v:.1f}%" for v in conversions],
            textposition='auto'
        ), secondary_y=False)

        fig_bar.add_trace(go.Bar(
            x=labels, y=max_temps,
            name='最高温度 (K)',
            marker_color=bar_colors,
            opacity=0.4,
            text=[f"{v:.1f}K" for v in max_temps],
            textposition='auto'
        ), secondary_y=True)

        fig_bar.update_layout(
            barmode='group',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig_bar.update_yaxes(title_text='转化率 (%)', secondary_y=False)
        fig_bar.update_yaxes(title_text='最高温度 (K)', secondary_y=True)
        st.plotly_chart(fig_bar, use_container_width=True)


def page_dynamic_simulation(reactor_type, params, reactions):
    st.header("⏱️ 动态仿真分析")

    init_saved_simulations()
    if 'last_simulation' not in st.session_state:
        st.session_state.last_simulation = None

    tab_single, tab_batch = st.tabs(["🔬 单次仿真", "📊 批次对比"])

    with tab_single:
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
                        n_stages = None
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
                        n_stages = None
                        plot_pfr_dynamic_results(t, y, params, n_points)

                    elif reactor_type == "半间歇反应器":
                        y0 = np.zeros(7)
                        y0[:5] = C0
                        y0[5] = T0
                        y0[6] = V0 if 'V0' in locals() else 0.5
                        t, y = run_dynamic_semibatch(params, reactions, y0, [0, t_final], disturbances)
                        n_stages = None
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

                    metrics = calculate_key_metrics(t, y, params, reactor_type, n_stages)
                    st.session_state.last_simulation = {
                        't': t,
                        'y': y,
                        'params': params,
                        'reactor_type': reactor_type,
                        'metrics': metrics,
                        'n_stages': n_stages
                    }

            if st.session_state.last_simulation is not None:
                st.markdown("---")
                col_save1, col_save2 = st.columns([2, 1])
                with col_save1:
                    default_label = f"工况{len(st.session_state.saved_simulations) + 1}"
                    sim_label = st.text_input("仿真标签", value=default_label,
                                             placeholder="如：基准工况、高温进料等",
                                             max_chars=20)
                with col_save2:
                    saved_count = len(st.session_state.saved_simulations)
                    st.info(f"已保存: {saved_count}/5")
                    if st.button("💾 保存当前仿真结果", type="secondary", use_container_width=True):
                        if not sim_label.strip():
                            st.error("请输入仿真标签！")
                        else:
                            last_sim = st.session_state.last_simulation
                            success = save_simulation(
                                sim_label.strip(),
                                last_sim['t'],
                                last_sim['y'],
                                last_sim['params'],
                                last_sim['reactor_type'],
                                last_sim['metrics'],
                                last_sim['n_stages']
                            )
                            if success:
                                st.success(f"✅ 已保存 \"{sim_label.strip()}\"")
                                st.rerun()

    with tab_batch:
        page_batch_comparison()


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


def page_parameter_estimation():
    st.header("🔬 实验数据拟合与动力学参数估计")

    if 'fit_data' not in st.session_state:
        st.session_state.fit_data = pd.DataFrame({
            'T': [350.0, 370.0, 390.0, 410.0, 430.0, 450.0],
            'tau': [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
            'CA': [70.0, 55.0, 40.0, 28.0, 18.0, 12.0],
            'CB': [30.0, 45.0, 60.0, 72.0, 82.0, 88.0]
        })

    if 'fit_result' not in st.session_state:
        st.session_state.fit_result = None

    if 'CAf_fit' not in st.session_state:
        st.session_state.CAf_fit = 100.0

    col_input, col_config = st.columns([1.2, 1])

    with col_input:
        st.subheader("📊 实验数据输入")

        CAf_fit = st.number_input("进料A浓度 CAf (mol/m³)", 0.0, 2000.0, st.session_state.CAf_fit, 1.0)
        st.session_state.CAf_fit = CAf_fit
        Tf_fit = st.number_input("进料温度 Tf (K)", 200.0, 600.0, 300.0, 1.0,
                                 help="绝热CSTR能量平衡计算所需的进料温度，仅在勾选拟合反应热时使用")

        col_ds1, col_ds2, col_ds3 = st.columns(3)
        with col_ds1:
            if st.button("📦 理想一级反应(等温)", use_container_width=True):
                data = generate_validation_data(
                    model_type='first_order',
                    noise_level=0.0,
                    n_points=12,
                    CAf=CAf_fit,
                    k0_true=1e10,
                    Ea_true=80.0,
                    include_dH=False,
                    Tf=Tf_fit
                )
                st.session_state.fit_data = pd.DataFrame({
                    'T': data['T'],
                    'tau': data['tau'],
                    'CA': data['CA'],
                    'CB': data['CB']
                })
                st.success("✅ 已加载理想一级反应（等温无噪声）数据集")
                st.info(f"真实参数: k0=1e10 s⁻¹, Ea=80 kJ/mol")

        with col_ds2:
            if st.button("📦 含噪声二级反应", use_container_width=True):
                data = generate_validation_data(
                    model_type='second_order',
                    noise_level=0.05,
                    n_points=12,
                    CAf=CAf_fit,
                    k0_true=5e8,
                    Ea_true=65.0,
                    include_dH=False,
                    Tf=Tf_fit
                )
                st.session_state.fit_data = pd.DataFrame({
                    'T': data['T'],
                    'tau': data['tau'],
                    'CA': data['CA'],
                    'CB': data['CB']
                })
                st.success("✅ 已加载含噪声二级反应验证数据集")
                st.info(f"真实参数: k0=5e8 m³/(mol·s), Ea=65 kJ/mol")

        with col_ds3:
            if st.button("🔥 绝热一级(含ΔH)", use_container_width=True):
                data = generate_validation_data(
                    model_type='first_order',
                    noise_level=0.0,
                    n_points=12,
                    CAf=CAf_fit,
                    k0_true=1e10,
                    Ea_true=80.0,
                    deltaH_true=-100.0,
                    include_dH=True,
                    Tf=Tf_fit
                )
                st.session_state.fit_data = pd.DataFrame({
                    'T': data['T'],
                    'tau': data['tau'],
                    'CA': data['CA'],
                    'CB': data['CB']
                })
                st.success("✅ 已加载绝热一级反应（含反应热）数据集")
                st.info(f"真实参数: k0=1e10 s⁻¹, Ea=80 kJ/mol, ΔH=-100 kJ/mol")

        st.markdown("---")

        st.markdown("**📋 实验数据表格（直接点击单元格编辑数值）**")

        T_vals = st.session_state.fit_data['T'].values.astype(float)
        tau_vals = st.session_state.fit_data['tau'].values.astype(float)
        CA_vals = st.session_state.fit_data['CA'].values.astype(float)
        CB_vals = st.session_state.fit_data['CB'].values.astype(float)

        errors = validate_experimental_data(T_vals, tau_vals, CA_vals, CB_vals)
        error_mask = get_cell_error_mask(T_vals, tau_vals, CA_vals, CB_vals)

        def highlight_errors(row):
            styles = [''] * len(row)
            idx = row.name
            if idx < len(error_mask['T']):
                if error_mask['T'][idx]:
                    styles[0] = 'background-color: #ffcccc'
                if error_mask['tau'][idx]:
                    styles[1] = 'background-color: #ffcccc'
                if error_mask['CA'][idx]:
                    styles[2] = 'background-color: #ffcccc'
                if error_mask['CB'][idx]:
                    styles[3] = 'background-color: #ffcccc'
            return styles

        column_config = {
            'T': st.column_config.NumberColumn(
                "温度 T (K)",
                min_value=0.0,
                max_value=2000.0,
                step=1.0,
                format="%.2f",
                required=True
            ),
            'tau': st.column_config.NumberColumn(
                "停留时间 τ (s)",
                min_value=0.0,
                max_value=1e6,
                step=0.1,
                format="%.4f",
                required=True
            ),
            'CA': st.column_config.NumberColumn(
                "出口A浓度 (mol/m³)",
                min_value=0.0,
                max_value=1e6,
                step=0.1,
                format="%.4f",
                required=True
            ),
            'CB': st.column_config.NumberColumn(
                "出口B浓度 (mol/m³)",
                min_value=0.0,
                max_value=1e6,
                step=0.1,
                format="%.4f",
                required=True
            )
        }

        edited_df = st.data_editor(
            st.session_state.fit_data.astype(float),
            column_config=column_config,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key="fit_data_editor"
        )
        st.session_state.fit_data = edited_df

        if errors:
            st.error("❌ 数据校验错误：")
            for err in errors:
                st.error(f"  • {err}")
        else:
            st.success("✅ 数据校验通过")

        col_add, col_del, col_clear = st.columns(3)
        with col_add:
            if st.button("➕ 添加行", use_container_width=True):
                new_row = pd.DataFrame({'T': [400.0], 'tau': [20.0], 'CA': [50.0], 'CB': [50.0]})
                st.session_state.fit_data = pd.concat(
                    [st.session_state.fit_data, new_row], ignore_index=True
                )
                st.rerun()

        with col_del:
            if st.button("➖ 删除最后一行", use_container_width=True):
                if len(st.session_state.fit_data) > 1:
                    st.session_state.fit_data = st.session_state.fit_data.iloc[:-1].reset_index(drop=True)
                    st.rerun()
                else:
                    st.warning("至少保留一行数据")

        with col_clear:
            if st.button("🗑️ 清空数据", use_container_width=True):
                st.session_state.fit_data = pd.DataFrame({
                    'T': [350.0], 'tau': [10.0], 'CA': [70.0], 'CB': [30.0]
                })
                st.rerun()

        uploaded_file = st.file_uploader("📥 导入CSV文件 (列名: T, tau, CA, CB)", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['T', 'tau', 'CA', 'CB']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols]
                    st.session_state.fit_data = df
                    st.success(f"✅ 成功导入 {len(df)} 条数据")
                    st.rerun()
                else:
                    missing = [col for col in required_cols if col not in df.columns]
                    st.error(f"CSV文件缺少必要列: {missing}")
            except Exception as e:
                st.error(f"导入失败: {str(e)}")

        n_points = len(st.session_state.fit_data)
        if n_points < 6:
            st.warning(f"⚠️ 当前有 {n_points} 个数据点，至少需要 6 个才能启动拟合")

    with col_config:
        st.subheader("⚙️ 拟合配置")

        model_type = st.selectbox(
            "反应模型",
            ["一级不可逆 A→B", "二级不可逆 A→B"],
            index=0
        )
        model_type_key = 'first_order' if '一级' in model_type else 'second_order'

        include_dH = st.checkbox(
            "同时拟合反应热 ΔH（绝热CSTR）",
            value=False,
            help="勾选后采用绝热CSTR能量平衡模型，通过温度-浓度耦合关系估计ΔH；不勾选则为等温模型，温度直接取自实验数据"
        )
        if include_dH:
            st.info("💡 拟合反应热需要实验数据中温度有显著的温升或温降（温度-转化率存在相关性）。"
                    "如果温度不随转化率变化，则ΔH参数不可辨识。")

        st.markdown("**参数搜索边界**")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            k0_min = st.number_input("k0 最小值", 1e3, 1e18, 1e5, format="%.2e")
            Ea_min = st.number_input("Ea 最小值 (kJ/mol)", 10.0, 300.0, 30.0, 1.0)
            if include_dH:
                dH_min = st.number_input("ΔH 最小值 (kJ/mol)", -500.0, 500.0, -300.0, 10.0)
        with col_b2:
            k0_max = st.number_input("k0 最大值", 1e3, 1e18, 1e15, format="%.2e")
            Ea_max = st.number_input("Ea 最大值 (kJ/mol)", 10.0, 300.0, 200.0, 1.0)
            if include_dH:
                dH_max = st.number_input("ΔH 最大值 (kJ/mol)", -500.0, 500.0, 100.0, 10.0)

        if include_dH:
            bounds = ([k0_min, Ea_min, dH_min], [k0_max, Ea_max, dH_max])
        else:
            bounds = ([k0_min, Ea_min], [k0_max, Ea_max])

        st.markdown("**初始猜测**")
        auto_init = st.checkbox("自动初始化", value=True)

        if not auto_init:
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                k0_guess = st.number_input("k0 初始值", k0_min, k0_max, 1e10, format="%.2e")
            with col_g2:
                Ea_guess = st.number_input("Ea 初始值", Ea_min, Ea_max, 80.0, 1.0)
            with col_g3:
                if include_dH:
                    dH_guess = st.number_input("ΔH 初始值", dH_min, dH_max, -50.0, 1.0)

            if include_dH:
                x0 = [k0_guess, Ea_guess, dH_guess]
            else:
                x0 = [k0_guess, Ea_guess]
        else:
            x0 = None

        can_fit = (n_points >= 6) and (not errors)

        if st.button("🚀 开始拟合", type="primary", use_container_width=True, disabled=not can_fit):
            if not can_fit:
                if n_points < 6:
                    st.error("需要至少6个数据点")
                if errors:
                    st.error("请先修正数据校验错误")
            else:
                with st.spinner("正在进行参数拟合..."):
                    T_data = st.session_state.fit_data['T'].values.astype(float)
                    tau_data = st.session_state.fit_data['tau'].values.astype(float)
                    CA_data = st.session_state.fit_data['CA'].values.astype(float)
                    CB_data = st.session_state.fit_data['CB'].values.astype(float)

                    result = fit_parameters(
                        T_data, tau_data, CA_data, CB_data, CAf_fit,
                        model_type=model_type_key,
                        include_dH=include_dH,
                        Tf=Tf_fit,
                        bounds=bounds,
                        x0=x0,
                        auto_init=auto_init
                    )
                    st.session_state.fit_result = result

                    if result['success']:
                        st.success("✅ 拟合成功！")
                    else:
                        st.error(f"❌ {result['message']}")
    
    if st.session_state.fit_result is not None:
        result = st.session_state.fit_result
        
        st.markdown("---")
        col_results, col_stats = st.columns([1, 1])
        
        with col_results:
            st.subheader("📈 拟合结果")
            
            if result['success']:
                param_names = result['param_names']
                param_units = result['param_units']
                params_opt = result['params']
                ci = result['confidence_intervals']
                
                param_display_names = {
                    'k0': '频率因子 k0',
                    'Ea': '活化能 Ea',
                    'deltaH': '反应热 ΔH'
                }
                
                for i, name in enumerate(param_names):
                    val = params_opt[i]
                    ci_val = ci[i]
                    unit = param_units[i]
                    display_name = param_display_names.get(name, name)
                    
                    if name == 'k0':
                        st.write(f"**{display_name}**: {val:.4e} ± {ci_val:.4e} {unit}")
                    else:
                        st.write(f"**{display_name}**: {val:.4f} ± {ci_val:.4f} {unit}")
                
                st.markdown("---")
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("残差平方和 RSS", f"{result['RSS']:.4e}")
                with metric_cols[1]:
                    st.metric("决定系数 R²", f"{result['R2']:.6f}")
                with metric_cols[2]:
                    st.metric("均方根误差 RMSE", f"{result['RMSE']:.4f}")
                
                st.info(f"自由度: {result['dof']}")
            else:
                st.error(result['message'])
        
        with col_stats:
            st.subheader("📊 统计分析")

            if result['success']:
                param_names = result['param_names']
                n_params = len(param_names)
                residuals_meaningful = result.get('residuals_meaningful', True)
                residual_scale = result.get('residual_scale', 0)

                st.markdown("**参数相关系数矩阵**")
                corr_matrix = result['corr_matrix']
                corr_df = pd.DataFrame(corr_matrix, columns=param_names, index=param_names)

                def highlight_high_corr(val):
                    if abs(val) > 0.95 and abs(val - 1.0) > 1e-10:
                        return 'background-color: #ffcccc; color: red; font-weight: bold'
                    return ''

                corr_styled = corr_df.style.applymap(highlight_high_corr).format("{:.4f}")
                st.dataframe(corr_styled, use_container_width=True)

                high_corr = []
                for i in range(n_params):
                    for j in range(i + 1, n_params):
                        if abs(corr_matrix[i, j]) > 0.95 and abs(corr_matrix[i, j] - 1.0) > 1e-10:
                            high_corr.append(f"{param_names[i]} - {param_names[j]}")

                if high_corr:
                    st.error("⚠️ 警告: 检测到高度相关的参数对")
                    for pair in high_corr:
                        st.error(f"  • {pair} - 相关系数 > 0.95，参数可能不可辨识")
                else:
                    st.success("✅ 参数无高度相关，可辨识性良好")

                st.markdown("---")

                stat_cols = st.columns(2)
                with stat_cols[0]:
                    st.markdown("**残差正态性检验**")
                    shapiro_p = result['shapiro_p']

                    if not residuals_meaningful:
                        st.info(f"ℹ️ 残差尺度极小 (相对尺度 = {residual_scale:.1e})，"
                                "接近数值噪声量级，正态性检验不适用，"
                                "表示拟合精度已达到数值极限")
                    elif np.isnan(shapiro_p):
                        st.write("样本量不足或残差方差为零")
                    else:
                        st.write(f"Shapiro-Wilk p值: {shapiro_p:.4f}")
                        if shapiro_p < 0.05:
                            st.error("⚠️ 残差非正态 (p < 0.05)，模型假设可能不成立")
                        else:
                            st.success("✅ 残差正态性检验通过")

                with stat_cols[1]:
                    st.markdown("**残差自相关检验**")
                    dw_stat = result['dw_stat']

                    if not residuals_meaningful:
                        st.info(f"ℹ️ 残差尺度极小 (相对尺度 = {residual_scale:.1e})，"
                                "接近数值噪声量级，自相关检验不适用，"
                                "表示模型几乎完美解释了数据")
                    elif np.isnan(dw_stat):
                        st.write("样本量不足或残差方差为零")
                    else:
                        st.write(f"Durbin-Watson 统计量: {dw_stat:.4f}")
                        if dw_stat < 1.5:
                            st.error("⚠️ DW < 1.5，残差存在正自相关，可能遗漏系统性效应")
                        elif dw_stat > 2.5:
                            st.error("⚠️ DW > 2.5，残差存在负自相关，可能遗漏系统性效应")
                        else:
                            st.success("✅ 残差无显著自相关")
        
        st.markdown("---")
        st.subheader("📉 拟合可视化")
        
        if result['success']:
            tab1, tab2, tab3, tab4 = st.tabs(["拟合对比图", "残差分布图", "参数置信椭圆", "灵敏度曲线"])
            
            with tab1:
                n_points = len(result['T_data'])
                x_indices = np.arange(1, n_points + 1)
                
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.scatter(x_indices, result['CA_data'], c='blue', s=80, zorder=5, label='实测 CA')
                ax1.plot(x_indices, result['CA_pred'], 'b-', linewidth=2, label='预测 CA')
                ax1.scatter(x_indices, result['CB_data'], c='orange', s=80, zorder=5, label='实测 CB')
                ax1.plot(x_indices, result['CB_pred'], 'orange', linestyle='--', linewidth=2, label='预测 CB')
                
                ax1.set_xlabel('数据点编号', fontsize=12)
                ax1.set_ylabel('出口浓度 (mol/m³)', fontsize=12)
                ax1.set_title(f'拟合对比图 (R² = {result["R2"]:.4f})', fontsize=14)
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_xticks(x_indices)
                st.pyplot(fig1)
            
            with tab2:
                CA_pred = result['CA_pred']
                CB_pred = result['CB_pred']
                res_CA = result['residuals_CA']
                res_CB = result['residuals_CB']
                
                fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax2a.scatter(CA_pred, res_CA, c='blue', s=80, alpha=0.7)
                ax2a.axhline(0, color='red', linestyle='--', linewidth=2)
                ax2a.set_xlabel('预测 CA (mol/m³)', fontsize=12)
                ax2a.set_ylabel('残差 (mol/m³)', fontsize=12)
                ax2a.set_title('CA 残差分布', fontsize=13)
                ax2a.grid(True, alpha=0.3)
                
                ax2b.scatter(CB_pred, res_CB, c='orange', s=80, alpha=0.7)
                ax2b.axhline(0, color='red', linestyle='--', linewidth=2)
                ax2b.set_xlabel('预测 CB (mol/m³)', fontsize=12)
                ax2b.set_ylabel('残差 (mol/m³)', fontsize=12)
                ax2b.set_title('CB 残差分布', fontsize=13)
                ax2b.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
            
            with tab3:
                params = result['params']
                cov_matrix = result['cov_matrix']
                param_names = result['param_names']
                n_params = len(params)
                
                display_labels = {
                    'k0': 'k0 (频率因子)',
                    'Ea': 'Ea (活化能, kJ/mol)',
                    'deltaH': 'ΔH (反应热, kJ/mol)'
                }
                
                param_pairs = []
                for i in range(n_params):
                    for j in range(i + 1, n_params):
                        param_pairs.append((i, j))
                
                if len(param_pairs) == 1:
                    fig3 = plt.figure(figsize=(8, 6))
                    ax = fig3.add_subplot(111)
                    
                    idx1, idx2 = param_pairs[0]
                    ellipse = compute_confidence_ellipse(cov_matrix, params, idx1, idx2)
                    
                    if param_names[idx1] == 'k0':
                        ax.semilogx(ellipse[:, 0], ellipse[:, 1], 'b-', linewidth=2, label='95% 置信椭圆')
                        ax.semilogx(params[idx1], params[idx2], 'ro', markersize=10, label='最优值')
                    else:
                        ax.plot(ellipse[:, 0], ellipse[:, 1], 'b-', linewidth=2, label='95% 置信椭圆')
                        ax.plot(params[idx1], params[idx2], 'ro', markersize=10, label='最优值')
                    
                    ax.set_xlabel(display_labels[param_names[idx1]], fontsize=12)
                    ax.set_ylabel(display_labels[param_names[idx2]], fontsize=12)
                    ax.set_title(f'{display_labels[param_names[idx1]]} vs {display_labels[param_names[idx2]]}', fontsize=13)
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                else:
                    fig3, axes = plt.subplots(1, len(param_pairs), figsize=(6 * len(param_pairs), 5))
                    
                    for idx, (i, j) in enumerate(param_pairs):
                        ax = axes[idx]
                        ellipse = compute_confidence_ellipse(cov_matrix, params, i, j)
                        
                        if param_names[i] == 'k0':
                            ax.semilogx(ellipse[:, 0], ellipse[:, 1], 'b-', linewidth=2)
                            ax.semilogx(params[i], params[j], 'ro', markersize=10)
                        else:
                            ax.plot(ellipse[:, 0], ellipse[:, 1], 'b-', linewidth=2)
                            ax.plot(params[i], params[j], 'ro', markersize=10)
                        
                        ax.set_xlabel(display_labels[param_names[i]], fontsize=11)
                        ax.set_ylabel(display_labels[param_names[j]], fontsize=11)
                        ax.set_title(f'{param_names[i]} vs {param_names[j]}', fontsize=12)
                        ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig3)
            
            with tab4:
                sensitivity_data, ref_idx = compute_sensitivity_curves(result)
                
                n_params = len(sensitivity_data)
                fig4, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
                
                if n_params == 1:
                    axes = [axes]
                
                for idx, (param_name, data) in enumerate(sensitivity_data.items()):
                    ax = axes[idx]
                    param_vals = data['param_values']
                    outputs = data['model_outputs']
                    
                    color = ['#1f77b4', '#ff7f0e', '#2ca02c'][idx]
                    
                    if param_name == 'k0':
                        ax.semilogx(param_vals, outputs, '-', color=color, linewidth=2)
                        ax.set_xscale('log')
                    else:
                        ax.plot(param_vals, outputs, '-', color=color, linewidth=2)
                    
                    opt_val = result['params'][result['param_names'].index(param_name)]
                    ax.axvline(opt_val, color='red', linestyle='--', linewidth=2, label='最优值')
                    
                    display_name = display_labels.get(param_name, param_name)
                    ax.set_xlabel(display_name, fontsize=11)
                    ax.set_ylabel('预测 CB (mol/m³)', fontsize=11)
                    ax.set_title(f'{param_name} 灵敏度曲线', fontsize=12)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig4)


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
         "⚠️ 安全分析", "📈 优化求解", "📊 参数灵敏度",
         "🔗 反应器网络设计", "🔬 数据拟合与参数估计"])
    
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
    elif page == "🔗 反应器网络设计":
        page_reactor_network(reactions)
    elif page == "🔬 数据拟合与参数估计":
        page_parameter_estimation()


EXAMPLE_NETWORKS = {
    "方案一：两级串联": {
        "description": "两个CSTR串联，第一级高温快速反应，第二级低温精细控制提高转化率（参数已优化至工业合理范围）",
        "nodes": [
            {"id": 0, "name": "R1", "reactor_type": "CSTR", "V": 1.5, "T_c": 340.0, "UA": 8000.0, "x": 150, "y": 200},
            {"id": 1, "name": "R2", "reactor_type": "CSTR", "V": 3.0, "T_c": 330.0, "UA": 8000.0, "x": 500, "y": 200},
        ],
        "connections": [
            {"source": 0, "target": 1, "split_ratio": 1.0},
        ],
        "feed_targets": {0: 1.0},
    },
    "方案二：并联分流": {
        "description": "一个进料分成两路，分别进入不同体积的CSTR，出料汇合作为产品（分流比40%/60%）",
        "nodes": [
            {"id": 0, "name": "R1-小径流", "reactor_type": "CSTR", "V": 2.0, "T_c": 335.0, "UA": 6000.0, "x": 150, "y": 100},
            {"id": 1, "name": "R2-大径流", "reactor_type": "CSTR", "V": 3.0, "T_c": 335.0, "UA": 7000.0, "x": 150, "y": 320},
            {"id": 2, "name": "混合器", "reactor_type": "CSTR", "V": 0.3, "T_c": 320.0, "UA": 3000.0, "x": 500, "y": 210},
        ],
        "connections": [
            {"source": 0, "target": 2, "split_ratio": 1.0},
            {"source": 1, "target": 2, "split_ratio": 1.0},
        ],
        "feed_targets": {0: 0.4, 1: 0.6},
    },
    "方案三：带回流循环": {
        "description": "主反应器出料的30%回流到入口，与新鲜进料混合，模拟工业循环反应器（提高单程转化率）",
        "nodes": [
            {"id": 0, "name": "混合器", "reactor_type": "CSTR", "V": 0.3, "T_c": 330.0, "UA": 2000.0, "x": 100, "y": 200},
            {"id": 1, "name": "R1主反应器", "reactor_type": "CSTR", "V": 3.0, "T_c": 340.0, "UA": 9000.0, "x": 420, "y": 200},
        ],
        "connections": [
            {"source": 0, "target": 1, "split_ratio": 1.0},
            {"source": 1, "target": 0, "split_ratio": 0.3},
        ],
        "feed_targets": {0: 1.0},
    },
}


def init_network_state():
    if 'network' not in st.session_state:
        st.session_state.network = ReactorNetwork()
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None
    if 'selected_conn' not in st.session_state:
        st.session_state.selected_conn = None
    if 'network_metrics' not in st.session_state:
        st.session_state.network_metrics = None
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'ui_salt' not in st.session_state:
        st.session_state.ui_salt = 0
    if 'network_snapshots' not in st.session_state:
        st.session_state.network_snapshots = []
    if 'param_scan_result' not in st.session_state:
        st.session_state.param_scan_result = None
    if 'heatmap_result' not in st.session_state:
        st.session_state.heatmap_result = None


def save_network_snapshot(name):
    init_network_state()
    if len(st.session_state.network_snapshots) >= 5:
        return False, "最多保存5个快照"
    if st.session_state.network_metrics is None:
        return False, "请先运行仿真"
    
    metrics = st.session_state.network_metrics
    snapshot = {
        'id': datetime.now().timestamp(),
        'name': name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_conversion': metrics['total_conversion'],
        'total_selectivity': metrics['total_selectivity'],
        'total_yield': metrics['total_yield'],
        'total_heat_load': metrics['total_heat_load'],
        'max_temperature': metrics['max_temperature'],
    }
    st.session_state.network_snapshots.append(snapshot)
    return True, f"已保存快照 \"{name}\""


def delete_network_snapshot(snapshot_id):
    init_network_state()
    st.session_state.network_snapshots = [
        s for s in st.session_state.network_snapshots if s['id'] != snapshot_id
    ]


def clear_all_snapshots():
    init_network_state()
    st.session_state.network_snapshots = []


SNAPSHOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def render_snapshot_comparison():
    init_network_state()
    snapshots = st.session_state.network_snapshots
    
    st.subheader("📸 方案对比")
    
    col_save, col_count = st.columns([2, 1])
    with col_save:
        snap_name = st.text_input(
            "快照名称",
            value=f"方案{len(snapshots) + 1}",
            max_chars=20,
            key="snap_name_input"
        )
        if st.button("💾 保存当前方案为快照", disabled=st.session_state.network_metrics is None):
            success, msg = save_network_snapshot(snap_name.strip())
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    with col_count:
        st.info(f"已保存: {len(snapshots)}/5")
        if st.button("🗑️ 清除全部快照", disabled=len(snapshots) == 0):
            clear_all_snapshots()
            st.rerun()
    
    if len(snapshots) == 0:
        st.info("暂无快照。运行仿真后，可保存当前方案为快照进行对比。")
        return
    
    st.markdown("**📊 核心指标对比表**")
    
    metric_keys = [
        ('total_conversion', '转化率 (%)', 100, '%'),
        ('total_selectivity', '选择性 (%)', 100, '%'),
        ('total_yield', '收率 (%)', 100, '%'),
        ('total_heat_load', '热负荷 (kW)', 1, 'kW'),
        ('max_temperature', '最高温度 (K)', 1, 'K'),
    ]
    
    max_diffs = {}
    for key, _, _, _ in metric_keys:
        values = [s[key] for s in snapshots]
        if len(values) >= 2:
            max_diffs[key] = max(values) - min(values)
        else:
            max_diffs[key] = 0
    
    table_html = '<table style="width:100%; border-collapse: collapse;">'
    table_html += '<tr><th style="text-align:left; padding:8px; border-bottom:2px solid #ddd;">指标</th>'
    for i, snap in enumerate(snapshots):
        color = SNAPSHOT_COLORS[i % len(SNAPSHOT_COLORS)]
        table_html += f'<th style="text-align:center; padding:8px; border-bottom:2px solid #ddd; color:{color};">{snap["name"]}</th>'
    table_html += '</tr>'
    
    for key, label, scale, unit in metric_keys:
        table_html += f'<tr><td style="padding:8px; border-bottom:1px solid #eee;">{label}</td>'
        max_diff_val = max_diffs[key]
        values = [s[key] for s in snapshots]
        max_val = max(values) if values else 0
        min_val = min(values) if values else 0
        
        for i, snap in enumerate(snapshots):
            val = snap[key] * scale
            is_max = (snap[key] == max_val) and (max_diff_val > 0)
            is_min = (snap[key] == min_val) and (max_diff_val > 0)
            bg_color = ''
            text_color = ''
            
            if key in ['total_conversion', 'total_selectivity', 'total_yield']:
                if is_max:
                    bg_color = 'background-color: #c8e6c9; color: #1b5e20;'
            elif key == 'total_heat_load':
                if is_min:
                    bg_color = 'background-color: #c8e6c9; color: #1b5e20;'
            elif key == 'max_temperature':
                if is_max:
                    bg_color = 'background-color: #ffcdd2; color: #b71c1c;'
            
            table_html += f'<td style="text-align:center; padding:8px; border-bottom:1px solid #eee; {bg_color}">{val:.2f} {unit}</td>'
        table_html += '</tr>'
    
    table_html += '</table>'
    
    col_table, col_actions = st.columns([4, 1])
    with col_table:
        st.markdown(table_html, unsafe_allow_html=True)
    
    with col_actions:
        st.markdown("**操作**")
        for i, snap in enumerate(snapshots):
            if st.button(f"删除 #{i+1}", key=f"del_snap_{snap['id']}"):
                delete_network_snapshot(snap['id'])
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("🕸️ **雷达图对比（归一化）**")
    
    categories = ['转化率', '选择性', '收率']
    radar_categories = categories.copy()
    for i in range(len(snapshots)):
        pass
    
    norm_snapshots = []
    for key, label, scale, _ in metric_keys:
        values = [s[key] for s in snapshots]
        vmin = min(values)
        vmax = max(values)
        vrange = vmax - vmin if vmax > vmin else 1.0
        
        norm_vals = []
        for s in snapshots:
            if key == 'total_heat_load':
                nv = 1.0 - (s[key] - vmin) / vrange
            else:
                nv = (s[key] - vmin) / vrange
            norm_vals.append(nv)
        
        for i, s in enumerate(snapshots):
            if i >= len(norm_snapshots):
                norm_snapshots.append({})
            norm_snapshots[i][key] = norm_vals[i]
    
    radar_categories_full = ['转化率', '选择性', '收率', '热负荷(逆向)', '最高温度(逆向)']
    
    fig_radar = go.Figure()
    for i, snap in enumerate(snapshots):
        r_vals = [
            norm_snapshots[i]['total_conversion'] * 100,
            norm_snapshots[i]['total_selectivity'] * 100,
            norm_snapshots[i]['total_yield'] * 100,
            norm_snapshots[i]['total_heat_load'] * 100,
            norm_snapshots[i]['max_temperature'] * 100,
        ]
        color = SNAPSHOT_COLORS[i % len(SNAPSHOT_COLORS)]
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=radar_categories_full,
            fill='toself',
            name=snap['name'],
            line=dict(color=color, width=2),
            opacity=0.6
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                title='归一化得分'
            )
        ),
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)


def get_scan_variables(network):
    variables = []
    
    for nid, node in network.nodes.items():
        variables.append({
            'type': 'node_Tc',
            'node_id': nid,
            'label': f"{node.name} - 冷却温度 T_c",
            'unit': 'K',
            'default_min': 260.0,
            'default_max': 350.0,
            'default_value': node.T_c,
        })
    
    for nid, node in network.nodes.items():
        out_conns = network.get_outgoing_connections(nid)
        if len(out_conns) >= 2:
            for conn in out_conns:
                src_name = network.nodes[conn.source_id].name
                tgt_name = network.nodes[conn.target_id].name
                variables.append({
                    'type': 'split_ratio',
                    'conn_id': conn.id,
                    'label': f"{src_name}→{tgt_name} - 分流比",
                    'unit': '',
                    'default_min': 0.05,
                    'default_max': 0.95,
                    'default_value': conn.split_ratio,
                })
    
    return variables


def run_parameter_scan(network, reactions, var_info, val_min, val_max, n_steps):
    results = []
    original_state = _save_network_state(network)
    
    values = np.linspace(val_min, val_max, n_steps)
    
    for val in values:
        _apply_variable(network, var_info, val)
        
        for nid in network.nodes:
            network.nodes[nid].C_out = None
            network.nodes[nid].T_out = None
        
        success = network.solve(reactions)
        
        if success:
            metrics = network.get_metrics()
            if metrics:
                results.append({
                    'value': float(val),
                    'success': True,
                    'conversion': metrics['total_conversion'],
                    'yield': metrics['total_yield'],
                    'heat_load': metrics['total_heat_load'],
                    'max_temperature': metrics['max_temperature'],
                })
            else:
                results.append({
                    'value': float(val),
                    'success': False,
                })
        else:
            results.append({
                'value': float(val),
                'success': False,
            })
    
    _restore_network_state(network, original_state)
    network.solve(reactions)
    
    return results


def _save_network_state(network):
    state = {
        'nodes': {},
        'connections': [],
    }
    for nid, node in network.nodes.items():
        state['nodes'][nid] = {
            'T_c': node.T_c,
            'C_out': node.C_out.copy() if node.C_out is not None else None,
            'T_out': node.T_out,
        }
    for conn in network.connections:
        state['connections'].append({
            'id': conn.id,
            'split_ratio': conn.split_ratio,
        })
    return state


def _restore_network_state(network, state):
    for nid, node_state in state['nodes'].items():
        if nid in network.nodes:
            network.nodes[nid].T_c = node_state['T_c']
            network.nodes[nid].C_out = node_state['C_out']
            network.nodes[nid].T_out = node_state['T_out']
    for conn_state in state['connections']:
        for conn in network.connections:
            if conn.id == conn_state['id']:
                conn.split_ratio = conn_state['split_ratio']
                break


def _apply_variable(network, var_info, value):
    if var_info['type'] == 'node_Tc':
        nid = var_info['node_id']
        if nid in network.nodes:
            network.nodes[nid].T_c = float(value)
    elif var_info['type'] == 'split_ratio':
        conn_id = var_info['conn_id']
        for conn in network.connections:
            if conn.id == conn_id:
                conn.split_ratio = float(value)
                break
        src_id = None
        for conn in network.connections:
            if conn.id == conn_id:
                src_id = conn.source_id
                break
        if src_id is not None:
            network._normalize_split_ratios()


def render_parameter_sweep(network, reactions):
    st.subheader("🔬 参数扫描")
    
    variables = get_scan_variables(network)
    
    if not variables:
        st.info("当前网络没有可扫描的参数。请添加反应器节点或并联连接。")
        return
    
    col_var, col_range = st.columns([2, 1])
    
    with col_var:
        var_labels = [v['label'] for v in variables]
        var_idx = st.selectbox("选择扫描变量", range(len(var_labels)),
                               format_func=lambda i: var_labels[i],
                               key="scan_var_select")
        selected_var = variables[var_idx]
    
    with col_range:
        n_steps = st.slider("扫描步数", 5, 20, 10, 1, key="scan_steps")
    
    col_min, col_max = st.columns(2)
    with col_min:
        val_min = st.number_input("起始值", value=float(selected_var['default_min']),
                                  format="%.2f", key="scan_min")
    with col_max:
        val_max = st.number_input("终止值", value=float(selected_var['default_max']),
                                  format="%.2f", key="scan_max")
    
    if st.button("▶️ 开始扫描", type="primary", use_container_width=True,
                 disabled=not network.solved):
        if val_min >= val_max:
            st.error("起始值必须小于终止值")
        else:
            with st.spinner(f"正在进行参数扫描 ({n_steps} 步)..."):
                results = run_parameter_scan(network, reactions, selected_var, val_min, val_max, n_steps)
                st.session_state.param_scan_result = {
                    'variable': selected_var,
                    'results': results,
                    'val_min': val_min,
                    'val_max': val_max,
                    'n_steps': n_steps,
                }
                st.success("✅ 参数扫描完成！")
    
    if not network.solved:
        st.warning("请先完成网络仿真，再进行参数扫描")
    
    if st.session_state.param_scan_result is not None:
        scan_data = st.session_state.param_scan_result
        results = scan_data['results']
        var_info = scan_data['variable']
        
        st.markdown("---")
        st.markdown("📈 **扫描结果**")
        
        fig = go.Figure()
        
        all_x = [r['value'] for r in results]
        all_conv = [r['conversion'] * 100 if r['success'] else np.nan for r in results]
        all_yield = [r['yield'] * 100 if r['success'] else np.nan for r in results]
        all_heat = [r['heat_load'] if r['success'] else np.nan for r in results]
        success_flags = [r['success'] for r in results]
        
        failed_indices = [i for i, ok in enumerate(success_flags) if not ok]
        
        def _build_segments(values):
            segments = []
            current_seg = []
            for i, v in enumerate(values):
                if success_flags[i]:
                    current_seg.append((all_x[i], v))
                else:
                    if len(current_seg) > 0:
                        segments.append(('solid', current_seg))
                        current_seg = []
                    prev_ok = i - 1 >= 0 and success_flags[i - 1]
                    next_ok = i + 1 < len(values) and success_flags[i + 1]
                    if prev_ok and next_ok:
                        dash_seg = [(all_x[i - 1], values[i - 1]), (all_x[i + 1], values[i + 1])]
                        segments.append(('dash', dash_seg))
            if len(current_seg) > 0:
                segments.append(('solid', current_seg))
            return segments
        
        for style, seg in _build_segments(all_conv):
            sx = [p[0] for p in seg]
            sy = [p[1] for p in seg]
            dash_style = None if style == 'solid' else 'dash'
            fig.add_trace(go.Scatter(
                x=sx, y=sy,
                name='转化率 (%)' if style == 'solid' else None,
                line=dict(color='#1f77b4', width=2, dash=dash_style),
                yaxis='y1',
                mode='lines+markers' if style == 'solid' else 'lines',
                showlegend=(style == 'solid'),
                legendgroup='conv',
                hovertemplate=f'转化率: %{{y:.2f}}%<br>{var_info["label"]}: %{{x:.2f}}<extra></extra>'
            ))
        
        for style, seg in _build_segments(all_yield):
            sx = [p[0] for p in seg]
            sy = [p[1] for p in seg]
            dash_style = None if style == 'solid' else 'dash'
            fig.add_trace(go.Scatter(
                x=sx, y=sy,
                name='收率 (%)' if style == 'solid' else None,
                line=dict(color='#2ca02c', width=2, dash=dash_style),
                yaxis='y1',
                mode='lines+markers' if style == 'solid' else 'lines',
                showlegend=(style == 'solid'),
                legendgroup='yield',
                hovertemplate=f'收率: %{{y:.2f}}%<br>{var_info["label"]}: %{{x:.2f}}<extra></extra>'
            ))
        
        for style, seg in _build_segments(all_heat):
            sx = [p[0] for p in seg]
            sy = [p[1] for p in seg]
            dash_style = None if style == 'solid' else 'dash'
            fig.add_trace(go.Scatter(
                x=sx, y=sy,
                name='热负荷 (kW)' if style == 'solid' else None,
                line=dict(color='#ff7f0e', width=2, dash=dash_style),
                yaxis='y2',
                mode='lines+markers' if style == 'solid' else 'lines',
                showlegend=(style == 'solid'),
                legendgroup='heat',
                hovertemplate=f'热负荷: %{{y:.2f}} kW<br>{var_info["label"]}: %{{x:.2f}}<extra></extra>'
            ))
        
        if failed_indices:
            failed_x = [results[i]['value'] for i in failed_indices]
            valid_conv = [v for v in all_conv if not np.isnan(v)]
            valid_yield = [v for v in all_yield if not np.isnan(v)]
            valid_heat = [v for v in all_heat if not np.isnan(v)]
            y_min_ref = min(min(valid_conv) if valid_conv else 0, min(valid_yield) if valid_yield else 0)
            y_max_ref = max(max(valid_conv) if valid_conv else 100, max(valid_yield) if valid_yield else 100)
            y_fail_marker = y_min_ref - (y_max_ref - y_min_ref) * 0.05 if y_max_ref > y_min_ref else -5
            
            for i, fx in enumerate(failed_x):
                fig.add_annotation(
                    x=fx,
                    y=y_fail_marker,
                    text='⚠️ 求解失败',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30,
                    font=dict(color='red', size=11),
                    bgcolor='rgba(255,230,230,0.8)',
                    bordercolor='red',
                    borderwidth=1,
                    borderpad=2,
                    yref='y1'
                )
            
            fig.add_trace(go.Scatter(
                x=failed_x,
                y=[y_fail_marker] * len(failed_x),
                mode='markers',
                marker=dict(symbol='x', size=12, color='red', line=dict(width=2)),
                name='求解失败',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        x_vals_success = [r['value'] for r in results if r['success']]
        conv_vals_success = [r['conversion'] * 100 for r in results if r['success']]
        yield_vals_success = [r['yield'] * 100 for r in results if r['success']]
        heat_vals_success = [r['heat_load'] for r in results if r['success']]
        
        if conv_vals_success:
            max_conv_idx = np.argmax(conv_vals_success)
            max_conv_x = x_vals_success[max_conv_idx]
            max_conv_y = conv_vals_success[max_conv_idx]
            fig.add_annotation(
                x=max_conv_x, y=max_conv_y,
                text=f"最高转化率<br>{max_conv_y:.1f}%<br>@{max_conv_x:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=0, ay=-40,
                font=dict(color='#1f77b4'),
                yref='y1'
            )
        
        if yield_vals_success:
            max_yield_idx = np.argmax(yield_vals_success)
            max_yield_x = x_vals_success[max_yield_idx]
            max_yield_y = yield_vals_success[max_yield_idx]
            fig.add_annotation(
                x=max_yield_x, y=max_yield_y,
                text=f"最高收率<br>{max_yield_y:.1f}%<br>@{max_yield_x:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=0, ay=60,
                font=dict(color='#2ca02c'),
                yref='y1'
            )
        
        if heat_vals_success:
            min_heat_idx = np.argmin(heat_vals_success)
            min_heat_x = x_vals_success[min_heat_idx]
            min_heat_y = heat_vals_success[min_heat_idx]
            fig.add_annotation(
                x=min_heat_x, y=min_heat_y,
                text=f"最低热负荷<br>{min_heat_y:.1f} kW<br>@{min_heat_x:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=0, ay=-60,
                font=dict(color='#ff7f0e'),
                yref='y2'
            )
        
        fig.update_layout(
            xaxis_title=f"{var_info['label']} ({var_info['unit']})",
            yaxis=dict(
                title='转化率 / 收率 (%)',
                side='left',
            ),
            yaxis2=dict(
                title='热负荷 (kW)',
                side='right',
                overlaying='y',
            ),
            height=520,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=10, r=10, t=40, b=10),
            hovermode='x unified',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("📋 **扫描数据明细**")
        detail_data = []
        for i, r in enumerate(results):
            if r['success']:
                detail_data.append({
                    '序号': i + 1,
                    '变量值': f"{r['value']:.3f}",
                    '转化率 (%)': f"{r['conversion']*100:.2f}",
                    '收率 (%)': f"{r['yield']*100:.2f}",
                    '热负荷 (kW)': f"{r['heat_load']:.2f}",
                    '最高温度 (K)': f"{r['max_temperature']:.1f}",
                    '状态': '✅ 成功',
                })
            else:
                detail_data.append({
                    '序号': i + 1,
                    '变量值': f"{r['value']:.3f}",
                    '转化率 (%)': '-',
                    '收率 (%)': '-',
                    '热负荷 (kW)': '-',
                    '最高温度 (K)': '-',
                    '状态': '❌ 求解失败',
                })
        st.table(pd.DataFrame(detail_data))


def get_heatmap_variables(network):
    variables = []
    
    variables.append({
        'type': 'feed_temp',
        'label': '总进料温度 T_feed',
        'unit': 'K',
        'default_min': 280.0,
        'default_max': 380.0,
        'default_value': network.T_feed,
    })
    
    variables.append({
        'type': 'feed_flow',
        'label': '总进料流量 F_feed',
        'unit': 'm³/s',
        'default_min': 0.005,
        'default_max': 0.05,
        'default_value': network.F_feed,
    })
    
    for nid, node in network.nodes.items():
        variables.append({
            'type': 'node_Tc',
            'node_id': nid,
            'label': f"{node.name} - 冷却温度 T_c",
            'unit': 'K',
            'default_min': 260.0,
            'default_max': 350.0,
            'default_value': node.T_c,
        })
    
    return variables


def run_heatmap_scan(network, reactions, var_x, var_y, x_min, x_max, x_steps, y_min, y_max, y_steps, T_max_constraint=473.15):
    results = {
        'x_vals': np.linspace(x_min, x_max, x_steps),
        'y_vals': np.linspace(y_min, y_max, y_steps),
        'yield_grid': np.full((y_steps, x_steps), np.nan),
        'conversion_grid': np.full((y_steps, x_steps), np.nan),
        'heat_grid': np.full((y_steps, x_steps), np.nan),
        'max_temp_grid': np.full((y_steps, x_steps), np.nan),
        'success_grid': np.full((y_steps, x_steps), False, dtype=bool),
    }
    
    original_state = _save_network_state(network)
    original_T_feed = network.T_feed
    original_F_feed = network.F_feed
    
    x_values = np.linspace(x_min, x_max, x_steps)
    y_values = np.linspace(y_min, y_max, y_steps)
    
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            _apply_heatmap_variable(network, var_x, x_val)
            _apply_heatmap_variable(network, var_y, y_val)
            
            for nid in network.nodes:
                network.nodes[nid].C_out = None
                network.nodes[nid].T_out = None
            
            success = network.solve(reactions)
            
            if success:
                metrics = network.get_metrics()
                if metrics:
                    results['yield_grid'][i, j] = metrics['total_yield']
                    results['conversion_grid'][i, j] = metrics['total_conversion']
                    results['heat_grid'][i, j] = metrics['total_heat_load']
                    results['max_temp_grid'][i, j] = metrics['max_temperature']
                    results['success_grid'][i, j] = True
    
    network.T_feed = original_T_feed
    network.F_feed = original_F_feed
    _restore_network_state(network, original_state)
    network.solve(reactions)
    
    return results


def _apply_heatmap_variable(network, var_info, value):
    if var_info['type'] == 'feed_temp':
        network.T_feed = float(value)
    elif var_info['type'] == 'feed_flow':
        network.F_feed = float(value)
    elif var_info['type'] == 'node_Tc':
        nid = var_info['node_id']
        if nid in network.nodes:
            network.nodes[nid].T_c = float(value)


def render_dual_variable_heatmap(network, reactions):
    st.subheader("🗺️ 双变量热力图")
    
    variables = get_heatmap_variables(network)
    
    if len(variables) < 2:
        st.info("需要至少2个可扫描变量来生成热力图。")
        return
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.markdown("**X轴变量**")
        x_labels = [v['label'] for v in variables]
        x_idx = st.selectbox("选择X轴变量", range(len(x_labels)),
                             format_func=lambda i: x_labels[i],
                             key="heatmap_x_var")
        var_x = variables[x_idx]
        x_steps = st.slider("X轴步数", 5, 10, 6, 1, key="heatmap_x_steps")
        x_min = st.number_input("X轴起始值", value=float(var_x['default_min']),
                               format="%.2f", key="heatmap_x_min")
        x_max = st.number_input("X轴终止值", value=float(var_x['default_max']),
                               format="%.2f", key="heatmap_x_max")
    
    with col_y:
        st.markdown("**Y轴变量**")
        y_labels = [v['label'] for v in variables]
        default_y_idx = min(x_idx + 1, len(variables) - 1)
        y_idx = st.selectbox("选择Y轴变量", range(len(y_labels)),
                             index=default_y_idx,
                             format_func=lambda i: y_labels[i],
                             key="heatmap_y_var")
        var_y = variables[y_idx]
        y_steps = st.slider("Y轴步数", 5, 10, 6, 1, key="heatmap_y_steps")
        y_min = st.number_input("Y轴起始值", value=float(var_y['default_min']),
                               format="%.2f", key="heatmap_y_min")
        y_max = st.number_input("Y轴终止值", value=float(var_y['default_max']),
                               format="%.2f", key="heatmap_y_max")
    
    T_max_constraint = st.number_input("温度上限约束 (K)", 350.0, 600.0, 473.15, 1.0,
                                       key="heatmap_T_max")
    
    if st.button("🎨 生成热力图", type="primary", use_container_width=True,
                 disabled=not network.solved):
        if x_min >= x_max or y_min >= y_max:
            st.error("起始值必须小于终止值")
        elif x_idx == y_idx:
            st.error("X轴和Y轴请选择不同的变量")
        else:
            total_calc = x_steps * y_steps
            with st.spinner(f"正在进行双变量扫描 ({total_calc} 次计算)..."):
                results = run_heatmap_scan(
                    network, reactions, var_x, var_y,
                    x_min, x_max, x_steps,
                    y_min, y_max, y_steps,
                    T_max_constraint
                )
                st.session_state.heatmap_result = {
                    'var_x': var_x,
                    'var_y': var_y,
                    'results': results,
                    'T_max_constraint': T_max_constraint,
                }
                st.success("✅ 热力图生成完成！")
    
    if not network.solved:
        st.warning("请先完成网络仿真，再生成热力图")
    
    if st.session_state.heatmap_result is not None:
        heatmap_data = st.session_state.heatmap_result
        results = heatmap_data['results']
        var_x = heatmap_data['var_x']
        var_y = heatmap_data['var_y']
        T_max_constraint = heatmap_data['T_max_constraint']
        
        st.markdown("---")
        st.markdown("📊 **收率热力图**")
        
        yield_grid = results['yield_grid'] * 100
        conv_grid = results['conversion_grid'] * 100
        heat_grid = results['heat_grid']
        x_vals = results['x_vals']
        y_vals = results['y_vals']
        
        valid_mask = ~np.isnan(yield_grid)
        
        hover_text = np.empty(yield_grid.shape, dtype=object)
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                if results['success_grid'][i, j]:
                    hover_text[i, j] = (
                        f"<b>{var_x['label']}:</b> {x_vals[j]:.2f} {var_x['unit']}<br>"
                        f"<b>{var_y['label']}:</b> {y_vals[i]:.2f} {var_y['unit']}<br>"
                        f"<b>转化率:</b> {conv_grid[i, j]:.2f}%<br>"
                        f"<b>收率:</b> {yield_grid[i, j]:.2f}%<br>"
                        f"<b>热负荷:</b> {heat_grid[i, j]:.2f} kW"
                    )
                else:
                    hover_text[i, j] = (
                        f"<b>{var_x['label']}:</b> {x_vals[j]:.2f} {var_x['unit']}<br>"
                        f"<b>{var_y['label']}:</b> {y_vals[i]:.2f} {var_y['unit']}<br>"
                        "<b>状态:</b> 求解失败"
                    )
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=yield_grid,
            x=x_vals,
            y=y_vals,
            text=hover_text,
            colorscale='Viridis',
            colorbar=dict(title='收率 (%)'),
            hoverongaps=False,
            name='收率',
            hovertemplate='%{text}<extra></extra>'
        ))
        
        temp_grid = results['max_temp_grid']
        valid_temp = ~np.isnan(temp_grid)
        
        if np.any(valid_temp) and T_max_constraint > 0:
            try:
                from scipy.interpolate import griddata
                
                xi, yi = np.meshgrid(x_vals, y_vals)
                valid_pts = valid_temp
                if np.any(valid_pts):
                    contour_levels = [T_max_constraint]
                    
                    temp_valid = temp_grid[valid_pts]
                    x_valid = xi[valid_pts]
                    y_valid = yi[valid_pts]
                    
                    if len(temp_valid) >= 4:
                        try:
                            xi_fine, yi_fine = np.meshgrid(
                                np.linspace(x_vals.min(), x_vals.max(), 100),
                                np.linspace(y_vals.min(), y_vals.max(), 100)
                            )
                            zi_fine = griddata(
                                (x_valid, y_valid), temp_valid,
                                (xi_fine, yi_fine), method='cubic'
                            )
                            
                            fig.add_trace(go.Contour(
                                x=xi_fine[0],
                                y=yi_fine[:, 0],
                                z=zi_fine,
                                contours=dict(
                                    type='constraint',
                                    operation='>=',
                                    value=T_max_constraint,
                                    coloring='none',
                                    showlabels=True,
                                ),
                                line=dict(color='red', width=2, dash='dash'),
                                name=f'T_max={T_max_constraint:.0f}K',
                                showscale=False,
                            ))
                        except Exception:
                            pass
            except ImportError:
                pass
        
        fig.update_layout(
            xaxis_title=f"{var_x['label']} ({var_x['unit']})",
            yaxis_title=f"{var_y['label']} ({var_y['unit']})",
            height=550,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        valid_yield = yield_grid[valid_mask]
        if len(valid_yield) > 0:
            max_yield = np.max(valid_yield)
            max_idx = np.unravel_index(np.nanargmax(yield_grid), yield_grid.shape)
            max_x = x_vals[max_idx[1]]
            max_y = y_vals[max_idx[0]]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最高收率", f"{max_yield:.2f}%")
            with col2:
                st.metric(f"对应{var_x['label']}", f"{max_x:.2f} {var_x['unit']}")
            with col3:
                st.metric(f"对应{var_y['label']}", f"{max_y:.2f} {var_y['unit']}")
        
        with st.expander("📋 详细数据表", expanded=False):
            table_data = []
            for i, y_val in enumerate(y_vals):
                for j, x_val in enumerate(x_vals):
                    if results['success_grid'][i, j]:
                        table_data.append({
                            f"{var_x['label']}": f"{x_val:.3f}",
                            f"{var_y['label']}": f"{y_val:.3f}",
                            '转化率 (%)': f"{results['conversion_grid'][i, j]*100:.2f}",
                            '收率 (%)': f"{results['yield_grid'][i, j]*100:.2f}",
                            '热负荷 (kW)': f"{results['heat_grid'][i, j]:.2f}",
                            '最高温度 (K)': f"{results['max_temp_grid'][i, j]:.1f}",
                            '状态': '✅',
                        })
                    else:
                        table_data.append({
                            f"{var_x['label']}": f"{x_val:.3f}",
                            f"{var_y['label']}": f"{y_val:.3f}",
                            '转化率 (%)': '-',
                            '收率 (%)': '-',
                            '热负荷 (kW)': '-',
                            '最高温度 (K)': '-',
                            '状态': '❌',
                        })
            st.dataframe(pd.DataFrame(table_data), height=400)


def load_example_network(example_key):
    example = EXAMPLE_NETWORKS[example_key]
    network = ReactorNetwork()
    
    for node_data in example['nodes']:
        node = network.add_node(
            reactor_type=node_data['reactor_type'],
            name=node_data['name'],
            x=node_data['x'],
            y=node_data['y']
        )
        if node:
            if 'V' in node_data:
                node.V = node_data['V']
            if 'T_c' in node_data:
                node.T_c = node_data['T_c']
            if 'UA' in node_data:
                node.UA = node_data['UA']
            if 'n_stages' in node_data:
                node.n_stages = node_data['n_stages']
    
    for conn_data in example['connections']:
        network.add_connection(
            source_id=conn_data['source'],
            target_id=conn_data['target'],
            split_ratio=conn_data['split_ratio']
        )
    
    for node_id, ratio in example['feed_targets'].items():
        network.feed_targets[node_id] = ratio
    
    if network.feed_targets:
        total = sum(network.feed_targets.values())
        if total > 0:
            for k in list(network.feed_targets.keys()):
                network.feed_targets[k] /= total
    
    st.session_state.network = network
    st.session_state.selected_node = None
    st.session_state.selected_conn = None
    st.session_state.network_metrics = None
    st.session_state.optimization_result = None
    st.session_state.network_snapshots = []
    st.session_state.param_scan_result = None
    st.session_state.heatmap_result = None
    st.session_state.ui_salt = st.session_state.get('ui_salt', 0) + 1
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()


def get_temperature_color(T, T_min=280, T_max=500):
    if T is None:
        return '#cccccc'
    T_norm = max(0, min(1, (T - T_min) / (T_max - T_min)))
    
    r = int(30 + T_norm * 200)
    g = int(100 + (1 - T_norm) * 100)
    b = int(200 + (1 - T_norm) * 55)
    
    return f'rgb({r}, {g}, {b})'


def render_network_canvas(network, reactions):
    st.subheader("🎨 网络拓扑画布")
    
    canvas_width = 800
    canvas_height = 500
    
    svg_parts = []
    svg_parts.append(f'<svg width="{canvas_width}" height="{canvas_height}" style="border:2px solid #ddd; border-radius:8px; background:#fafafa;">')
    
    svg_parts.append(f'<defs>')
    svg_parts.append(f'<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
    svg_parts.append(f'<polygon points="0 0, 10 3.5, 0 7" fill="#555"/>')
    svg_parts.append(f'</marker>')
    svg_parts.append(f'</defs>')
    
    feed_x = 30
    for target_id, ratio in network.feed_targets.items():
        node = network.nodes[target_id]
        mid_y = node.y + 40
        
        path = f'M {feed_x} {mid_y} Q {node.x - 80} {mid_y + 30} {node.x - 5} {mid_y}'
        color = '#2196F3'
        stroke_width = 2 + 3 * ratio
        
        svg_parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-dasharray="8,4" marker-end="url(#arrowhead)"/>')
        mid_label_x = (feed_x + node.x) / 2 - 30
        svg_parts.append(f'<text x="{mid_label_x}" y="{mid_y - 10}" font-size="11" fill="{color}">进料 {ratio*100:.0f}%</text>')
    
    for conn in network.connections:
        source = network.nodes.get(conn.source_id)
        target = network.nodes.get(conn.target_id)
        if not source or not target:
            continue
        
        sx, sy = source.x + 80, source.y + 40
        tx, ty = target.x, target.y + 40
        
        if conn.split_ratio < 0.99:
            offset = 15 * hash(conn.id) % 3 - 15
            cy1, cy2 = sy + offset, ty + offset
            path = f'M {sx} {sy} C {(sx+tx)/2} {cy1}, {(sx+tx)/2} {cy2}, {tx} {ty}'
        else:
            path = f'M {sx} {sy} L {tx} {ty}'
        
        is_selected = (st.session_state.selected_conn == conn.id)
        stroke_width = 2.5 if is_selected else 2
        color = '#FF5722' if is_selected else '#666'
        
        svg_parts.append(f'<a href="?sel_conn={conn.id}" style="cursor:pointer;" title="点击编辑连接线 #{conn.id}">')
        svg_parts.append(f'<path d="{path}" fill="none" stroke="transparent" stroke-width="{stroke_width + 14}" />')
        svg_parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke_width}" marker-end="url(#arrowhead)" />')
        svg_parts.append(f'</a>')
        
        mid_x = (sx + tx) / 2
        mid_y = (sy + ty) / 2 - 8
        
        label_texts = [f'分流:{conn.split_ratio*100:.0f}%']
        if network.solved and conn.C_flow is not None:
            label_texts.append(f'C_A={conn.C_flow[0]:.1f}')
        
        for i, txt in enumerate(label_texts):
            svg_parts.append(f'<a href="?sel_conn={conn.id}" style="cursor:pointer;">')
            svg_parts.append(f'<rect x="{mid_x - 40}" y="{mid_y + i*16 - 10}" width="80" height="14" fill="white" stroke="#ddd" rx="3" opacity="0.9"/>')
            svg_parts.append(f'<text x="{mid_x}" y="{mid_y + i*16}" font-size="10" text-anchor="middle" fill="#333" pointer-events="none">{txt}</text>')
            svg_parts.append(f'</a>')
    
    for nid, node in network.nodes.items():
        is_selected = (st.session_state.selected_node == nid)
        border_color = '#4CAF50' if is_selected else '#333'
        border_width = 3 if is_selected else 2
        
        fill_color = '#ffffff'
        if network.solved and node.T_out is not None:
            fill_color = get_temperature_color(node.T_out)
        
        type_icon = {
            'CSTR': '⚪',
            'PFR': '📏',
            '半间歇': '⏳',
            '多级CSTR': '🔗'
        }.get(node.reactor_type, '⚪')
        
        svg_parts.append(f'<a href="?sel_node={nid}" style="cursor:pointer;" title="点击配置 {node.name}">')
        svg_parts.append(f'<g transform="translate({node.x}, {node.y})">')
        svg_parts.append(f'<rect x="0" y="0" width="80" height="80" rx="10" ry="10" fill="{fill_color}" stroke="{border_color}" stroke-width="{border_width}" opacity="0.9"/>')
        svg_parts.append(f'<text x="40" y="25" font-size="20" text-anchor="middle" pointer-events="none">{type_icon}</text>')
        svg_parts.append(f'<text x="40" y="45" font-size="13" font-weight="bold" text-anchor="middle" fill="#333" pointer-events="none">{node.name}</text>')
        svg_parts.append(f'<text x="40" y="60" font-size="9" text-anchor="middle" fill="#555" pointer-events="none">{node.reactor_type}</text>')
        if network.solved and node.T_out is not None:
            svg_parts.append(f'<text x="40" y="74" font-size="10" font-weight="bold" text-anchor="middle" fill="#c62828" pointer-events="none">T={node.T_out:.0f}K</text>')
        svg_parts.append(f'</g>')
        svg_parts.append(f'</a>')
    
    legend_y = canvas_height - 60
    svg_parts.append(f'<g transform="translate(20, {legend_y})">')
    svg_parts.append(f'<rect x="0" y="0" width="300" height="50" fill="white" stroke="#ccc" rx="5" opacity="0.95"/>')
    svg_parts.append(f'<text x="10" y="18" font-size="11" font-weight="bold" fill="#333">温度图例 (冷→暖):</text>')
    for i in range(10):
        t = i / 9
        x = 10 + i * 28
        r = int(30 + t * 200)
        g = int(100 + (1 - t) * 100)
        b = int(200 + (1 - t) * 55)
        svg_parts.append(f'<rect x="{x}" y="25" width="26" height="18" fill="rgb({r},{g},{b})" stroke="#999"/>')
        if i == 0:
            svg_parts.append(f'<text x="{x}" y="52" font-size="9" fill="#333">280K</text>')
        elif i == 9:
            svg_parts.append(f'<text x="{x - 15}" y="52" font-size="9" fill="#333">500K</text>')
    svg_parts.append(f'</g>')
    
    svg_parts.append('</svg>')
    
    st.markdown(''.join(svg_parts), unsafe_allow_html=True)
    
    col_canvas1, col_canvas2, col_canvas3 = st.columns(3)
    with col_canvas1:
        st.markdown("**节点操作**")
        node_options = [(nid, f"{n.name} ({n.reactor_type})") for nid, n in network.nodes.items()]
        if node_options:
            selected = st.selectbox("选择节点", [nid for nid, _ in node_options],
                                   format_func=lambda x: dict(node_options)[x],
                                   key=f"canvas_node_select_{st.session_state.ui_salt}")
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                if st.button("📝 配置节点", use_container_width=True, key=f"cnf_node_{st.session_state.ui_salt}"):
                    st.session_state.selected_node = selected
                    st.session_state.selected_conn = None
            with col_sel2:
                if st.button("🗑️ 删除节点", use_container_width=True, key=f"del_node_{st.session_state.ui_salt}"):
                    network.remove_node(selected)
                    if st.session_state.selected_node == selected:
                        st.session_state.selected_node = None
                    st.rerun()
        else:
            st.info("画布为空，请添加节点")
    
    with col_canvas2:
        st.markdown("**连接操作**")
        conn_options = [(c.id, f"{network.nodes.get(c.source_id, ReactorNode(-1,'')).name}→{network.nodes.get(c.target_id, ReactorNode(-1,'')).name}") 
                       for c in network.connections if c.source_id in network.nodes and c.target_id in network.nodes]
        if conn_options:
            selected_c = st.selectbox("选择连接", [cid for cid, _ in conn_options],
                                     format_func=lambda x: dict(conn_options)[x],
                                     key=f"canvas_conn_select_{st.session_state.ui_salt}")
            col_cs1, col_cs2 = st.columns(2)
            with col_cs1:
                if st.button("📐 修改分流比", use_container_width=True, key=f"cnf_conn_{st.session_state.ui_salt}"):
                    st.session_state.selected_conn = selected_c
                    st.session_state.selected_node = None
            with col_cs2:
                if st.button("❌ 删除连接", use_container_width=True, key=f"del_conn_{st.session_state.ui_salt}"):
                    network.remove_connection(selected_c)
                    if st.session_state.selected_conn == selected_c:
                        st.session_state.selected_conn = None
                    st.rerun()
        else:
            st.info("暂无连接，请添加节点间连接")
    
    with col_canvas3:
        st.markdown("**快捷操作**")
        with st.popover("➕ 添加反应器节点", use_container_width=True):
            st.markdown("**选择反应器类型**")
            for rt in REACTOR_TYPES:
                if st.button(f"添加 {REACTOR_TYPE_CN.get(rt, rt)}", key=f"add_{rt}_{st.session_state.ui_salt}", use_container_width=True):
                    network.add_node(reactor_type=rt)
                    st.success(f"已添加 {rt} 反应器")
                    st.rerun()
        with st.popover("🔗 建立节点连接", use_container_width=True):
            st.markdown("**配置连接参数**")
            if len(network.nodes) >= 2:
                src_ids = list(network.nodes.keys())
                src_id = st.selectbox("源节点", src_ids, format_func=lambda x: network.nodes[x].name, key=f"conn_src_{st.session_state.ui_salt}")
                tgt_ids = [nid for nid in src_ids if nid != src_id]
                if tgt_ids:
                    tgt_id = st.selectbox("目标节点", tgt_ids, format_func=lambda x: network.nodes[x].name, key=f"conn_tgt_{st.session_state.ui_salt}")
                    ratio = st.slider("分流比", 0.0, 1.0, 1.0, 0.01, key=f"conn_ratio_{st.session_state.ui_salt}")
                    if st.button("✅ 创建连接", use_container_width=True, key=f"make_conn_{st.session_state.ui_salt}"):
                        conn, err = network.add_connection(src_id, tgt_id, ratio)
                        if err:
                            st.error(err)
                        else:
                            st.success(f"已创建连接: {network.nodes[src_id].name}→{network.nodes[tgt_id].name}")
                        st.rerun()
            else:
                st.info("至少需要2个节点才能建立连接")
        with st.popover("📥 设置进料目标", use_container_width=True):
            st.markdown("**进料分配设置**")
            nid_list = list(network.nodes.keys())
            for nid in nid_list:
                node = network.nodes[nid]
                current_ratio = network.feed_targets.get(nid, 0.0)
                new_ratio = st.slider(f"{node.name} 进料比例", 0.0, 1.0, float(current_ratio), 0.01, key=f"feed_{nid}_{st.session_state.ui_salt}")
                if abs(new_ratio - current_ratio) > 0.001:
                    if new_ratio > 0:
                        network.set_feed_target(nid, new_ratio)
                    else:
                        network.remove_feed_target(nid)
                    st.rerun()
            if st.button("⚖️ 自动归一化", use_container_width=True, key=f"norm_feed_{st.session_state.ui_salt}"):
                network._normalize_feed_ratios()
                st.rerun()


def render_config_panel(network, reactions):
    st.subheader("⚙️ 网络配置面板")
    
    col_list, col_conn = st.columns(2)
    
    with col_list:
        st.markdown("**📋 节点列表**")
        if not network.nodes:
            st.info("暂无节点，请在画布上添加反应器")
        else:
            node_data = []
            for nid, node in network.nodes.items():
                row = {
                    'ID': nid,
                    '名称': node.name,
                    '类型': node.reactor_type,
                    '体积 (m³)': f"{node.V:.2f}" if node.reactor_type in ["CSTR", "半间歇"] else "-",
                    '冷却温度 (K)': f"{node.T_c:.1f}",
                    'UA (W/K)': f"{node.UA:.0f}",
                }
                if st.session_state.selected_node == nid:
                    row['名称'] = "👉 " + row['名称']
                node_data.append(row)
            st.table(pd.DataFrame(node_data).set_index('ID'))
        
        col_add1, col_add2, col_add3, col_add4 = st.columns(4)
        with col_add1:
            if st.button("➕ CSTR", use_container_width=True):
                network.add_node("CSTR"); st.rerun()
        with col_add2:
            if st.button("➕ PFR", use_container_width=True):
                network.add_node("PFR"); st.rerun()
        with col_add3:
            if st.button("➕ 半间歇", use_container_width=True):
                network.add_node("半间歇"); st.rerun()
        with col_add4:
            if st.button("➕ 多级CSTR", use_container_width=True):
                network.add_node("多级CSTR"); st.rerun()
    
    with col_conn:
        st.markdown("**🔗 连接关系表**")
        if not network.connections:
            st.info("暂无连接")
        else:
            conn_data = []
            for c in network.connections:
                src_name = network.nodes[c.source_id].name if c.source_id in network.nodes else "?"
                tgt_name = network.nodes[c.target_id].name if c.target_id in network.nodes else "?"
                row = {
                    'ID': c.id,
                    '源节点': src_name,
                    '目标节点': tgt_name,
                    '分流比': f"{c.split_ratio:.3f}",
                }
                if st.session_state.selected_conn == c.id:
                    row['源节点'] = "👉 " + row['源节点']
                conn_data.append(row)
            st.table(pd.DataFrame(conn_data).set_index('ID'))
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            with st.popover("➕ 添加连接", use_container_width=True):
                if len(network.nodes) >= 2:
                    ids = list(network.nodes.keys())
                    s = st.selectbox("源", ids, format_func=lambda x: network.nodes[x].name, key=f"panel_s_{st.session_state.ui_salt}")
                    ts = [i for i in ids if i != s]
                    if ts:
                        t = st.selectbox("目标", ts, format_func=lambda x: network.nodes[x].name, key=f"panel_t_{st.session_state.ui_salt}")
                        r = st.slider("分流比", 0.05, 1.0, 1.0, 0.01, key=f"panel_r_{st.session_state.ui_salt}")
                        if st.button("创建", use_container_width=True, key=f"panel_create_{st.session_state.ui_salt}"):
                            _, err = network.add_connection(s, t, r)
                            if err: st.error(err)
                            else: st.rerun()
                else:
                    st.info("需要至少2个节点")
        with col_btn2:
            if st.button("🗑️ 删除选中", use_container_width=True, disabled=st.session_state.selected_conn is None, key=f"panel_del_{st.session_state.ui_salt}"):
                if st.session_state.selected_conn is not None:
                    network.remove_connection(st.session_state.selected_conn)
                    st.session_state.selected_conn = None
                    st.rerun()
    
    st.markdown("---")
    
    if st.session_state.selected_node is not None and st.session_state.selected_node in network.nodes:
        node = network.nodes[st.session_state.selected_node]
        st.markdown(f"**🔧 节点配置 - {node.name}**")
        
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1:
            node.name = st.text_input("节点名称", node.name)
            node.reactor_type = st.selectbox("反应器类型", REACTOR_TYPES, 
                                            index=REACTOR_TYPES.index(node.reactor_type) if node.reactor_type in REACTOR_TYPES else 0,
                                            format_func=lambda x: REACTOR_TYPE_CN.get(x, x))
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                node.x = st.number_input("画布X坐标", 0, 700, int(node.x), 10)
            with col_x2:
                node.y = st.number_input("画布Y坐标", 0, 400, int(node.y), 10)
        
        with col_n2:
            if node.reactor_type == "CSTR":
                node.V = st.number_input("体积 V (m³)", 0.1, 10.0, float(node.V), 0.1)
                node.F = st.number_input("流量 F (m³/s)", 0.001, 1.0, float(node.F), 0.001, format="%.3f")
            elif node.reactor_type == "PFR":
                node.L = st.number_input("管长 L (m)", 1.0, 50.0, float(node.L), 1.0)
                node.A_cross = st.number_input("横截面积 (m²)", 0.001, 0.1, float(node.A_cross), 0.001)
                node.u = st.number_input("流速 u (m/s)", 0.01, 5.0, float(node.u), 0.01)
                node.perimeter = st.number_input("管周长 (m)", 0.05, 1.0, float(node.perimeter), 0.01)
                node.UA_per_L = st.number_input("单位长度UA (W/(m·K))", 100.0, 5000.0, float(node.UA_per_L), 100.0)
            elif node.reactor_type == "半间歇":
                node.V_max = st.number_input("最大体积 (m³)", 0.5, 10.0, float(node.V_max), 0.1)
                node.F_in = st.number_input("进料流量 (m³/s)", 0.001, 0.1, float(node.F_in), 0.001, format="%.3f")
            elif node.reactor_type == "多级CSTR":
                node.n_stages = st.slider("级数", 2, 5, int(node.n_stages), 1)
                node.F = st.number_input("总流量 F (m³/s)", 0.001, 1.0, float(node.F), 0.001, format="%.3f")
        
        with col_n3:
            node.rho_cp = st.number_input("ρCp (J/(m³·K))", 1e3, 1e4, float(node.rho_cp), 100.0)
            node.T_c = st.slider("冷却温度 T_c (K)", 250.0, 360.0, float(node.T_c), 1.0)
            node.UA = st.number_input("传热系数 UA (W/K)", 100.0, 10000.0, float(node.UA), 100.0)
            default_feed = node.feed_temperature if node.feed_temperature else network.T_feed
            node.feed_temperature = st.number_input("独立进料温度 (K)", 0.0, 500.0, float(default_feed), 1.0,
                                                    help="设为0表示使用全局进料温度")
            if node.feed_temperature <= 0:
                node.feed_temperature = None
        
        if st.button("✅ 关闭配置", use_container_width=True):
            st.session_state.selected_node = None
            st.rerun()
    
    elif st.session_state.selected_conn is not None:
        conn = None
        for c in network.connections:
            if c.id == st.session_state.selected_conn:
                conn = c
                break
        if conn:
            src_name = network.nodes[conn.source_id].name if conn.source_id in network.nodes else "?"
            tgt_name = network.nodes[conn.target_id].name if conn.target_id in network.nodes else "?"
            st.markdown(f"**🔧 连接配置 - {src_name} → {tgt_name}**")
            conn.split_ratio = st.slider("分流比", 0.01, 1.0, float(conn.split_ratio), 0.01,
                                        help="同一源节点的所有输出分流比之和应为1.0")
            out_conns = network.get_outgoing_connections(conn.source_id)
            total = sum(c.split_ratio for c in out_conns)
            st.info(f"当前源节点输出分流比总和: {total:.3f} {'✅' if abs(total - 1.0) < 0.01 else '⚠️ 应为1.0'}")
            if st.button("✅ 完成修改", use_container_width=True):
                network._normalize_split_ratios()
                st.session_state.selected_conn = None
                st.rerun()
    
    st.markdown("---")
    
    validation_error = network.validate_network()
    if validation_error:
        st.warning(f"⚠️ {validation_error}")
    else:
        st.success("✅ 网络拓扑校验通过")


def render_feed_config(network):
    st.subheader("📥 总进料条件")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        network.T_feed = st.slider("进料温度 T_feed (K)", 250.0, 420.0, float(network.T_feed), 1.0, key=f"T_feed_{st.session_state.ui_salt}")
        network.F_feed = st.number_input("总进料流量 F (m³/s)", 0.001, 1.0, float(network.F_feed), 0.001, format="%.3f", key=f"F_feed_{st.session_state.ui_salt}")
        network.rho_cp = st.number_input("ρCp (J/(m³·K))", 1e3, 1e4, float(network.rho_cp), 100.0, key=f"rho_cp_{st.session_state.ui_salt}")
    with col_f2:
        st.markdown("**进料浓度 (mol/m³)**")
        for i, comp in enumerate(COMPONENTS):
            network.C_feed[i] = st.number_input(f"C_{comp}", 0.0, 2000.0, float(network.C_feed[i]), 1.0, key=f"feed_C_{comp}_{st.session_state.ui_salt}")
    with col_f3:
        st.markdown("**进料分配比例**")
        total_ratio = 0.0
        node_ids = list(network.nodes.keys())
        for nid in node_ids:
            node = network.nodes[nid]
            current = network.feed_targets.get(nid, 0.0)
            new_val = st.slider(f"进入 {node.name}", 0.0, 1.0, float(current), 0.01, key=f"feed_slider_{nid}_{st.session_state.ui_salt}")
            if abs(new_val - current) > 0.001:
                if new_val > 0:
                    network.feed_targets[nid] = new_val
                elif nid in network.feed_targets:
                    del network.feed_targets[nid]
            total_ratio += new_val
        st.info(f"进料总和: {total_ratio*100:.1f}% {'✅' if abs(total_ratio - 1.0) < 0.02 else '⚠️ 建议为100%'}")
        if st.button("⚖️ 归一化进料比例", use_container_width=True, key=f"norm_feed_btn_{st.session_state.ui_salt}"):
            network._normalize_feed_ratios()
            st.rerun()


def render_simulation_section(network, reactions):
    st.subheader("🚀 网络仿真求解")
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        solve_method = st.radio("求解策略", ["自动检测", "强制循环迭代（Wegstein加速）"], index=0)
        if st.button("▶️ 开始仿真", type="primary", use_container_width=True):
            with st.spinner("正在求解整个网络..."):
                ok = network.solve(reactions)
                if ok:
                    if network.solve_error:
                        st.warning(network.solve_error)
                    else:
                        st.success("✅ 仿真求解完成！")
                    st.session_state.network_metrics = network.get_metrics()
                    st.session_state.optimization_result = None
                else:
                    st.error(f"❌ 仿真失败: {network.solve_error}")
    
    with col_s2:
        if network.solved and network.solve_error is None:
            st.success("✅ 求解状态：已完成")
            if network._has_cycle():
                st.info("🔄 检测到循环结构，已使用Wegstein迭代加速法")
            else:
                st.info("📐 无循环网络，已使用拓扑排序法")
        elif network.solve_error:
            st.warning(f"⚠️ {network.solve_error}")
        else:
            st.info("请点击\"开始仿真\"按钮进行求解")


def render_performance_panel(network, metrics):
    if metrics is None:
        st.warning("请先运行仿真以获取性能指标")
        return
    
    st.subheader("📊 网络性能评估面板")
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("总转化率", f"{metrics['total_conversion']*100:.2f}%")
    with col_m2:
        st.metric("总选择性", f"{metrics['total_selectivity']*100:.2f}%")
    with col_m3:
        st.metric("总收率", f"{metrics['total_yield']*100:.2f}%")
    with col_m4:
        st.metric("网络热负荷", f"{metrics['total_heat_load']:.2f} kW")
    with col_m5:
        st.metric("最高节点温度", f"{metrics['max_temperature']:.1f} K")
    
    st.markdown("---")
    
    col_sankey, col_bar = st.columns([2, 1])
    
    with col_sankey:
        st.markdown("**🔄 物料流向桑基图**")
        try:
            C_A0 = metrics['C_feed_value'] * metrics['F_feed_value']
            labels = ['新鲜进料 A', '消耗的 A', '生成产物 B', '生成产物 C', '未反应 A', '其他产物']
            
            a_consumed = metrics['total_A_consumed']
            a_unreacted = max(0, C_A0 - a_consumed)
            b_prod = metrics['total_B_produced']
            c_prod = metrics['total_C_produced']
            other = max(0, a_consumed - b_prod - c_prod)
            
            source = [0, 1, 1, 1, 0]
            target = [1, 2, 3, 5, 4]
            value = [a_consumed, b_prod, c_prod, other, a_unreacted]
            
            valid = [i for i, v in enumerate(value) if v > 0.01]
            source = [source[i] for i in valid]
            target = [target[i] for i in valid]
            value = [value[i] for i in valid]
            
            fig_sankey = go.Figure(go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=labels,
                    color=['#2196F3', '#FF9800', '#4CAF50', '#f44336', '#9E9E9E', '#9C27B0']
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=['rgba(33,150,243,0.4)', 'rgba(255,152,0,0.4)', 'rgba(76,175,80,0.4)',
                           'rgba(156,39,176,0.4)', 'rgba(158,158,158,0.4)']
                )
            ))
            fig_sankey.update_layout(height=400, title_text="物料流向与转化分布", font_size=12)
            st.plotly_chart(fig_sankey, use_container_width=True)
        except Exception as e:
            st.info(f"桑基图渲染中: {e}")
    
    with col_bar:
        st.markdown("**📊 各节点转化率贡献**")
        node_names = []
        conversions = []
        for nid, conv in metrics['node_conversions'].items():
            node = network.nodes.get(nid)
            if node:
                node_names.append(node.name)
                conversions.append(conv * 100)
        
        if node_names:
            colors_bar = [get_temperature_color(network.nodes[nid].T_out) 
                         if network.nodes[nid].T_out else '#cccccc' 
                         for nid in metrics['node_conversions'].keys()]
            fig_bar = go.Figure(go.Bar(
                x=node_names,
                y=conversions,
                marker_color=colors_bar,
                text=[f"{v:.1f}%" for v in conversions],
                textposition='auto',
            ))
            fig_bar.update_layout(
                yaxis_title='节点转化率 (%)',
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    if network.solved:
        st.markdown("**📋 节点仿真结果详情**")
        detail_data = []
        for nid, node in network.nodes.items():
            if node.C_out is not None and node.T_out is not None:
                row = {
                    '节点': node.name,
                    '类型': node.reactor_type,
                    '温度 (K)': f"{node.T_out:.1f}",
                    'C_A 出口': f"{node.C_out[0]:.2f}",
                    'C_B 出口': f"{node.C_out[1]:.2f}",
                    'C_C 出口': f"{node.C_out[2]:.2f}",
                    '节点转化率': f"{node.conversion*100:.1f}%",
                    '冷却需求 (kW)': f"{node.Q_cooling/1000:.2f}",
                }
                detail_data.append(row)
        if detail_data:
            st.table(pd.DataFrame(detail_data))
    
    st.markdown("---")
    render_snapshot_comparison()


def render_optimization_panel(network, reactions):
    st.subheader("⚡ 网络优化")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        st.markdown("**🎯 优化目标与约束**")
        objective = st.selectbox("优化目标", 
                                ["max_yield 最大化总收率", 
                                 "min_heat 最小化热负荷",
                                 "max_conv 最大化转化率"],
                                index=0)
        obj_key = objective.split()[0]
        
        T_max = st.number_input("温度上限 T_max (K)", 350.0, 600.0, 473.15, 1.0)
        X_min = st.slider("最低总转化率要求 (%)", 0.0, 95.0, 30.0, 1.0) / 100.0
        V_total_max = st.number_input("总容量上限 (m³)", 1.0, 50.0, 10.0, 0.5)
        
        st.markdown("**📐 变量范围**")
        T_c_low = st.number_input("冷却温度下限 (K)", 240.0, 280.0, 260.0, 1.0)
        T_c_high = st.number_input("冷却温度上限 (K)", 300.0, 380.0, 340.0, 1.0)
        T_f_low = st.number_input("进料温度下限 (K)", 260.0, 320.0, 280.0, 1.0)
        T_f_high = st.number_input("进料温度上限 (K)", 350.0, 450.0, 380.0, 1.0)
    
    with col_opt2:
        st.markdown("**📊 当前状态与控制**")
        var_count = 0
        for nid, node in network.nodes.items():
            var_count += 2
            out_conns = network.get_outgoing_connections(nid)
            if len(out_conns) >= 2:
                var_count += len(out_conns) - 1
        st.info(f"**优化维度:** {var_count} 个变量\n\n"
                f"- 每个节点: 冷却温度 + 进料温度 (2个)\n"
                f"- 并联输出: 分流比 (N-1个/节点)")
        
        if st.button("🚀 开始优化", type="primary", use_container_width=True, 
                    disabled=not network.solved):
            with st.spinner(f"正在进行{var_count}维优化搜索..."):
                result, error = network.optimize(
                    reactions,
                    objective=obj_key,
                    T_max=T_max,
                    X_min=X_min,
                    V_total_max=V_total_max,
                    T_c_bounds=(T_c_low, T_c_high),
                    T_feed_bounds=(T_f_low, T_f_high),
                )
                if error:
                    st.error(error)
                else:
                    st.session_state.optimization_result = result
                    if result['success']:
                        st.success("✅ 优化完成!")
                    else:
                        st.warning(f"⚠️ {result['message']}")
        
        if not network.solved:
            st.warning("请先完成仿真，再进行优化")
    
    if st.session_state.optimization_result:
        result = st.session_state.optimization_result
        base = result['base_metrics']
        opt = result['optimized_metrics']
        
        st.markdown("---")
        st.markdown("**📈 优化前后参数对比**")
        st.table(pd.DataFrame(result['params']))
        
        st.markdown("**📊 性能指标对比**")
        compare_data = [
            {'指标': '总转化率 (%)', '优化前': f"{base['total_conversion']*100:.2f}",
             '优化后': f"{opt['total_conversion']*100:.2f}",
             '变化': f"{(opt['total_conversion']-base['total_conversion'])*100:+.2f}"},
            {'指标': '总选择性 (%)', '优化前': f"{base['total_selectivity']*100:.2f}",
             '优化后': f"{opt['total_selectivity']*100:.2f}",
             '变化': f"{(opt['total_selectivity']-base['total_selectivity'])*100:+.2f}"},
            {'指标': '总收率 (%)', '优化前': f"{base['total_yield']*100:.2f}",
             '优化后': f"{opt['total_yield']*100:.2f}",
             '变化': f"{(opt['total_yield']-base['total_yield'])*100:+.2f}"},
            {'指标': '热负荷 (kW)', '优化前': f"{base['total_heat_load']:.2f}",
             '优化后': f"{opt['total_heat_load']:.2f}",
             '变化': f"{(opt['total_heat_load']-base['total_heat_load']):+.2f}"},
            {'指标': '最高温度 (K)', '优化前': f"{base['max_temperature']:.1f}",
             '优化后': f"{opt['max_temperature']:.1f}",
             '变化': f"{(opt['max_temperature']-base['max_temperature']):+.1f}"},
        ]
        st.table(pd.DataFrame(compare_data))
        
        col_rad1, col_rad2 = st.columns(2)
        with col_rad1:
            categories = ['转化率', '选择性', '收率']
            base_vals = [base['total_conversion']*100, base['total_selectivity']*100, base['total_yield']*100]
            opt_vals = [opt['total_conversion']*100, opt['total_selectivity']*100, opt['total_yield']*100]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=base_vals, theta=categories, fill='toself', name='优化前', line=dict(color='blue')))
            fig_radar.add_trace(go.Scatterpolar(r=opt_vals, theta=categories, fill='toself', name='优化后', line=dict(color='red')))
            max_val = max(max(base_vals), max(opt_vals)) * 1.1
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_val])),
                                   title="关键性能雷达图对比", height=400)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col_rad2:
            fig_compare = go.Figure()
            x = ['转化率', '选择性', '收率']
            y1 = [base['total_conversion']*100, base['total_selectivity']*100, base['total_yield']*100]
            y2 = [opt['total_conversion']*100, opt['total_selectivity']*100, opt['total_yield']*100]
            fig_compare.add_trace(go.Bar(name='优化前', x=x, y=y1, marker_color='rgba(100,149,237,0.7)', text=[f'{v:.1f}%' for v in y1], textposition='auto'))
            fig_compare.add_trace(go.Bar(name='优化后', x=x, y=y2, marker_color='rgba(220,20,60,0.7)', text=[f'{v:.1f}%' for v in y2], textposition='auto'))
            fig_compare.update_layout(title='优化前后核心指标对比', barmode='group', height=400, yaxis_title='%')
            st.plotly_chart(fig_compare, use_container_width=True)


def page_reactor_network(reactions):
    st.header("🔗 反应器网络拓扑设计与优化")
    
    init_network_state()
    network = st.session_state.network
    
    params_changed = False
    try:
        if 'sel_node' in st.query_params:
            val = st.query_params.sel_node
            try:
                nid = int(val)
                if nid in network.nodes and st.session_state.selected_node != nid:
                    st.session_state.selected_node = nid
                    st.session_state.selected_conn = None
                    params_changed = True
            except (ValueError, TypeError):
                pass
            del st.query_params['sel_node']
        if 'sel_conn' in st.query_params:
            val = st.query_params.sel_conn
            try:
                cid = int(val)
                if any(c.id == cid for c in network.connections) and st.session_state.selected_conn != cid:
                    st.session_state.selected_conn = cid
                    st.session_state.selected_node = None
                    params_changed = True
            except (ValueError, TypeError):
                pass
            del st.query_params['sel_conn']
    except Exception:
        pass
    if params_changed:
        st.rerun()
    
    st.markdown("---")
    
    with st.expander("📚 内置示例网络方案", expanded=True):
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            st.markdown("**方案一: 两级串联**")
            st.caption(EXAMPLE_NETWORKS["方案一：两级串联"]["description"])
            if st.button("📥 加载两级串联", use_container_width=True, key=f"ex1_{st.session_state.ui_salt}"):
                load_example_network("方案一：两级串联")
                st.rerun()
        with col_ex2:
            st.markdown("**方案二: 并联分流**")
            st.caption(EXAMPLE_NETWORKS["方案二：并联分流"]["description"])
            if st.button("📥 加载并联分流", use_container_width=True, key=f"ex2_{st.session_state.ui_salt}"):
                load_example_network("方案二：并联分流")
                st.rerun()
        with col_ex3:
            st.markdown("**方案三: 带回流循环**")
            st.caption(EXAMPLE_NETWORKS["方案三：带回流循环"]["description"])
            if st.button("📥 加载回流循环", use_container_width=True, key=f"ex3_{st.session_state.ui_salt}"):
                load_example_network("方案三：带回流循环")
                st.rerun()
    
    st.markdown("---")
    
    render_feed_config(network)
    
    st.markdown("---")
    
    tab_canvas, tab_config = st.tabs(["🎨 画布设计", "⚙️ 配置面板"])
    with tab_canvas:
        render_network_canvas(network, reactions)
    with tab_config:
        render_config_panel(network, reactions)
    
    st.markdown("---")
    
    render_simulation_section(network, reactions)
    
    st.markdown("---")
    
    tab_perf, tab_opt, tab_scan = st.tabs(["📊 性能评估", "⚡ 网络优化", "🔬 参数扫描"])
    with tab_perf:
        render_performance_panel(network, st.session_state.network_metrics)
    with tab_opt:
        render_optimization_panel(network, reactions)
    with tab_scan:
        render_parameter_sweep(network, reactions)
        st.markdown("---")
        render_dual_variable_heatmap(network, reactions)
    
    st.markdown("---")
    with st.expander("ℹ️ 使用说明", expanded=False):
        st.markdown("""
**快速开始步骤:**

1. **加载示例** - 从上方内置方案中选择一个，快速了解功能
2. **添加节点** - 在画布上放置1-6个反应器节点
3. **建立连接** - 配置节点间的物料流向和分流比
4. **设置进料** - 配置总进料条件和分配比例
5. **开始仿真** - 点击"开始仿真"计算稳态分布
6. **查看结果** - 在性能面板查看转化率、选择性等指标
7. **自动优化** - 设置目标和约束，一键寻找最优操作

**网络校验规则:**
- 不允许孤立节点（每个节点至少有一个输入或输出）
- 同一节点的并联输出分流比之和必须为1.0（容差0.01）
- 最多支持6个反应器节点同时参与计算
""")


if __name__ == "__main__":
    main()


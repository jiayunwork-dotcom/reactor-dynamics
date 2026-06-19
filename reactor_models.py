import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

R_GAS = 8.314


def reaction_rates(C, T, reactions):
    rates = []
    for rxn in reactions:
        k = rxn['A'] * np.exp(-rxn['Ea'] * 1e3 / (R_GAS * T))
        order_product = 1.0
        for comp in ['A', 'B', 'C', 'D', 'E']:
            idx = ['A', 'B', 'C', 'D', 'E'].index(comp)
            conc = max(C[idx], 0.0)
            if comp in rxn.get('reactants', []):
                ri = rxn['reactants'].index(comp)
                order_product *= conc ** rxn['reactant_orders'][ri]
        rates.append(k * order_product)
    return rates


def stoich_net(reactions):
    net = {c: 0.0 for c in ['A', 'B', 'C', 'D', 'E']}
    for rxn in reactions:
        for comp, coeff in rxn.get('reactant_coeffs', {}).items():
            net[comp] -= coeff
        for comp, coeff in rxn.get('product_coeffs', {}).items():
            net[comp] += coeff
    return net


def cstr_ode(t, y, params, reactions, disturbances=None, pid_func=None):
    C = np.clip(y[:5], 0, None)
    T = y[5]
    V = params.get('V', 1.0)
    F = params.get('F', 1.0)
    tau = V / F
    C_f = params.get('C_f', np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    T_f = params.get('T_f', 300.0)
    rho_cp = params.get('rho_cp', 4.0e3)
    T_c = params.get('T_c', 290.0)
    UA = params.get('UA', 2000.0)

    if disturbances:
        for dist in disturbances:
            if dist['time'] <= t:
                if dist['type'] == 'T_f':
                    T_f = dist['value']
                elif dist['type'] == 'F':
                    F = dist['value']
                    tau = V / F
                elif dist['type'] == 'C_f':
                    C_f = dist['value']

    if pid_func:
        UA_eff = pid_func(t, T)
    else:
        UA_eff = UA

    rates = reaction_rates(C, T, reactions)
    net = stoich_net(reactions)

    dCdt = np.zeros(5)
    for i, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
        dCdt[i] = (C_f[i] - C[i]) / tau + net[comp] * sum(rates)

    Q_gen = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V
    Q_rem = rho_cp * F * (T - T_f) + UA_eff * (T - T_c)

    dTdt = (Q_gen - Q_rem) / (rho_cp * V)

    dydt = np.zeros(6)
    dydt[:5] = dCdt
    dydt[5] = dTdt
    return dydt


def pfr_ode(z, y, params, reactions):
    C = np.clip(y[:5], 0, None)
    T = y[5]
    A_cross = params.get('A_cross', 0.01)
    u = params.get('u', 1.0)
    rho_cp = params.get('rho_cp', 4.0e3)
    T_c = params.get('T_c', 290.0)
    UA_per_L = params.get('UA_per_L', 2000.0)
    P = params.get('perimeter', 0.1)

    rates = reaction_rates(C, T, reactions)
    net = stoich_net(reactions)

    dCdz = np.zeros(5)
    for i, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
        dCdz[i] = net[comp] * sum(rates) / u

    Q_gen_vol = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates))
    Q_rem_vol = UA_per_L * (T - T_c) / A_cross * P

    dTdz = (Q_gen_vol - Q_rem_vol) / (rho_cp * u)

    dydz = np.zeros(6)
    dydz[:5] = dCdz
    dydz[5] = dTdz
    return dydz


def semibatch_ode(t, y, params, reactions, disturbances=None):
    C = np.clip(y[:5], 0, None)
    T = y[5]
    V = max(y[6], 1e-10)
    F_in = params.get('F_in', 0.01)
    C_f = params.get('C_f', np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    T_f = params.get('T_f', 300.0)
    rho_cp = params.get('rho_cp', 4.0e3)
    T_c = params.get('T_c', 290.0)
    UA = params.get('UA', 2000.0)
    V_max = params.get('V_max', 2.0)

    if V >= V_max:
        F_in = 0.0

    if disturbances:
        for dist in disturbances:
            if dist['time'] <= t:
                if dist['type'] == 'F_in':
                    F_in = dist['value']
                elif dist['type'] == 'T_f':
                    T_f = dist['value']

    rates = reaction_rates(C, T, reactions)
    net = stoich_net(reactions)

    dVCdt = np.zeros(5)
    for i, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
        dVCdt[i] = F_in * C_f[i] + V * net[comp] * sum(rates)

    Q_gen = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V
    Q_rem = rho_cp * F_in * (T - T_f) + UA * (T - T_c)

    dVTdt = Q_gen - Q_rem

    dVdt = F_in

    dydt = np.zeros(7)
    dydt[:5] = dVCdt
    dydt[5] = dVTdt / (rho_cp * V) if V > 1e-10 else 0.0
    dydt[6] = dVdt
    return dydt


def multicstr_ode(t, y, params, reactions, n_stages, disturbances=None, pid_func=None):
    n_vars = 6 * n_stages
    dydt = np.zeros(n_vars)

    C_f = params.get('C_f', np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    T_f = params.get('T_f', 300.0)
    rho_cp = params.get('rho_cp', 4.0e3)

    if disturbances:
        for dist in disturbances:
            if dist['time'] <= t:
                if dist['type'] == 'T_f':
                    T_f = dist['value']

    for i in range(n_stages):
        idx = i * 6
        C_i = np.clip(y[idx:idx + 5], 0, None)
        T_i = y[idx + 5]

        V_i = params.get(f'V_{i}', params.get('V', 1.0))
        F_i = params.get('F', 1.0)
        tau_i = V_i / F_i
        T_c_i = params.get(f'T_c_{i}', params.get('T_c', 290.0))
        UA_i = params.get(f'UA_{i}', params.get('UA', 2000.0))

        if pid_func and i == 0:
            UA_i = pid_func(t, T_i)

        if i == 0:
            C_in = C_f
            T_in = T_f
        else:
            prev_idx = (i - 1) * 6
            C_in = y[prev_idx:prev_idx + 5]
            T_in = y[prev_idx + 5]

        rates = reaction_rates(C_i, T_i, reactions)
        net = stoich_net(reactions)

        dCdt = np.zeros(5)
        for j, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
            dCdt[j] = (C_in[j] - C_i[j]) / tau_i + net[comp] * sum(rates)

        Q_gen = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V_i
        Q_rem = rho_cp * F_i * (T_i - T_in) + UA_i * (T_i - T_c_i)

        dTdt = (Q_gen - Q_rem) / (rho_cp * V_i)

        dydt[idx:idx + 5] = dCdt
        dydt[idx + 5] = dTdt

    return dydt


def solve_cstr_steady(params, reactions, T_range=None):
    if T_range is None:
        T_range = np.linspace(280, 500, 2000)

    V = params.get('V', 1.0)
    F = params.get('F', 1.0)
    tau = V / F
    C_f = params.get('C_f', np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    T_f = params.get('T_f', 300.0)
    rho_cp = params.get('rho_cp', 4.0e3)
    T_c = params.get('T_c', 290.0)
    UA = params.get('UA', 2000.0)

    Q_gen = np.zeros_like(T_range)
    C_ss = np.zeros((len(T_range), 5))

    for idx, T in enumerate(T_range):
        C = C_f.copy()
        for _ in range(200):
            rates = reaction_rates(C, T, reactions)
            net = stoich_net(reactions)
            C_new = np.zeros(5)
            for j, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
                C_new[j] = C_f[j] / (1 + tau * abs(net[comp]) * sum(rates) / max(C_f[j], 1e-15))
                C_new[j] = max(C_new[j], 0)
            if np.max(np.abs(C_new - C)) < 1e-8:
                C = C_new
                break
            C = C_new
        C_ss[idx] = C
        rates = reaction_rates(C, T, reactions)
        Q_gen[idx] = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V

    Q_rem = rho_cp * F * (T_range - T_f) + UA * (T_range - T_c)

    diff = Q_gen - Q_rem
    steady_states = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            T_ss = T_range[i] - diff[i] * (T_range[i + 1] - T_range[i]) / (diff[i + 1] - diff[i])
            C_ss_pt = np.interp(T_ss, T_range, C_ss[:, 0])
            steady_states.append({
                'T': T_ss,
                'C_A': C_ss_pt,
                'stable': None
            })

    for ss in steady_states:
        T_ss = ss['T']
        C = C_f.copy()
        for _ in range(300):
            rates = reaction_rates(C, T_ss, reactions)
            net = stoich_net(reactions)
            C_new = np.zeros(5)
            for j, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
                rate_sum = sum(rates) * abs(net[comp])
                C_new[j] = C_f[j] / (1 + tau * rate_sum / max(C_f[j], 1e-15))
                C_new[j] = max(C_new[j], 0)
            if np.max(np.abs(C_new - C)) < 1e-10:
                C = C_new
                break
            C = C_new

        eps = 0.01
        rates_p = reaction_rates(C, T_ss + eps, reactions)
        net_p = stoich_net(reactions)
        dQgen_dT = (sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates_p)) * V - 
                     sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, reaction_rates(C, T_ss, reactions))) * V) / eps
        dQrem_dT = rho_cp * F + UA

        ss['stable'] = dQgen_dT < dQrem_dT
        ss['C'] = C.copy()

    return T_range, Q_gen, Q_rem, steady_states


def solve_cstr_steady_fast(params, reactions):
    V = params.get('V', 1.0)
    F = params.get('F', 1.0)
    tau = V / F
    C_f = params.get('C_f', np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    T_f = params.get('T_f', 300.0)
    rho_cp = params.get('rho_cp', 4.0e3)
    T_c = params.get('T_c', 290.0)
    UA = params.get('UA', 2000.0)
    
    def compute_C_at_T(T):
        C = C_f.copy()
        for _ in range(100):
            rates = reaction_rates(C, T, reactions)
            net = stoich_net(reactions)
            C_new = np.zeros(5)
            for j, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
                C_new[j] = C_f[j] / (1 + tau * abs(net[comp]) * sum(rates) / max(C_f[j], 1e-15))
                C_new[j] = max(C_new[j], 0)
            if np.max(np.abs(C_new - C)) < 1e-8:
                return C_new
            C = C_new
        return C
    
    def Q_gen_at_T(T):
        C = compute_C_at_T(T)
        rates = reaction_rates(C, T, reactions)
        return sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V
    
    def Q_rem_at_T(T):
        return rho_cp * F * (T - T_f) + UA * (T - T_c)
    
    def residual(T):
        return Q_gen_at_T(T) - Q_rem_at_T(T)
    
    T_candidates = np.linspace(280, 550, 100)
    residuals = np.array([residual(T) for T in T_candidates])
    
    steady_states = []
    for i in range(len(residuals) - 1):
        if residuals[i] * residuals[i + 1] < 0:
            try:
                T_root = fsolve(residual, (T_candidates[i] + T_candidates[i+1]) / 2, full_output=False)[0]
                C_root = compute_C_at_T(T_root)
                steady_states.append({'T': float(T_root), 'C': C_root})
            except:
                pass
    
    steady_states = sorted(steady_states, key=lambda x: x['T'])
    
    for ss in steady_states:
        eps = 0.01
        T_ss = ss['T']
        C_ss = ss['C']
        dQgen = (Q_gen_at_T(T_ss + eps) - Q_gen_at_T(T_ss)) / eps
        dQrem = rho_cp * F + UA
        ss['stable'] = dQgen < dQrem
    
    return steady_states


def get_steady_temp_fast(params, reactions):
    V = params.get('V', 1.0)
    F = params.get('F', 1.0)
    tau = V / F
    C_f = params['C_f']
    T_f = params['T_f']
    rho_cp = params.get('rho_cp', 4.0e3)
    T_c = params['T_c']
    UA = params['UA']
    
    def Q_balance(T):
        C = C_f.copy()
        for _ in range(30):
            rates = reaction_rates(C, T, reactions)
            net = stoich_net(reactions)
            C_new = np.zeros(5)
            for j, comp in enumerate(['A', 'B', 'C', 'D', 'E']):
                C_new[j] = C_f[j] / (1 + tau * abs(net[comp]) * sum(rates) / max(C_f[j], 1e-15))
                C_new[j] = max(C_new[j], 0)
            if np.max(np.abs(C_new - C)) < 1e-6:
                break
            C = C_new
        rates = reaction_rates(C, T, reactions)
        Q_gen = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V
        Q_rem = rho_cp * F * (T - T_f) + UA * (T - T_c)
        return Q_gen - Q_rem, C
    
    T_test = [T_f, T_f + 30, T_f + 60, T_f + 100, T_f + 150, 450, 500]
    last_Q = None
    last_T = T_f
    last_C = C_f
    
    for T in T_test:
        Q, C = Q_balance(T)
        if Q < 0 and last_Q is not None and last_Q > 0:
            try:
                T_root = fsolve(lambda t: Q_balance(t)[0], (last_T + T) / 2, full_output=False)[0]
                _, C_root = Q_balance(T_root)
                return float(T_root), C_root
            except:
                pass
        if Q < 0:
            return T, C
        last_Q = Q
        last_T = T
        last_C = C
    
    return last_T, last_C


def find_critical_cooling(params, reactions, T_c_range=None):
    if T_c_range is None:
        T_c_range = np.linspace(250, 350, 500)

    n_steady = []
    for T_c in T_c_range:
        p = params.copy()
        p['T_c'] = T_c
        _, _, _, ss = solve_cstr_steady(p, reactions, T_range=np.linspace(280, 500, 500))
        n_steady.append(len(ss))

    critical_points = []
    for i in range(len(n_steady) - 1):
        if n_steady[i] != n_steady[i + 1]:
            critical_points.append({
                'T_c': (T_c_range[i] + T_c_range[i + 1]) / 2,
                'from': n_steady[i],
                'to': n_steady[i + 1]
            })

    return T_c_range, n_steady, critical_points


def run_dynamic_cstr(params, reactions, y0, t_span, disturbances=None, pid_func=None):
    t_eval = np.linspace(t_span[0], t_span[1], min(int((t_span[1] - t_span[0]) * 10), 2000))
    sol = solve_ivp(
        lambda t, y: cstr_ode(t, y, params, reactions, disturbances, pid_func),
        t_span, y0, t_eval=t_eval, method='LSODA',
        rtol=1e-8, atol=1e-10, max_step=1.0
    )
    return sol.t, sol.y


def run_dynamic_pfr(params, reactions, C_f, T_f, L=10.0, n_points=200):
    z_span = (0, L)
    z_eval = np.linspace(0, L, n_points)
    y0 = np.zeros(6)
    y0[:5] = C_f
    y0[5] = T_f

    sol = solve_ivp(
        lambda z, y: pfr_ode(z, y, params, reactions),
        z_span, y0, z_eval=z_eval, method='LSODA',
        rtol=1e-8, atol=1e-10
    )
    return sol.t, sol.y


def run_dynamic_semibatch(params, reactions, y0, t_span, disturbances=None):
    t_eval = np.linspace(t_span[0], t_span[1], min(int((t_span[1] - t_span[0]) * 10), 2000))
    sol = solve_ivp(
        lambda t, y: semibatch_ode(t, y, params, reactions, disturbances),
        t_span, y0, t_eval=t_eval, method='LSODA',
        rtol=1e-8, atol=1e-10, max_step=1.0
    )
    return sol.t, sol.y


def run_dynamic_multicstr(params, reactions, y0, t_span, n_stages, disturbances=None, pid_func=None):
    t_eval = np.linspace(t_span[0], t_span[1], min(int((t_span[1] - t_span[0]) * 10), 2000))
    sol = solve_ivp(
        lambda t, y: multicstr_ode(t, y, params, reactions, n_stages, disturbances, pid_func),
        t_span, y0, t_eval=t_eval, method='LSODA',
        rtol=1e-8, atol=1e-10, max_step=1.0
    )
    return sol.t, sol.y

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t, chi2, shapiro
from scipy.linalg import svd

R_GAS = 8.314
RHO_CP = 4000.0
DEFAULT_TF = 300.0


def arrhenius(T, k0, Ea):
    return k0 * np.exp(-Ea * 1e3 / (R_GAS * T))


def cstr_steady_isothermal_first_order(T, tau, CAf, k0, Ea):
    k = arrhenius(T, k0, Ea)
    CA = CAf / (1 + tau * k)
    CB = CAf - CA
    return CA, CB


def cstr_steady_isothermal_second_order(T, tau, CAf, k0, Ea):
    k = arrhenius(T, k0, Ea)
    sqrt_term = np.sqrt(1 + 4 * tau * k * CAf)
    CA = (sqrt_term - 1) / (2 * tau * k)
    CB = CAf - CA
    return CA, CB


def cstr_steady_single_data(T, tau, CAf, k0, Ea, model_type):
    if model_type == 'first_order':
        return cstr_steady_isothermal_first_order(T, tau, CAf, k0, Ea)
    else:
        return cstr_steady_isothermal_second_order(T, tau, CAf, k0, Ea)


def model_predict(params, T_data, tau_data, CAf, Tf, model_type, include_dH=False):
    if include_dH:
        k0, Ea, deltaH = params
    else:
        k0, Ea = params
        deltaH = 0.0

    n = len(T_data)
    CA_pred = np.zeros(n)
    CB_pred = np.zeros(n)

    for i in range(n):
        T = T_data[i]
        tau = tau_data[i]
        CA_pred[i], CB_pred[i] = cstr_steady_single_data(
            T, tau, CAf, k0, Ea, model_type
        )

    return CA_pred, CB_pred


def residuals(params, T_data, tau_data, CA_data, CB_data, CAf, Tf, model_type, include_dH=False):
    if include_dH:
        k0, Ea, deltaH = params
    else:
        k0, Ea = params

    n = len(T_data)
    CA_pred = np.zeros(n)
    CB_pred = np.zeros(n)

    for i in range(n):
        T = T_data[i]
        tau = tau_data[i]
        CA_pred[i], CB_pred[i] = cstr_steady_single_data(
            T, tau, CAf, k0, Ea, model_type
        )

    res_CA = CA_data - CA_pred
    res_CB = CB_data - CB_pred

    if include_dH:
        deltaH_SI = deltaH * 1e3
        conversion = 1.0 - CA_pred / CAf if CAf > 0 else np.zeros_like(CA_pred)
        T_pred_energy = Tf + (-deltaH_SI) * CAf * conversion / RHO_CP
        T_scale = max(np.std(T_data) if np.std(T_data) > 0 else 1.0, 1.0)
        res_T = (T_data - T_pred_energy) / T_scale
        res_conc_scale = max(CAf * 0.01, 1.0)
        res_CA_scaled = res_CA / res_conc_scale
        res_CB_scaled = res_CB / res_conc_scale
        return np.concatenate([res_CA_scaled, res_CB_scaled, res_T])
    else:
        return np.concatenate([res_CA, res_CB])


def estimate_initial_guess(T_data, tau_data, CA_data, CB_data, CAf, model_type, include_dH=False, Tf=DEFAULT_TF):
    conversions = 1 - CA_data / CAf
    valid_mask = (conversions > 0.05) & (conversions < 0.95)

    if np.sum(valid_mask) < 2:
        valid_mask = np.ones_like(T_data, dtype=bool)

    T_valid = T_data[valid_mask]
    tau_valid = tau_data[valid_mask]
    conversions_valid = conversions[valid_mask]

    if model_type == 'first_order':
        k_values = conversions_valid / (tau_valid * (1 - conversions_valid))
    else:
        k_values = conversions_valid / (tau_valid * CAf * (1 - conversions_valid) ** 2)

    ln_k = np.log(np.maximum(k_values, 1e-10))
    inv_T = 1000.0 / (R_GAS * T_valid)

    slope, intercept = np.polyfit(inv_T, ln_k, 1)

    Ea_guess = max(30.0, min(200.0, -slope))
    k0_guess = np.exp(intercept)
    k0_guess = max(1e5, min(1e15, k0_guess))

    if include_dH:
        dH_guesses = []
        for i in range(len(T_data)):
            X = 1 - CA_data[i] / CAf
            if X > 0.05:
                dH_est = -RHO_CP * (T_data[i] - Tf) / max(CAf * X * 1e3, 1e-10)
                dH_guesses.append(dH_est)

        if len(dH_guesses) > 0:
            dH_mean = np.median(dH_guesses)
            dH_guess = max(-300.0, min(100.0, dH_mean))
        else:
            dH_guess = -80.0

        guess = [k0_guess, Ea_guess, dH_guess]
    else:
        guess = [k0_guess, Ea_guess]

    return np.array(guess)


def fit_parameters(T_data, tau_data, CA_data, CB_data, CAf, model_type='first_order',
                   include_dH=False, Tf=DEFAULT_TF, bounds=None, x0=None, auto_init=True,
                   max_iter=1000, ftol=1e-10, xtol=1e-10):
    n_data = len(T_data)
    if n_data < 6:
        return {'success': False, 'message': '至少需要6个数据点才能进行拟合'}

    if bounds is None:
        if include_dH:
            bounds = ([1e5, 30.0, -300.0], [1e15, 200.0, 100.0])
        else:
            bounds = ([1e5, 30.0], [1e15, 200.0])

    if x0 is None or auto_init:
        x0 = estimate_initial_guess(T_data, tau_data, CA_data, CB_data, CAf, model_type, include_dH, Tf)

    try:
        result = least_squares(
            residuals, x0,
            args=(T_data, tau_data, CA_data, CB_data, CAf, Tf, model_type, include_dH),
            bounds=bounds,
            method='trf',
            max_nfev=max_iter,
            ftol=ftol,
            xtol=xtol,
            jac='2-point'
        )
    except Exception as e:
        return {'success': False, 'message': f'拟合失败: {str(e)}'}

    if not result.success:
        if result.status == 0:
            message = '拟合未收敛：达到最大迭代次数'
        elif result.status == 2:
            message = '拟合未收敛：雅可比矩阵奇异，参数可能不可辨识'
        else:
            message = f'拟合未收敛：{result.message}'
        return {'success': False, 'message': message}

    params_opt = result.x
    n_params = len(params_opt)
    n = len(T_data)

    CA_pred, CB_pred = model_predict(params_opt, T_data, tau_data, CAf, Tf, model_type, include_dH)

    res_CA_raw = CA_data - CA_pred
    res_CB_raw = CB_data - CB_pred
    residuals_conc = np.concatenate([res_CA_raw, res_CB_raw])
    n_obs = 2 * n  # 浓度残差始终是2n个

    RSS = np.sum(residuals_conc ** 2)
    dof = n_obs - n_params
    if dof <= 0:
        dof = 1
    MSE = RSS / dof
    RMSE = np.sqrt(MSE)

    y_mean = np.mean(np.concatenate([CA_data, CB_data]))
    TSS = np.sum((np.concatenate([CA_data, CB_data]) - y_mean) ** 2)
    R2 = 1 - RSS / TSS if TSS > 0 else np.nan

    J = result.jac
    try:
        U, S, Vt = svd(J, full_matrices=False)
        S2 = S ** 2
        cov_matrix = np.dot(Vt.T, np.dot(np.diag(1.0 / np.maximum(S2, 1e-15)), Vt)) * MSE
    except:
        JTJ = J.T @ J
        cov_matrix = np.linalg.pinv(JTJ) * MSE

    param_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 0))

    alpha = 0.05
    t_crit = t.ppf(1 - alpha / 2, dof)
    confidence_intervals = []
    for i in range(n_params):
        ci = param_errors[i] * t_crit
        confidence_intervals.append(ci)

    corr_matrix = np.zeros_like(cov_matrix)
    for i in range(n_params):
        for j in range(n_params):
            denom = np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
            if denom > 1e-15:
                corr_matrix[i, j] = cov_matrix[i, j] / denom
            else:
                corr_matrix[i, j] = 1.0 if i == j else 0.0

    residuals_all = residuals_conc

    concentration_scale = max(CAf, 1.0)
    residual_scale = RMSE / concentration_scale if concentration_scale > 0 else RMSE
    residuals_meaningful = residual_scale > 1e-8 and RMSE > 1e-8

    shapiro_p = np.nan
    if residuals_meaningful and len(residuals_all) >= 3:
        try:
            _, shapiro_p = shapiro(residuals_all)
        except:
            shapiro_p = np.nan

    dw_stat = np.nan
    dw_residuals = CA_data - CA_pred
    if residuals_meaningful and len(dw_residuals) > 2:
        res_sq_sum = np.sum(dw_residuals ** 2)
        if res_sq_sum > 1e-30:
            diff = np.diff(dw_residuals)
            dw_stat = np.sum(diff ** 2) / res_sq_sum

    param_names = ['k0', 'Ea']
    if include_dH:
        param_names.append('deltaH')

    param_units = ['s⁻¹' if model_type == 'first_order' else 'm³/(mol·s)', 'kJ/mol']
    if include_dH:
        param_units.append('kJ/mol')

    return {
        'success': True,
        'params': params_opt,
        'param_names': param_names,
        'param_units': param_units,
        'param_errors': param_errors,
        'confidence_intervals': confidence_intervals,
        't_critical': t_crit,
        'cov_matrix': cov_matrix,
        'corr_matrix': corr_matrix,
        'RSS': RSS,
        'R2': R2,
        'RMSE': RMSE,
        'MSE': MSE,
        'dof': dof,
        'CA_pred': CA_pred,
        'CB_pred': CB_pred,
        'residuals': residuals_all,
        'residuals_CA': CA_data - CA_pred,
        'residuals_CB': CB_data - CB_pred,
        'shapiro_p': shapiro_p,
        'dw_stat': dw_stat,
        'model_type': model_type,
        'include_dH': include_dH,
        'CAf': CAf,
        'Tf': Tf,
        'T_data': T_data,
        'tau_data': tau_data,
        'CA_data': CA_data,
        'CB_data': CB_data,
        'residuals_meaningful': residuals_meaningful,
        'residual_scale': residual_scale,
        'message': '拟合成功'
    }


def generate_validation_data(model_type='first_order', noise_level=0.0, n_points=12, CAf=100.0,
                             k0_true=None, Ea_true=None, deltaH_true=None, Tf=DEFAULT_TF, seed=42,
                             include_dH=False):
    np.random.seed(seed)

    if model_type == 'first_order':
        k0_true = k0_true if k0_true is not None else 1e10
        Ea_true = Ea_true if Ea_true is not None else 80.0
    else:
        k0_true = k0_true if k0_true is not None else 5e8
        Ea_true = Ea_true if Ea_true is not None else 65.0

    if include_dH and deltaH_true is None:
        deltaH_true = -30.0

    tau_range = np.logspace(-1, 2, n_points)
    indices = np.random.permutation(n_points)
    tau_data = tau_range[indices]

    if include_dH:
        deltaH_SI = deltaH_true * 1e3
        max_X_for_temp = min(0.8, 180.0 * RHO_CP / max(abs(deltaH_SI) * CAf, 1e-6))
        X_targets = np.linspace(0.05, max_X_for_temp, n_points)
        X_targets = np.clip(X_targets, 0.01, 0.9)
        X_targets = X_targets[indices]

        T_data = np.zeros(n_points)
        CA_data = np.zeros(n_points)
        CB_data = np.zeros(n_points)

        for i in range(n_points):
            X_target = X_targets[i]
            T_target = Tf + (-deltaH_SI) * CAf * X_target / RHO_CP
            T_target = max(230.0, min(700.0, T_target))

            k_at_T = arrhenius(T_target, k0_true, Ea_true)
            if model_type == 'first_order':
                tau_required = X_target / (k_at_T * max(1 - X_target, 1e-4))
            else:
                tau_required = X_target / (k_at_T * CAf * max((1 - X_target) ** 2, 1e-4))

            tau_use = 0.7 * tau_data[i] + 0.3 * max(tau_required, 0.01)
            tau_use = max(0.05, min(tau_use, 200.0))

            k_use = arrhenius(T_target, k0_true, Ea_true)
            if model_type == 'first_order':
                X_model = tau_use * k_use / (1 + tau_use * k_use)
            else:
                sqrt_val = np.sqrt(1 + 4 * tau_use * k_use * CAf)
                X_model = (sqrt_val - 1) / (2 * tau_use * k_use * CAf) if (tau_use * k_use * CAf) > 1e-15 else 0.0
            X_model = np.clip(X_model, 0.0, 0.95)

            T_actual = Tf + (-deltaH_SI) * CAf * X_model / RHO_CP
            T_actual = max(230.0, min(700.0, T_actual))

            k_actual = arrhenius(T_actual, k0_true, Ea_true)
            if model_type == 'first_order':
                X_final = tau_use * k_actual / (1 + tau_use * k_actual)
            else:
                sqrt_val = np.sqrt(1 + 4 * tau_use * k_actual * CAf)
                X_final = (sqrt_val - 1) / (2 * tau_use * k_actual * CAf) if (tau_use * k_actual * CAf) > 1e-15 else 0.0
            X_final = np.clip(X_final, 0.0, 0.95)

            T_final = Tf + (-deltaH_SI) * CAf * X_final / RHO_CP
            T_final = max(230.0, min(700.0, T_final))

            T_data[i] = T_final
            tau_data[i] = tau_use
            CA_data[i] = max(0, CAf * (1 - X_final))
            CB_data[i] = max(0, CAf * X_final)
    else:
        T_range = np.linspace(320, 520, n_points)
        T_data = T_range[indices]
        CA_data = np.zeros(n_points)
        CB_data = np.zeros(n_points)

        for i in range(n_points):
            T = T_data[i]
            tau = tau_data[i]
            if model_type == 'first_order':
                CA, CB = cstr_steady_isothermal_first_order(T, tau, CAf, k0_true, Ea_true)
            else:
                CA, CB = cstr_steady_isothermal_second_order(T, tau, CAf, k0_true, Ea_true)
            CA_data[i] = max(0, CA)
            CB_data[i] = max(0, CB)

    if noise_level > 0:
        for i in range(n_points):
            noise_CA = np.random.normal(0, noise_level * CAf)
            noise_CB = np.random.normal(0, noise_level * CAf)
            CA_data[i] = max(0, CA_data[i] + noise_CA)
            CB_data[i] = max(0, CB_data[i] + noise_CB)
        if include_dH:
            for i in range(n_points):
                noise_T = np.random.normal(0, noise_level * 20.0)
                T_data[i] = max(220.0, min(780.0, T_data[i] + noise_T))

    result = {
        'T': T_data,
        'tau': tau_data,
        'CA': CA_data,
        'CB': CB_data,
        'CAf': CAf,
        'k0_true': k0_true,
        'Ea_true': Ea_true,
        'model_type': model_type,
        'noise_level': noise_level,
        'Tf': Tf,
        'include_dH': include_dH
    }
    if include_dH:
        result['deltaH_true'] = deltaH_true

    return result


def compute_confidence_ellipse(cov_matrix, params, idx1, idx2, confidence=0.95, n_points=100):
    chi2_crit = chi2.ppf(confidence, 2)

    sub_cov = cov_matrix[[[idx1], [idx2]], [idx1, idx2]]

    vals, vecs = np.linalg.eigh(sub_cov)
    vals = np.maximum(vals, 0)

    theta = np.linspace(0, 2 * np.pi, n_points)
    ellipse = np.zeros((n_points, 2))
    for i in range(n_points):
        ellipse[i, 0] = np.sqrt(chi2_crit * vals[0]) * np.cos(theta[i])
        ellipse[i, 1] = np.sqrt(chi2_crit * vals[1]) * np.sin(theta[i])

    ellipse_rot = np.dot(ellipse, vecs.T)

    ellipse_rot[:, 0] += params[idx1]
    ellipse_rot[:, 1] += params[idx2]

    return ellipse_rot


def compute_sensitivity_curves(fit_result, n_points=50):
    params = fit_result['params']
    param_names = fit_result['param_names']
    n_params = len(params)
    ci = fit_result['confidence_intervals']
    model_type = fit_result['model_type']
    include_dH = fit_result['include_dH']
    CAf = fit_result['CAf']
    Tf = fit_result.get('Tf', DEFAULT_TF)

    T_data = fit_result['T_data']
    tau_data = fit_result['tau_data']

    ref_idx = np.argmax(fit_result['CB_pred'])

    sensitivity_data = {}

    for i in range(n_params):
        p_min = params[i] - 2 * ci[i]
        p_max = params[i] + 2 * ci[i]

        if np.isinf(p_min) or np.isinf(p_max):
            p_min = params[i] * 0.5 if params[i] != 0 else -100.0
            p_max = params[i] * 1.5 if params[i] != 0 else 100.0

        p_min = min(p_min, params[i] * 0.999 if abs(params[i]) > 1e-10 else params[i] - 1)
        p_max = max(p_max, params[i] * 1.001 if abs(params[i]) > 1e-10 else params[i] + 1)

        param_values = np.linspace(p_min, p_max, n_points)
        model_outputs = np.zeros(n_points)

        for j, p_val in enumerate(param_values):
            test_params = params.copy()
            test_params[i] = p_val

            CA_pred, CB_pred = model_predict(
                test_params,
                np.array([T_data[ref_idx]]),
                np.array([tau_data[ref_idx]]),
                CAf, Tf, model_type, include_dH
            )
            model_outputs[j] = CB_pred[0]

        sensitivity_data[param_names[i]] = {
            'param_values': param_values,
            'model_outputs': model_outputs,
            'param_range': (p_min, p_max)
        }

    return sensitivity_data, ref_idx


def validate_experimental_data(T, tau, CA, CB):
    errors = []
    n = len(T)

    for i in range(n):
        if T[i] < 200 or T[i] > 800:
            errors.append(f'行{i+1}: 温度{T[i]:.1f}K超出范围[200, 800]K')
        if tau[i] <= 0:
            errors.append(f'行{i+1}: 停留时间必须为正数，当前为{tau[i]:.3f}')
        if CA[i] < 0:
            errors.append(f'行{i+1}: A浓度不能为负，当前为{CA[i]:.2f}')
        if CB[i] < 0:
            errors.append(f'行{i+1}: B浓度不能为负，当前为{CB[i]:.2f}')

    return errors


def get_cell_error_mask(T, tau, CA, CB):
    n = len(T)
    error_mask = {
        'T': np.zeros(n, dtype=bool),
        'tau': np.zeros(n, dtype=bool),
        'CA': np.zeros(n, dtype=bool),
        'CB': np.zeros(n, dtype=bool)
    }

    for i in range(n):
        if T[i] < 200 or T[i] > 800:
            error_mask['T'][i] = True
        if tau[i] <= 0:
            error_mask['tau'][i] = True
        if CA[i] < 0:
            error_mask['CA'][i] = True
        if CB[i] < 0:
            error_mask['CB'][i] = True

    return error_mask

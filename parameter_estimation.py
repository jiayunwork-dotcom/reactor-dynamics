import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t, chi2, shapiro
from scipy.linalg import svd

R_GAS = 8.314

def arrhenius(T, k0, Ea):
    return k0 * np.exp(-Ea * 1e3 / (R_GAS * T))

def cstr_steady_first_order(T, tau, CAf, k0, Ea, deltaH=None):
    k = arrhenius(T, k0, Ea)
    CA = CAf / (1 + tau * k)
    CB = CAf - CA
    if deltaH is not None:
        pass
    return CA, CB

def cstr_steady_second_order(T, tau, CAf, k0, Ea, deltaH=None):
    k = arrhenius(T, k0, Ea)
    sqrt_term = np.sqrt(1 + 4 * tau * k * CAf)
    CA = (sqrt_term - 1) / (2 * tau * k)
    CB = CAf - CA
    return CA, CB

def model_predict(params, T_data, tau_data, CAf, model_type, include_dH=False):
    if include_dH:
        if model_type == 'first_order':
            k0, Ea, deltaH = params
        else:
            k0, Ea, deltaH = params
    else:
        k0, Ea = params
        deltaH = None
    
    n = len(T_data)
    CA_pred = np.zeros(n)
    CB_pred = np.zeros(n)
    
    for i in range(n):
        T = T_data[i]
        tau = tau_data[i]
        if model_type == 'first_order':
            CA_pred[i], CB_pred[i] = cstr_steady_first_order(T, tau, CAf, k0, Ea, deltaH)
        else:
            CA_pred[i], CB_pred[i] = cstr_steady_second_order(T, tau, CAf, k0, Ea, deltaH)
    
    return CA_pred, CB_pred

def residuals(params, T_data, tau_data, CA_data, CB_data, CAf, model_type, include_dH=False):
    CA_pred, CB_pred = model_predict(params, T_data, tau_data, CAf, model_type, include_dH)
    res_CA = CA_data - CA_pred
    res_CB = CB_data - CB_pred
    return np.concatenate([res_CA, res_CB])

def estimate_initial_guess(T_data, tau_data, CA_data, CB_data, CAf, model_type, include_dH=False):
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
        guess = [k0_guess, Ea_guess, -50.0]
    else:
        guess = [k0_guess, Ea_guess]
    
    return np.array(guess)

def fit_parameters(T_data, tau_data, CA_data, CB_data, CAf, model_type='first_order',
                   include_dH=False, bounds=None, x0=None, auto_init=True,
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
        x0 = estimate_initial_guess(T_data, tau_data, CA_data, CB_data, CAf, model_type, include_dH)
    
    try:
        result = least_squares(
            residuals, x0,
            args=(T_data, tau_data, CA_data, CB_data, CAf, model_type, include_dH),
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
    res = result.fun
    n_params = len(params_opt)
    n_obs = len(res)
    dof = n_obs - n_params
    
    RSS = np.sum(res ** 2)
    MSE = RSS / dof if dof > 0 else np.inf
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
        cov_matrix = np.linalg.pinv(J.T @ J) * MSE
    
    param_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 0))
    
    alpha = 0.05
    t_crit = t.ppf(1 - alpha / 2, dof) if dof > 0 else np.inf
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
    
    CA_pred, CB_pred = model_predict(params_opt, T_data, tau_data, CAf, model_type, include_dH)
    residuals_all = np.concatenate([CA_data - CA_pred, CB_data - CB_pred])
    
    if len(residuals_all) >= 3:
        try:
            _, shapiro_p = shapiro(residuals_all)
        except:
            shapiro_p = np.nan
    else:
        shapiro_p = np.nan
    
    dw_residuals = CA_data - CA_pred
    if len(dw_residuals) > 2:
        diff = np.diff(dw_residuals)
        dw_stat = np.sum(diff ** 2) / np.sum(dw_residuals ** 2)
    else:
        dw_stat = np.nan
    
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
        'T_data': T_data,
        'tau_data': tau_data,
        'CA_data': CA_data,
        'CB_data': CB_data,
        'message': '拟合成功'
    }

def generate_validation_data(model_type='first_order', noise_level=0.0, n_points=12, CAf=100.0,
                             k0_true=None, Ea_true=None, seed=42):
    np.random.seed(seed)
    
    if model_type == 'first_order':
        k0_true = k0_true if k0_true is not None else 1e10
        Ea_true = Ea_true if Ea_true is not None else 80.0
    else:
        k0_true = k0_true if k0_true is not None else 5e8
        Ea_true = Ea_true if Ea_true is not None else 65.0
    
    T_range = np.linspace(300, 500, n_points)
    tau_range = np.logspace(-1, 2, n_points)
    
    indices = np.random.permutation(n_points)
    T_data = T_range[indices]
    tau_data = tau_range[indices]
    
    CA_data = np.zeros(n_points)
    CB_data = np.zeros(n_points)
    
    for i in range(n_points):
        T = T_data[i]
        tau = tau_data[i]
        if model_type == 'first_order':
            CA, CB = cstr_steady_first_order(T, tau, CAf, k0_true, Ea_true)
        else:
            CA, CB = cstr_steady_second_order(T, tau, CAf, k0_true, Ea_true)
        
        if noise_level > 0:
            noise_CA = np.random.normal(0, noise_level * CAf)
            noise_CB = np.random.normal(0, noise_level * CAf)
            CA += noise_CA
            CB += noise_CB
        
        CA_data[i] = max(0, CA)
        CB_data[i] = max(0, CB)
    
    return {
        'T': T_data,
        'tau': tau_data,
        'CA': CA_data,
        'CB': CB_data,
        'CAf': CAf,
        'k0_true': k0_true,
        'Ea_true': Ea_true,
        'model_type': model_type,
        'noise_level': noise_level
    }

def compute_confidence_ellipse(cov_matrix, params, idx1, idx2, confidence=0.95, n_points=100):
    chi2_crit = chi2.ppf(confidence, 2)
    
    sub_cov = cov_matrix[[[idx1], [idx2]], [idx1, idx2]]
    
    vals, vecs = np.linalg.eigh(sub_cov)
    
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
    
    T_data = fit_result['T_data']
    tau_data = fit_result['tau_data']
    
    ref_idx = np.argmax(fit_result['CB_pred'])
    
    sensitivity_data = {}
    
    for i in range(n_params):
        p_min = params[i] - 2 * ci[i]
        p_max = params[i] + 2 * ci[i]
        
        param_values = np.linspace(p_min, p_max, n_points)
        model_outputs = np.zeros(n_points)
        
        for j, p_val in enumerate(param_values):
            test_params = params.copy()
            test_params[i] = p_val
            
            CA_pred, CB_pred = model_predict(
                test_params, 
                np.array([T_data[ref_idx]]), 
                np.array([tau_data[ref_idx]]), 
                CAf, model_type, include_dH
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

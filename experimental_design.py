import numpy as np

R_GAS = 8.314


def arrhenius(T, k0, Ea):
    return k0 * np.exp(-Ea * 1e3 / (R_GAS * T))


def model_output(T, tau, CAf, k0, Ea, model_type):
    k = arrhenius(T, k0, Ea)
    if model_type == 'first_order':
        CA = CAf / (1 + tau * k)
    else:
        sqrt_term = np.sqrt(1 + 4 * tau * k * CAf)
        CA = (sqrt_term - 1) / (2 * tau * k) if (tau * k) > 1e-15 else CAf
    return CA


def numerical_derivatives(T, tau, CAf, k0, Ea, model_type):
    eps_k0 = 0.001 * abs(k0) if abs(k0) > 1e-15 else 1e-10
    eps_Ea = 0.001 * abs(Ea) if abs(Ea) > 1e-15 else 1e-10

    CA_base = model_output(T, tau, CAf, k0, Ea, model_type)

    CA_k0_p = model_output(T, tau, CAf, k0 + eps_k0, Ea, model_type)
    CA_k0_m = model_output(T, tau, CAf, k0 - eps_k0, Ea, model_type)
    dCA_dk0 = (CA_k0_p - CA_k0_m) / (2 * eps_k0)

    CA_Ea_p = model_output(T, tau, CAf, k0, Ea + eps_Ea, model_type)
    CA_Ea_m = model_output(T, tau, CAf, k0, Ea - eps_Ea, model_type)
    dCA_dEa = (CA_Ea_p - CA_Ea_m) / (2 * eps_Ea)

    return np.array([dCA_dk0, dCA_dEa])


def fisher_information_matrix(T_array, tau_array, CAf, k0, Ea, model_type, sigma_sq=1.0):
    n_points = len(T_array)
    F = np.zeros((2, 2))

    for i in range(n_points):
        grad = numerical_derivatives(T_array[i], tau_array[i], CAf, k0, Ea, model_type)
        F += np.outer(grad, grad) / sigma_sq

    return F


def compute_precision_metrics(F, k0, Ea):
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(F)

    se_k0 = np.sqrt(max(cov[0, 0], 0))
    se_Ea = np.sqrt(max(cov[1, 1], 0))

    rse_k0 = (se_k0 / abs(k0) * 100) if abs(k0) > 1e-15 else np.inf
    rse_Ea = (se_Ea / abs(Ea) * 100) if abs(Ea) > 1e-15 else np.inf

    denom = np.sqrt(cov[0, 0] * cov[1, 1])
    corr = cov[0, 1] / denom if denom > 1e-15 else 0.0

    return {
        'cov_matrix': cov,
        'se_k0': se_k0,
        'se_Ea': se_Ea,
        'rse_k0': rse_k0,
        'rse_Ea': rse_Ea,
        'correlation': corr
    }


def generate_candidates(T_min, T_max, tau_min, tau_max, n_candidates=500, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n_T = int(np.ceil(np.sqrt(n_candidates)))
    n_tau = int(np.ceil(n_candidates / n_T))
    
    T_edges = np.linspace(T_min, T_max, n_T + 1)
    tau_edges = np.linspace(tau_min, tau_max, n_tau + 1)
    
    T_centers = (T_edges[:-1] + T_edges[1:]) / 2
    tau_centers = (tau_edges[:-1] + tau_edges[1:]) / 2
    
    T_grid, tau_grid = np.meshgrid(T_centers, tau_centers)
    T_base = T_grid.ravel()
    tau_base = tau_grid.ravel()
    
    jitter_T = np.random.uniform(-0.4 * (T_max - T_min) / n_T, 
                                  0.4 * (T_max - T_min) / n_T, 
                                  len(T_base))
    jitter_tau = np.random.uniform(-0.4 * (tau_max - tau_min) / n_tau, 
                                   0.4 * (tau_max - tau_min) / n_tau, 
                                   len(tau_base))
    
    T_candidates = np.clip(T_base + jitter_T, T_min, T_max)
    tau_candidates = np.clip(tau_base + jitter_tau, tau_min, tau_max)
    
    if len(T_candidates) > n_candidates:
        perm = np.random.permutation(len(T_candidates))
        T_candidates = T_candidates[perm[:n_candidates]]
        tau_candidates = tau_candidates[perm[:n_candidates]]
    
    return np.column_stack([T_candidates, tau_candidates])


def _points_are_equal(p1, p2, tol_T=0.1, tol_tau=0.01):
    return abs(p1[0] - p2[0]) < tol_T and abs(p1[1] - p2[1]) < tol_tau


def _points_distance(p1, p2, T_range, tau_range):
    dT_norm = abs(p1[0] - p2[0]) / T_range if T_range > 0 else 0
    dtau_norm = abs(p1[1] - p2[1]) / tau_range if tau_range > 0 else 0
    return np.sqrt(dT_norm**2 + dtau_norm**2)


def _is_point_unique_and_spaced(new_point, existing_points, T_range, tau_range, 
                                exclude_idx=None, min_distance=0.05, tol_T=0.1, tol_tau=0.01):
    for i, p in enumerate(existing_points):
        if exclude_idx is not None and i == exclude_idx:
            continue
        if _points_are_equal(new_point, p, tol_T, tol_tau):
            return False
        if min_distance > 0:
            dist = _points_distance(new_point, p, T_range, tau_range)
            if dist < min_distance:
                return False
    return True


def doptimal_design(T_min, T_max, tau_min, tau_max, CAf, k0, Ea, model_type,
                    n_points=8, n_candidates=500, n_iterations=None, seed=None):
    if n_iterations is None:
        n_iterations = n_points * 500

    T_range = T_max - T_min
    tau_range = tau_max - tau_min

    min_distance = 1.0 / (np.sqrt(n_points) * 2.5)

    candidates = generate_candidates(T_min, T_max, tau_min, tau_max, n_candidates, seed)

    if seed is not None:
        np.random.seed(seed + 1 if seed is not None else None)

    initial_indices = []
    available_indices = list(range(n_candidates))
    while len(initial_indices) < n_points and available_indices:
        idx = np.random.choice(available_indices)
        candidate = candidates[idx]
        current_points = [candidates[i] for i in initial_indices]
        if _is_point_unique_and_spaced(candidate, current_points, T_range, tau_range,
                                       min_distance=min_distance):
            initial_indices.append(idx)
        available_indices.remove(idx)

    if len(initial_indices) < n_points:
        initial_indices = []
        available_indices = list(range(n_candidates))
        while len(initial_indices) < n_points and available_indices:
            idx = np.random.choice(available_indices)
            candidate = candidates[idx]
            current_points = [candidates[i] for i in initial_indices]
            if _is_point_unique_and_spaced(candidate, current_points, T_range, tau_range,
                                           min_distance=min_distance * 0.5):
                initial_indices.append(idx)
            available_indices.remove(idx)

    if len(initial_indices) < n_points:
        initial_indices = list(range(n_points))

    design = candidates[initial_indices].copy()

    T_design = design[:, 0]
    tau_design = design[:, 1]
    F = fisher_information_matrix(T_design, tau_design, CAf, k0, Ea, model_type)
    current_det = np.linalg.det(F)

    if np.isnan(current_det) or np.isinf(current_det):
        current_det = 0.0

    for it in range(n_iterations):
        idx_replace = np.random.randint(0, n_points)
        
        valid_candidates = []
        for idx_c in range(n_candidates):
            if _is_point_unique_and_spaced(candidates[idx_c], design, T_range, tau_range,
                                           exclude_idx=idx_replace, min_distance=min_distance):
                valid_candidates.append(idx_c)
        
        if not valid_candidates:
            for idx_c in range(n_candidates):
                if _is_point_unique_and_spaced(candidates[idx_c], design, T_range, tau_range,
                                               exclude_idx=idx_replace, min_distance=min_distance * 0.5):
                    valid_candidates.append(idx_c)
        
        if not valid_candidates:
            continue
        
        idx_candidate = np.random.choice(valid_candidates)

        new_design = design.copy()
        new_design[idx_replace] = candidates[idx_candidate]

        T_new = new_design[:, 0]
        tau_new = new_design[:, 1]
        F_new = fisher_information_matrix(T_new, tau_new, CAf, k0, Ea, model_type)
        new_det = np.linalg.det(F_new)

        if np.isnan(new_det) or np.isinf(new_det):
            continue

        if new_det > current_det:
            design = new_design
            current_det = new_det
            F = F_new

    sort_idx = np.argsort(design[:, 0])
    design = design[sort_idx]

    return design, F, current_det


def uniform_grid_design(T_min, T_max, tau_min, tau_max, n_points, CAf, k0, Ea, model_type):
    n_side = int(np.ceil(np.sqrt(n_points)))

    T_grid_vals = np.linspace(T_min, T_max, n_side)
    tau_grid_vals = np.linspace(tau_min, tau_max, n_side)

    T_mesh, tau_mesh = np.meshgrid(T_grid_vals, tau_grid_vals)
    grid_points = np.column_stack([T_mesh.ravel(), tau_mesh.ravel()])

    if len(grid_points) > n_points:
        selected_indices = np.linspace(0, len(grid_points) - 1, n_points).astype(int)
        grid_points = grid_points[selected_indices]

    sort_idx = np.argsort(grid_points[:, 0])
    grid_points = grid_points[sort_idx]

    T_grid = grid_points[:, 0]
    tau_grid = grid_points[:, 1]

    F_grid = fisher_information_matrix(T_grid, tau_grid, CAf, k0, Ea, model_type)
    det_grid = np.linalg.det(F_grid)

    if np.isnan(det_grid) or np.isinf(det_grid):
        det_grid = max(np.linalg.det(F_grid + 1e-15 * np.eye(2)), 1e-30)

    return grid_points, F_grid, det_grid


def compute_d_efficiency(det_opt, det_grid, n_points):
    if det_grid <= 0 or det_opt <= 0:
        return 1.0
    return (det_opt / det_grid) ** (1.0 / n_points)


def run_design_optimization(config):
    model_type = 'first_order' if config['model_type'] == '一级不可逆' else 'second_order'

    design_opt, F_opt, det_opt = doptimal_design(
        T_min=config['T_min'],
        T_max=config['T_max'],
        tau_min=config['tau_min'],
        tau_max=config['tau_max'],
        CAf=config['CAf'],
        k0=config['k0'],
        Ea=config['Ea'],
        model_type=model_type,
        n_points=config['n_points'],
        n_candidates=500,
        seed=42
    )

    design_grid, F_grid, det_grid = uniform_grid_design(
        T_min=config['T_min'],
        T_max=config['T_max'],
        tau_min=config['tau_min'],
        tau_max=config['tau_max'],
        n_points=config['n_points'],
        CAf=config['CAf'],
        k0=config['k0'],
        Ea=config['Ea'],
        model_type=model_type
    )

    precision_opt = compute_precision_metrics(F_opt, config['k0'], config['Ea'])
    precision_grid = compute_precision_metrics(F_grid, config['k0'], config['Ea'])

    d_efficiency = compute_d_efficiency(det_opt, det_grid, config['n_points'])

    return {
        'design_opt': design_opt,
        'F_opt': F_opt,
        'det_opt': det_opt,
        'precision_opt': precision_opt,
        'design_grid': design_grid,
        'F_grid': F_grid,
        'det_grid': det_grid,
        'precision_grid': precision_grid,
        'd_efficiency': d_efficiency
    }

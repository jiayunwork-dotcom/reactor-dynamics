import numpy as np
from parameter_estimation import (
    fit_parameters,
    generate_validation_data,
    compute_confidence_ellipse,
    compute_sensitivity_curves,
    validate_experimental_data,
    get_cell_error_mask
)

def test_validation_dataset_1():
    print("=" * 60)
    print("测试数据集一: 理想一级反应 (无噪声)")
    print("=" * 60)
    
    data = generate_validation_data(
        model_type='first_order',
        noise_level=0.0,
        n_points=12,
        CAf=100.0,
        k0_true=1e10,
        Ea_true=80.0,
        seed=42
    )
    
    result = fit_parameters(
        data['T'], data['tau'], data['CA'], data['CB'], data['CAf'],
        model_type='first_order',
        include_dH=False
    )
    
    print(f"拟合成功: {result['success']}")
    print(f"消息: {result['message']}")
    
    if result['success']:
        k0_fit = result['params'][0]
        Ea_fit = result['params'][1]
        
        k0_error = abs(k0_fit - data['k0_true']) / data['k0_true'] * 100
        Ea_error = abs(Ea_fit - data['Ea_true']) / data['Ea_true'] * 100
        
        print(f"\n真实 k0: {data['k0_true']:.4e} s⁻¹")
        print(f"拟合 k0: {k0_fit:.4e} s⁻¹")
        print(f"k0 相对误差: {k0_error:.6f}%")
        
        print(f"\n真实 Ea: {data['Ea_true']:.4f} kJ/mol")
        print(f"拟合 Ea: {Ea_fit:.4f} kJ/mol")
        print(f"Ea 相对误差: {Ea_error:.6f}%")
        
        print(f"\nR²: {result['R2']:.8f}")
        print(f"RMSE: {result['RMSE']:.6f}")
        print(f"RSS: {result['RSS']:.6e}")
        
        print(f"\n95% 置信区间:")
        print(f"  k0: ±{result['confidence_intervals'][0]:.4e}")
        print(f"  Ea: ±{result['confidence_intervals'][1]:.4f} kJ/mol")
        
        print(f"\n参数相关系数矩阵:")
        print(result['corr_matrix'])
        
        print(f"\n残差正态性检验 p值: {result['shapiro_p']:.4f}")
        print(f"Durbin-Watson 统计量: {result['dw_stat']:.4f}")
        
        if k0_error < 0.1 and Ea_error < 0.1:
            print("\n✅ 测试通过: 参数误差小于0.1%")
            return True
        else:
            print(f"\n❌ 测试失败: 参数误差超过0.1%")
            return False
    else:
        print("\n❌ 测试失败: 拟合未成功")
        return False

def test_validation_dataset_2():
    print("\n" + "=" * 60)
    print("测试数据集二: 含噪声二级反应 (5%噪声)")
    print("=" * 60)
    
    data = generate_validation_data(
        model_type='second_order',
        noise_level=0.05,
        n_points=12,
        CAf=100.0,
        k0_true=5e8,
        Ea_true=65.0,
        seed=42
    )
    
    result = fit_parameters(
        data['T'], data['tau'], data['CA'], data['CB'], data['CAf'],
        model_type='second_order',
        include_dH=False
    )
    
    print(f"拟合成功: {result['success']}")
    print(f"消息: {result['message']}")
    
    if result['success']:
        k0_fit = result['params'][0]
        Ea_fit = result['params'][1]
        
        k0_error = abs(k0_fit - data['k0_true']) / data['k0_true'] * 100
        Ea_error = abs(Ea_fit - data['Ea_true']) / data['Ea_true'] * 100
        
        print(f"\n真实 k0: {data['k0_true']:.4e} m³/(mol·s)")
        print(f"拟合 k0: {k0_fit:.4e} m³/(mol·s)")
        print(f"k0 相对误差: {k0_error:.2f}%")
        
        print(f"\n真实 Ea: {data['Ea_true']:.4f} kJ/mol")
        print(f"拟合 Ea: {Ea_fit:.4f} kJ/mol")
        print(f"Ea 相对误差: {Ea_error:.2f}%")
        
        print(f"\nR²: {result['R2']:.6f}")
        print(f"RMSE: {result['RMSE']:.4f}")
        
        print(f"\n残差正态性检验 p值: {result['shapiro_p']:.4f}")
        print(f"Durbin-Watson 统计量: {result['dw_stat']:.4f}")
        
        if result['R2'] > 0.95:
            print(f"\n✅ 测试通过: R² = {result['R2']:.6f} > 0.95")
            return True
        else:
            print(f"\n❌ 测试失败: R² = {result['R2']:.6f} < 0.95")
            return False
    else:
        print("\n❌ 测试失败: 拟合未成功")
        return False

def test_validation_dataset_3params():
    print("\n" + "=" * 60)
    print("测试三参数拟合 (k0, Ea, deltaH)")
    print("=" * 60)
    
    data = generate_validation_data(
        model_type='first_order',
        noise_level=0.02,
        n_points=20,
        CAf=100.0,
        k0_true=1e10,
        Ea_true=80.0,
        seed=42
    )
    
    result = fit_parameters(
        data['T'], data['tau'], data['CA'], data['CB'], data['CAf'],
        model_type='first_order',
        include_dH=True
    )
    
    print(f"拟合成功: {result['success']}")
    print(f"参数数量: {len(result['params'])}")
    print(f"参数名称: {result['param_names']}")
    print(f"R²: {result['R2']:.6f}")
    
    if result['success']:
        print(f"\n参数估计结果:")
        for i, name in enumerate(result['param_names']):
            print(f"  {name}: {result['params'][i]:.4e} ± {result['confidence_intervals'][i]:.4e}")
        
        print(f"\n参数相关系数矩阵:")
        print(result['corr_matrix'])
        
        high_corr = []
        corr = result['corr_matrix']
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(corr[i, j]) > 0.95:
                    high_corr.append(f"{result['param_names'][i]}-{result['param_names'][j]}: {corr[i,j]:.4f}")
        
        if high_corr:
            print(f"\n⚠️ 检测到高度相关参数:")
            for c in high_corr:
                print(f"  {c}")
        
        return True
    return False

def test_data_validation():
    print("\n" + "=" * 60)
    print("测试数据校验功能")
    print("=" * 60)
    
    T = np.array([350, 150, 400, 500, 900, 300])
    tau = np.array([10, 20, -5, 30, 40, 50])
    CA = np.array([70, 60, 50, 40, 30, -10])
    CB = np.array([30, 40, 50, 60, 70, 110])
    
    errors = validate_experimental_data(T, tau, CA, CB)
    mask = get_cell_error_mask(T, tau, CA, CB)
    
    print(f"检测到的错误数量: {len(errors)}")
    for err in errors:
        print(f"  {err}")
    
    print(f"\n错误掩码:")
    print(f"  T:   {mask['T']}")
    print(f"  tau: {mask['tau']}")
    print(f"  CA:  {mask['CA']}")
    print(f"  CB:  {mask['CB']}")
    
    expected_errors = 4
    if len(errors) == expected_errors:
        print(f"\n✅ 测试通过: 检测到 {len(errors)} 个错误 (预期 {expected_errors})")
        return True
    else:
        print(f"\n❌ 测试失败: 检测到 {len(errors)} 个错误 (预期 {expected_errors})")
        return False

def test_confidence_ellipse():
    print("\n" + "=" * 60)
    print("测试置信椭圆计算")
    print("=" * 60)
    
    data = generate_validation_data(
        model_type='first_order',
        noise_level=0.0,
        n_points=12,
        seed=42
    )
    
    result = fit_parameters(
        data['T'], data['tau'], data['CA'], data['CB'], data['CAf'],
        model_type='first_order',
        include_dH=False
    )
    
    if result['success']:
        ellipse = compute_confidence_ellipse(
            result['cov_matrix'], result['params'], 0, 1
        )
        print(f"置信椭圆点数: {len(ellipse)}")
        print(f"椭圆范围: k0 [{ellipse[:,0].min():.4e}, {ellipse[:,0].max():.4e}]")
        print(f"           Ea [{ellipse[:,1].min():.4f}, {ellipse[:,1].max():.4f}]")
        print("✅ 测试通过")
        return True
    return False

def test_sensitivity_curves():
    print("\n" + "=" * 60)
    print("测试灵敏度曲线计算")
    print("=" * 60)
    
    data = generate_validation_data(
        model_type='first_order',
        noise_level=0.0,
        n_points=12,
        seed=42
    )
    
    result = fit_parameters(
        data['T'], data['tau'], data['CA'], data['CB'], data['CAf'],
        model_type='first_order',
        include_dH=False
    )
    
    if result['success']:
        sensitivity_data, ref_idx = compute_sensitivity_curves(result)
        print(f"灵敏度参数数量: {len(sensitivity_data)}")
        for name, data_s in sensitivity_data.items():
            print(f"  {name}: {len(data_s['param_values'])} 个点, 输出范围 [{data_s['model_outputs'].min():.2f}, {data_s['model_outputs'].max():.2f}]")
        print("✅ 测试通过")
        return True
    return False

def test_insufficient_data():
    print("\n" + "=" * 60)
    print("测试不足6个数据点的情况")
    print("=" * 60)
    
    data = generate_validation_data(
        model_type='first_order',
        noise_level=0.0,
        n_points=5,
        seed=42
    )
    
    result = fit_parameters(
        data['T'], data['tau'], data['CA'], data['CB'], data['CAf'],
        model_type='first_order',
        include_dH=False
    )
    
    print(f"拟合成功: {result['success']}")
    print(f"消息: {result['message']}")
    
    if not result['success'] and "至少需要6个数据点" in result['message']:
        print("✅ 测试通过: 正确拒绝不足6个数据点的拟合请求")
        return True
    else:
        print("❌ 测试失败: 应该拒绝不足6个数据点的拟合请求")
        return False

if __name__ == '__main__':
    print("参数估计模块单元测试")
    print("=" * 60)
    
    tests = [
        test_validation_dataset_1,
        test_validation_dataset_2,
        test_validation_dataset_3params,
        test_data_validation,
        test_confidence_ellipse,
        test_sensitivity_curves,
        test_insufficient_data
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✅ 所有测试通过！")
    else:
        print(f"❌ {total - passed} 个测试失败")

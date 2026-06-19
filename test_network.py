import sys
import numpy as np
from reactor_network import ReactorNetwork

sys.path.insert(0, '.')

reactions = [
    {
        'reactants': ['A'], 'reactant_coeffs': {'A': 1.0}, 'reactant_orders': [1],
        'products': ['B'], 'product_coeffs': {'B': 1.0},
        'A': 5e5, 'Ea': 45.0, 'dH': -40.0
    },
    {
        'reactants': ['B'], 'reactant_coeffs': {'B': 1.0}, 'reactant_orders': [1],
        'products': ['C'], 'product_coeffs': {'C': 1.0},
        'A': 2e3, 'Ea': 65.0, 'dH': -25.0
    }
]

print("=" * 60)
print("测试1: 两级串联CSTR")
print("=" * 60)
network1 = ReactorNetwork()

n1 = network1.add_node("CSTR", "R1", x=100, y=200)
n1.V = 1.0
n1.T_c = 305.0
n1.UA = 3500.0

n2 = network1.add_node("CSTR", "R2", x=450, y=200)
n2.V = 1.5
n2.T_c = 288.0
n2.UA = 4000.0

network1.set_feed_target(n1.id, 1.0)
network1.add_connection(n1.id, n2.id, 1.0)

err = network1.validate_network()
print(f"网络校验: {'通过' if err is None else '失败: ' + err}")

ok = network1.solve(reactions)
print(f"求解状态: {'成功' if ok else '失败: ' + str(network1.solve_error)}")

if network1.solved:
    metrics = network1.get_metrics()
    print(f"总转化率: {metrics['total_conversion']*100:.2f}%")
    print(f"总选择性: {metrics['total_selectivity']*100:.2f}%")
    print(f"总收率: {metrics['total_yield']*100:.2f}%")
    print(f"热负荷: {metrics['total_heat_load']:.2f} kW")
    for nid, node in network1.nodes.items():
        print(f"  {node.name}: T={node.T_out:.1f}K, C_A={node.C_out[0]:.2f}, C_B={node.C_out[1]:.2f}, C_C={node.C_out[2]:.2f}")

print()
print("=" * 60)
print("测试2: 带回流循环")
print("=" * 60)
network2 = ReactorNetwork()

m1 = network2.add_node("CSTR", "混合器", x=100, y=200)
m1.V = 0.3
m1.T_c = 290.0
m1.UA = 1500.0

r1 = network2.add_node("CSTR", "R1主反应器", x=420, y=200)
r1.V = 2.5
r1.T_c = 300.0
r1.UA = 4000.0

network2.set_feed_target(m1.id, 1.0)
network2.add_connection(m1.id, r1.id, 1.0)
network2.add_connection(r1.id, m1.id, 0.3)

print(f"检测循环: {'有' if network2._has_cycle() else '无'}")

err = network2.validate_network()
print(f"网络校验: {'通过' if err is None else '失败: ' + err}")

ok = network2.solve(reactions)
print(f"求解状态: {'成功' if ok else '失败: ' + str(network2.solve_error)}")

if network2.solved:
    metrics = network2.get_metrics()
    print(f"总转化率: {metrics['total_conversion']*100:.2f}%")
    print(f"总选择性: {metrics['total_selectivity']*100:.2f}%")
    print(f"总收率: {metrics['total_yield']*100:.2f}%")
    for nid, node in network2.nodes.items():
        print(f"  {node.name}: T={node.T_out:.1f}K, C_A={node.C_out[0]:.2f}, C_B={node.C_out[1]:.2f}, C_C={node.C_out[2]:.2f}")
    r1_prod_split = 1.0 - sum(c.split_ratio for c in network2.get_outgoing_connections(r1.id))
    print(f"  R1产品分流比: {r1_prod_split:.2f}")

print()
print("=" * 60)
print("测试3: 并联分流")
print("=" * 60)
network3 = ReactorNetwork()

r1 = network3.add_node("CSTR", "R1", x=100, y=100)
r1.V = 0.8
r1.T_c = 295.0
r1.UA = 3000.0

r2 = network3.add_node("CSTR", "R2", x=100, y=320)
r2.V = 2.0
r2.T_c = 295.0
r2.UA = 3500.0

mix = network3.add_node("CSTR", "混合器", x=500, y=210)
mix.V = 0.3
mix.T_c = 290.0
mix.UA = 2000.0

network3.set_feed_target(r1.id, 0.4)
network3.set_feed_target(r2.id, 0.6)
network3.add_connection(r1.id, mix.id, 1.0)
network3.add_connection(r2.id, mix.id, 1.0)

err = network3.validate_network()
print(f"网络校验: {'通过' if err is None else '失败: ' + err}")

ok = network3.solve(reactions)
print(f"求解状态: {'成功' if ok else '失败: ' + str(network3.solve_error)}")

if network3.solved:
    metrics = network3.get_metrics()
    print(f"总转化率: {metrics['total_conversion']*100:.2f}%")
    print(f"总选择性: {metrics['total_selectivity']*100:.2f}%")
    print(f"总收率: {metrics['total_yield']*100:.2f}%")
    for nid, node in network3.nodes.items():
        print(f"  {node.name}: T={node.T_out:.1f}K, C_A={node.C_out[0]:.2f}, C_B={node.C_out[1]:.2f}, C_C={node.C_out[2]:.2f}")

print()
print("=" * 60)
print("测试4: 优化功能")
print("=" * 60)
if network1.solved:
    print("对两级串联网络进行优化 (最大化总收率)...")
    result, error = network1.optimize(
        reactions,
        objective="max_yield",
        T_max=450.0,
        X_min=0.2,
        V_total_max=10.0,
        T_c_bounds=(270.0, 340.0),
        T_feed_bounds=(280.0, 360.0),
    )
    if error:
        print(f"优化失败: {error}")
    else:
        print(f"优化成功: {result['success']}")
        base = result['base_metrics']
        opt = result['optimized_metrics']
        print(f"  收率: {base['total_yield']*100:.2f}% -> {opt['total_yield']*100:.2f}%")
        print(f"  转化率: {base['total_conversion']*100:.2f}% -> {opt['total_conversion']*100:.2f}%")
        print(f"  选择性: {base['total_selectivity']*100:.2f}% -> {opt['total_selectivity']*100:.2f}%")
        print(f"  参数调整:")
        for p in result['params']:
            if 'T_c' in p['description']:
                print(f"    {p['description']}: {p['before']:.1f}{p['unit']} -> {p['after']:.1f}{p['unit']}")

print()
print("=" * 60)
print("测试5: 多种反应器类型混合")
print("=" * 60)
network5 = ReactorNetwork()
c1 = network5.add_node("CSTR", "预混合CSTR", x=100, y=150)
c1.V = 0.5
c1.T_c = 300.0
c1.UA = 2000.0

p1 = network5.add_node("PFR", "管式反应器", x=400, y=150)
p1.L = 8.0
p1.A_cross = 0.01
p1.u = 0.5
p1.perimeter = 0.3
p1.UA_per_L = 800.0
p1.T_c = 295.0

network5.set_feed_target(c1.id, 1.0)
network5.add_connection(c1.id, p1.id, 1.0)

err = network5.validate_network()
print(f"网络校验: {'通过' if err is None else '失败: ' + err}")

ok = network5.solve(reactions)
print(f"求解状态: {'成功' if ok else '失败: ' + str(network5.solve_error)}")
if network5.solved:
    metrics = network5.get_metrics()
    print(f"总转化率: {metrics['total_conversion']*100:.2f}%")
    for nid, node in network5.nodes.items():
        print(f"  {node.name} ({node.reactor_type}): T={node.T_out:.1f}K, C_A={node.C_out[0]:.2f}, C_B={node.C_out[1]:.2f}")

print()
print("所有测试完成!")

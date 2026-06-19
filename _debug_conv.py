import sys
sys.path.insert(0, '.')
from reactor_network import ReactorNetwork, ReactorNode
import numpy as np

reactions = [
    {
        'A': 1e10, 'Ea': 80.0, 'dH': -100.0,
        'reactants': ['A'], 'products': ['B'],
        'reactant_coeffs': {'A': 1.0}, 'product_coeffs': {'B': 1.0},
        'reactant_orders': [1], 'product_orders': [0]
    }
]

print("reactions格式:", reactions[0])
print("检查reaction_rates:")
from reactor_models import reaction_rates
r = reaction_rates([100.0, 0, 0, 0, 0], 300.0, reactions)
print(f"  C=100, T=300K时速率 r={r}")
r = reaction_rates([100.0, 0, 0, 0, 0], 400.0, reactions)
print(f"  C=100, T=400K时速率 r={r}")

print("="*60)
print("测试单CSTR (和稳态模块相同参数)")
print("="*60)
net = ReactorNetwork()
r1 = net.add_node('CSTR', 'R1')
r1.V = 1.0
r1.T_c = 290.0
r1.UA = 2000.0
net.T_feed = 300.0
net.F_feed = 0.01
net.rho_cp = 4000.0
net.C_feed = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
net.feed_targets[r1.id] = 1.0

err = net.validate_network()
print('校验:', err if err else '通过')
ok = net.solve(reactions)
print('求解:', '成功' if ok else '失败 ' + str(net.solve_error))
if net.solved:
    m = net.get_metrics()
    tc = m['total_conversion'] * 100
    print(f'网络总转化率: {tc:.2f}%')
    print(f'R1 C_A_out={r1.C_out[0]:.2f}, T_out={r1.T_out:.2f}K')
    print(f'  节点转化率={(100-r1.C_out[0])/100*100:.2f}%')
    print(f'  停留时间 tau={r1.V/net.F_feed:.2f}s')

print()
print("="*60)
print("测试两级串联 (示例一参数)")
print("="*60)
net2 = ReactorNetwork()
n1 = net2.add_node('CSTR', 'R1')
n1.V = 1.0
n1.T_c = 310.0
n1.UA = 2000.0
n2 = net2.add_node('CSTR', 'R2')
n2.V = 1.5
n2.T_c = 285.0
n2.UA = 2500.0
net2.add_connection(n1.id, n2.id, 1.0)
net2.T_feed = 300.0
net2.F_feed = 0.01
net2.rho_cp = 4000.0
net2.C_feed = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
net2.feed_targets[n1.id] = 1.0

err = net2.validate_network()
print('校验:', err if err else '通过')
ok = net2.solve(reactions)
print('求解:', '成功' if ok else '失败 ' + str(net2.solve_error))
if net2.solved:
    m = net2.get_metrics()
    print(f'总转化率: {m["total_conversion"]*100:.2f}%')
    print(f'总选择性: {m["total_selectivity"]*100:.2f}%')
    print(f'R1 C_A={n1.C_out[0]:.2f} T={n1.T_out:.1f}K')
    print(f'R2 C_A={n2.C_out[0]:.2f} T={n2.T_out:.1f}K')
    print(f'  R2理论转化率={(n1.C_out[0]-n2.C_out[0])/max(n1.C_out[0],1e-6)*100:.2f}%')

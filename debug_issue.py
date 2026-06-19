import sys
sys.path.insert(0, '.')
from reactor_network import ReactorNetwork, ReactorNode
import numpy as np

COMPONENTS = ['A', 'B', 'C', 'D', 'E']

reactions = [
    {
        'reactants': ['A'], 'reactant_coeffs': {'A': 1.0}, 'reactant_orders': [1],
        'products': ['B'], 'product_coeffs': {'B': 1.0},
        'A': 1e10, 'Ea': 80.0, 'dH': -100.0
    }
]

print("="*60)
print("单反应器 CSTR 测试：相同参数对比")
print("="*60)
print()

print("【场景1】默认参数（类似单反应器稳态分析默认值：V=1.0, F=0.01, Tc=290, UA=2000")
node1 = ReactorNode(0, "R1", "CSTR", 0, 0)
node1.V = 1.0
node1.T_c = 290.0
node1.UA = 2000.0
T1, Cout1 = node1.solve_steady(np.array([100., 0, 0, 0, 0]), 300.0, 0.01, reactions)
print(f"  T_out = {T1:.2f} K")
print(f"  C_A   = {Cout1[0]:.2f} mol/m3")
print(f"  C_B   = {Cout1[1]:.2f} mol/m3")
print(f"  X_Conv = {(100 - Cout1[0]) / 100 * 100:.2f} %")
print()

print("【场景2】两级串联示例R1参数: V=1.0, F=0.01, Tc=310, UA=2000")
node2 = ReactorNode(1, "R1", "CSTR", 0, 0)
node2.V = 1.0
node2.T_c = 310.0
node2.UA = 2000.0
T2, Cout2 = node2.solve_steady(np.array([100., 0, 0, 0, 0]), 300.0, 0.01, reactions)
print(f"  T_out = {T2:.2f} K")
print(f"  C_A   = {Cout2[0]:.2f} mol/m3")
print(f"  X_Conv = {(100 - Cout2[0]) / 100 * 100:.2f} %")
print()

print("【场景3】完整两级串联网络: R1(Tc=310) -> R2(Tc=285)")
net = ReactorNetwork()
r1 = net.add_node("CSTR", "R1", 150, 200)
r1.V = 1.0
r1.T_c = 310.0
r1.UA = 2000.0
r2 = net.add_node("CSTR", "R2", 500, 200)
r2.V = 1.5
r2.T_c = 285.0
r2.UA = 2500.0
net.add_connection(r1.id, r2.id, 1.0)
net.feed_targets[r1.id] = 1.0
net.C_feed = np.array([100.0, 0, 0, 0, 0])
net.T_feed = 300.0
net.F_feed = 0.01

err = net.validate_network()
print(f"  校验: {'通过' if err is None else err}")
ok = net.solve(reactions)
print(f"  求解状态: {'成功' if ok else '失败: ' + str(net.solve_error)}")
for nid, node in net.nodes.items():
    print(f"  {node.name}: T={node.T_out:.2f}K, C_A={node.C_out[0]:.2f}, C_B={node.C_out[1]:.2f}, X={node.conversion*100:.2f}%")
m = net.get_metrics()
print(f"  网络总转化率 = {m['total_conversion']*100:.2f}%")
print()

print("="*60)
print("参数敏感性分析 (单反应器)")
print("="*60)

test_params = []
for ua in [500, 800, 1000, 1500, 2000, 3000, 5000, 8000]:
    for v in [1.0, 1.5, 2.0, 3.0]:
        for tc in [310, 320, 325, 330, 335, 340, 350, 360, 370]:
            for tf in [300, 320, 340]:
                test_params.append((v, tc, ua, 0.01, tf))

results = []
for V, Tc, UA, F, Tf in test_params:
    n = ReactorNode(99, "test", "CSTR", 0, 0)
    n.V = V
    n.T_c = Tc
    n.UA = UA
    T, Cout = n.solve_steady(np.array([100.,0,0,0,0]), Tf, F, reactions)
    X = (100 - Cout[0]) / 100 * 100
    if 500 > T > 300 and 30 < X < 98:
        results.append((X, T, f"V={V},Tc={Tc},UA={UA},Tf={Tf}", Cout[0]))

results.sort(reverse=True)
print(f"找到 {len(results)} 组参数满足条件 (T=300-500K, X=30-98%):")
for X, T, name, ca in results[:30]:
    print(f"  X={X:5.1f}% T={T:5.1f}K C_A={ca:5.1f} | {name}")

if not results:
    print(" 没有找到！测试PFR：")
    test_pfr = []
    for L in [5, 10, 15, 20]:
        for u in [0.05, 0.1, 0.2]:
            for Tc in [320, 340, 360]:
                for UAperL in [100, 200, 500]:
                    test_pfr.append((L,u,Tc,UAperL))
    for L,u,Tc,UAperL in test_pfr:
        n = ReactorNode(99, "test", "PFR", 0, 0)
        n.L = L
        n.u = u
        n.A_cross = 0.01
        n.perimeter = 0.35
        n.UA_per_L = UAperL
        n.T_c = Tc
        n.rho_cp = 4e3
        T, Cout = n.solve_steady(np.array([100.,0,0,0,0]), Tf, 0.01, reactions)
        X = (100 - Cout[0]) / 100 * 100
        if 500 > T > 300 and 30 < X < 99:
            print(f"  PFR X={X:5.1f}% T={T:5.1f}K | L={L},u={u},Tc={Tc},UA/L={UAperL}")

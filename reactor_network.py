import numpy as np
from scipy.optimize import minimize, fsolve
from reactor_models import (
    R_GAS, reaction_rates, stoich_net, get_steady_temp_fast
)
from collections import deque

REACTOR_TYPES = ["CSTR", "PFR", "半间歇", "多级CSTR"]
REACTOR_TYPE_CN = {
    "CSTR": "连续搅拌釜式反应器 (CSTR)",
    "PFR": "管式反应器 (PFR)",
    "半间歇": "半间歇反应器",
    "多级CSTR": "多级串联CSTR"
}

COMPONENTS = ['A', 'B', 'C', 'D', 'E']


class ReactorNode:
    def __init__(self, node_id, name="R1", reactor_type="CSTR", x=200, y=200):
        self.id = node_id
        self.name = name
        self.reactor_type = reactor_type
        self.x = x
        self.y = y
        
        self.V = 1.0
        self.F = 0.01
        self.rho_cp = 4.0e3
        self.T_c = 290.0
        self.UA = 2000.0
        
        self.A_cross = 0.01
        self.u = 1.0
        self.perimeter = 0.35
        self.UA_per_L = 1000.0
        self.L = 10.0
        
        self.V_max = 2.0
        self.F_in = 0.01
        
        self.n_stages = 2
        self.V_stages = [0.5, 0.5]
        self.T_c_stages = [290.0, 290.0]
        self.UA_stages = [2000.0, 2000.0]
        
        self.feed_temperature = None
        
        self.C_out = None
        self.T_out = None
        self.Q_cooling = 0.0
        self.conversion = 0.0
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'reactor_type': self.reactor_type,
            'x': self.x,
            'y': self.y,
            'V': self.V,
            'F': self.F,
            'rho_cp': self.rho_cp,
            'T_c': self.T_c,
            'UA': self.UA,
            'A_cross': self.A_cross,
            'u': self.u,
            'perimeter': self.perimeter,
            'UA_per_L': self.UA_per_L,
            'L': self.L,
            'V_max': self.V_max,
            'F_in': self.F_in,
            'n_stages': self.n_stages,
            'V_stages': self.V_stages,
            'T_c_stages': self.T_c_stages,
            'UA_stages': self.UA_stages,
            'feed_temperature': self.feed_temperature,
        }
    
    @classmethod
    def from_dict(cls, d):
        node = cls(d['id'], d['name'], d['reactor_type'], d['x'], d['y'])
        for k, v in d.items():
            if hasattr(node, k):
                setattr(node, k, v)
        return node
    
    def get_params(self):
        params = {
            'rho_cp': self.rho_cp,
            'T_c': self.T_c,
            'UA': self.UA,
        }
        if self.reactor_type == "CSTR":
            params['V'] = self.V
            params['F'] = self.F
        elif self.reactor_type == "PFR":
            params['A_cross'] = self.A_cross
            params['u'] = self.u
            params['perimeter'] = self.perimeter
            params['UA_per_L'] = self.UA_per_L
            params['L'] = self.L
        elif self.reactor_type == "半间歇":
            params['V_max'] = self.V_max
            params['F_in'] = self.F_in
        elif self.reactor_type == "多级CSTR":
            params['n_stages'] = self.n_stages
            params['F'] = self.F
            for i in range(self.n_stages):
                params[f'V_{i}'] = self.V_stages[i] if i < len(self.V_stages) else self.V / self.n_stages
                params[f'T_c_{i}'] = self.T_c_stages[i] if i < len(self.T_c_stages) else self.T_c
                params[f'UA_{i}'] = self.UA_stages[i] if i < len(self.UA_stages) else self.UA
        return params
    
    def solve_steady(self, C_in, T_in, F_in_total, reactions):
        params = self.get_params()
        params['C_f'] = np.array(C_in, dtype=float)
        params['T_f'] = float(T_in)
        
        C_in_A0 = C_in[0] if C_in[0] > 0 else 1e-10
        
        if self.reactor_type == "CSTR":
            params['F'] = max(F_in_total, 1e-10)
            tau = self.V / params['F']
            params['tau'] = tau
            T_out, C_out = self._direct_iteration_cstr(params, reactions)
        
        elif self.reactor_type == "PFR":
            T_out, C_out = self._solve_pfr(params, reactions, C_in, T_in)
        
        elif self.reactor_type == "半间歇":
            T_out, C_out = self._solve_semibatch(params, reactions, C_in, T_in)
        
        elif self.reactor_type == "多级CSTR":
            params['F'] = max(F_in_total, 1e-10)
            T_out, C_out = self._solve_multi_cstr(params, reactions, C_in, T_in)
        
        else:
            T_out, C_out = T_in, np.array(C_in, dtype=float)
        
        self.C_out = C_out
        self.T_out = T_out
        
        if C_in_A0 > 1e-10:
            self.conversion = max(0, min(1, (C_in_A0 - C_out[0]) / C_in_A0))
        else:
            self.conversion = 0.0
        
        if self.reactor_type == "CSTR":
            self.Q_cooling = self.UA * max(0, T_out - self.T_c)
        elif self.reactor_type == "多级CSTR":
            self.Q_cooling = sum(
                self.UA_stages[i] * max(0, T_out - self.T_c_stages[i])
                for i in range(min(self.n_stages, len(self.UA_stages)))
            )
        elif self.reactor_type == "PFR":
            self.Q_cooling = self.UA_per_L * self.L * max(0, T_out - self.T_c)
        else:
            self.Q_cooling = self.UA * max(0, T_out - self.T_c)
        
        return T_out, C_out
    
    def _direct_iteration_cstr(self, params, reactions, max_iter=500, tol=1e-8):
        V = params['V']
        F = params['F']
        tau = V / F
        C_f = params['C_f']
        T_f = params['T_f']
        rho_cp = params['rho_cp']
        T_c = params['T_c']
        UA = params['UA']
        
        C = C_f.copy()
        T = T_f
        
        for iter_idx in range(max_iter):
            rates = reaction_rates(C, T, reactions)
            
            rate_effects = np.zeros(5)
            for rxn, r in zip(reactions, rates):
                for comp, coeff in rxn.get('reactant_coeffs', {}).items():
                    j = COMPONENTS.index(comp)
                    rate_effects[j] -= coeff * r
                for comp, coeff in rxn.get('product_coeffs', {}).items():
                    j = COMPONENTS.index(comp)
                    rate_effects[j] += coeff * r
            
            C_new = np.zeros(5)
            for j in range(5):
                if rate_effects[j] < 0:
                    consumption_rate = abs(rate_effects[j])
                    denom = 1 + tau * consumption_rate / max(C_f[j], 1e-15)
                    C_new[j] = max(0, C_f[j] / denom)
                else:
                    C_new[j] = C_f[j] + tau * rate_effects[j]
                C_new[j] = max(0, C_new[j])
            
            Q_gen = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V
            Q_rem_sensible = rho_cp * F * (T - T_f)
            T_new = T_c + (Q_gen - Q_rem_sensible) / max(UA, 1e-10)
            T_new = max(200, min(800, T_new))
            
            dC = np.max(np.abs(C_new - C))
            dT = abs(T_new - T)
            
            C = 0.3 * C_new + 0.7 * C
            T = 0.3 * T_new + 0.7 * T
            
            if dC < tol and dT < tol * 10:
                break
        
        return T, C
    
    def _solve_pfr(self, params, reactions, C_in, T_in):
        L = params['L']
        u = params['u']
        A_cross = params['A_cross']
        rho_cp = params['rho_cp']
        T_c = params['T_c']
        UA_per_L = params['UA_per_L']
        P = params['perimeter']
        
        n_steps = 200
        dz = L / n_steps
        
        C = np.array(C_in, dtype=float)
        T = float(T_in)
        
        for step in range(n_steps):
            rates = reaction_rates(C, T, reactions)
            
            rate_effects = np.zeros(5)
            for rxn, r in zip(reactions, rates):
                for comp, coeff in rxn.get('reactant_coeffs', {}).items():
                    j = COMPONENTS.index(comp)
                    rate_effects[j] -= coeff * r
                for comp, coeff in rxn.get('product_coeffs', {}).items():
                    j = COMPONENTS.index(comp)
                    rate_effects[j] += coeff * r
            
            dCdz = rate_effects / max(u, 1e-10)
            
            Q_gen_vol = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates))
            Q_rem_vol = UA_per_L * (T - T_c) / max(A_cross, 1e-10) * P
            
            dTdz = (Q_gen_vol - Q_rem_vol) / max(rho_cp * u, 1e-10)
            
            C = np.clip(C + dCdz * dz, 0, None)
            T = T + dTdz * dz
            T = max(200, min(800, T))
        
        return T, C
    
    def _solve_semibatch(self, params, reactions, C_in, T_in):
        V_max = params['V_max']
        F_in = params['F_in']
        rho_cp = params['rho_cp']
        T_c = params['T_c']
        UA = params['UA']
        C_f = np.array(C_in, dtype=float)
        T_f = float(T_in)
        
        t_end = max(V_max / max(F_in, 1e-10) * 2, 100)
        n_steps = 200
        dt = t_end / n_steps
        
        V = 0.1
        C = C_f.copy() * 0.01
        T = T_f
        
        for step in range(n_steps):
            if V >= V_max:
                F_in_eff = 0.0
            else:
                F_in_eff = F_in
            
            rates = reaction_rates(C, T, reactions)
            
            rate_effects = np.zeros(5)
            for rxn, r in zip(reactions, rates):
                for comp, coeff in rxn.get('reactant_coeffs', {}).items():
                    j = COMPONENTS.index(comp)
                    rate_effects[j] -= coeff * r
                for comp, coeff in rxn.get('product_coeffs', {}).items():
                    j = COMPONENTS.index(comp)
                    rate_effects[j] += coeff * r
            
            dCdt = np.zeros(5)
            for j in range(5):
                dCdt[j] = (F_in_eff * (C_f[j] - C[j]) / max(V, 1e-10)) + rate_effects[j]
            
            Q_gen = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V
            Q_rem = rho_cp * F_in_eff * (T - T_f) + UA * (T - T_c)
            
            dTdt = (Q_gen - Q_rem) / max(rho_cp * V, 1e-10)
            dVdt = F_in_eff
            
            C = np.clip(C + dCdt * dt, 0, None)
            T = T + dTdt * dt
            T = max(200, min(800, T))
            V = min(V_max, V + dVdt * dt)
        
        return T, C
    
    def _solve_multi_cstr(self, params, reactions, C_in, T_in):
        n_stages = params['n_stages']
        F = params['F']
        rho_cp = params['rho_cp']
        
        C_prev = np.array(C_in, dtype=float)
        T_prev = float(T_in)
        
        for i in range(n_stages):
            V_i = params.get(f'V_{i}', self.V / n_stages)
            T_c_i = params.get(f'T_c_{i}', self.T_c)
            UA_i = params.get(f'UA_{i}', self.UA)
            
            tau_i = V_i / max(F, 1e-10)
            C_f_i = C_prev.copy()
            T_f_i = T_prev
            
            C = C_f_i.copy()
            T = T_f_i
            
            for _ in range(200):
                rates = reaction_rates(C, T, reactions)
                
                rate_effects = np.zeros(5)
                for rxn, r in zip(reactions, rates):
                    for comp, coeff in rxn.get('reactant_coeffs', {}).items():
                        j = COMPONENTS.index(comp)
                        rate_effects[j] -= coeff * r
                    for comp, coeff in rxn.get('product_coeffs', {}).items():
                        j = COMPONENTS.index(comp)
                        rate_effects[j] += coeff * r
                
                C_new = np.zeros(5)
                for j in range(5):
                    if rate_effects[j] < 0:
                        consumption_rate = abs(rate_effects[j])
                        denom = 1 + tau_i * consumption_rate / max(C_f_i[j], 1e-15)
                        C_new[j] = max(0, C_f_i[j] / denom)
                    else:
                        C_new[j] = C_f_i[j] + tau_i * rate_effects[j]
                    C_new[j] = max(0, C_new[j])
                
                Q_gen = sum(-rxn['dH'] * 1e3 * r for rxn, r in zip(reactions, rates)) * V_i
                Q_rem_sensible = rho_cp * F * (T - T_f_i)
                T_new = T_c_i + (Q_gen - Q_rem_sensible) / max(UA_i, 1e-10)
                T_new = max(200, min(800, T_new))
                
                dC = np.max(np.abs(C_new - C))
                dT = abs(T_new - T)
                
                C = 0.5 * C_new + 0.5 * C
                T = 0.5 * T_new + 0.5 * T
                
                if dC < 1e-8 and dT < 1e-6:
                    break
            
            C_prev = C
            T_prev = T
        
        return T_prev, C_prev


class Connection:
    def __init__(self, conn_id, source_id, target_id, split_ratio=1.0):
        self.id = conn_id
        self.source_id = source_id
        self.target_id = target_id
        self.split_ratio = split_ratio
        self.F_flow = 0.0
        self.C_flow = None
        self.T_flow = None


class ReactorNetwork:
    def __init__(self):
        self.nodes = {}
        self.connections = []
        self.next_node_id = 0
        self.next_conn_id = 0
        self.max_nodes = 6
        
        self.C_feed = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        self.T_feed = 300.0
        self.F_feed = 0.01
        self.rho_cp = 4.0e3
        
        self.feed_targets = {}
        
        self.solved = False
        self.solve_error = None
    
    def add_node(self, reactor_type="CSTR", name=None, x=None, y=None):
        if len(self.nodes) >= self.max_nodes:
            return None
        
        node_id = self.next_node_id
        self.next_node_id += 1
        
        if name is None:
            name = f"R{node_id + 1}"
        if x is None:
            x = 150 + (node_id % 3) * 250
        if y is None:
            y = 150 + (node_id // 3) * 200
        
        node = ReactorNode(node_id, name, reactor_type, x, y)
        self.nodes[node_id] = node
        return node
    
    def remove_node(self, node_id):
        if node_id not in self.nodes:
            return False
        
        self.connections = [
            c for c in self.connections
            if c.source_id != node_id and c.target_id != node_id
        ]
        
        if node_id in self.feed_targets:
            del self.feed_targets[node_id]
        
        del self.nodes[node_id]
        return True
    
    def add_connection(self, source_id, target_id, split_ratio=1.0):
        if source_id not in self.nodes or target_id not in self.nodes:
            return None, "节点不存在"
        
        if source_id == target_id:
            return None, "不能自连接"
        
        existing = [c for c in self.connections if c.source_id == source_id and c.target_id == target_id]
        if existing:
            return existing[0], None
        
        conn_id = self.next_conn_id
        self.next_conn_id += 1
        
        conn = Connection(conn_id, source_id, target_id, split_ratio)
        self.connections.append(conn)
        
        error = self.validate_network()
        return conn, error
    
    def remove_connection(self, conn_id):
        self.connections = [c for c in self.connections if c.id != conn_id]
        self._normalize_split_ratios()
        return True
    
    def set_feed_target(self, node_id, ratio=1.0):
        if node_id not in self.nodes:
            return False
        self.feed_targets[node_id] = ratio
        self._normalize_feed_ratios()
        return True
    
    def remove_feed_target(self, node_id):
        if node_id in self.feed_targets:
            del self.feed_targets[node_id]
            self._normalize_feed_ratios()
            return True
        return False
    
    def _normalize_feed_ratios(self):
        if not self.feed_targets:
            return
        total = sum(self.feed_targets.values())
        if total > 0:
            for k in self.feed_targets:
                self.feed_targets[k] /= total
    
    def _normalize_split_ratios(self):
        source_groups = {}
        for c in self.connections:
            if c.source_id not in source_groups:
                source_groups[c.source_id] = []
            source_groups[c.source_id].append(c)
        
        for source_id, conns in source_groups.items():
            total = sum(c.split_ratio for c in conns)
            if total > 0:
                for c in conns:
                    c.split_ratio /= total
    
    def validate_network(self):
        if not self.nodes:
            return "网络为空"
        
        if not self.feed_targets:
            return "请至少设置一个进料目标节点"
        
        for node_id, node in self.nodes.items():
            has_input = (node_id in self.feed_targets) or any(
                c.target_id == node_id for c in self.connections
            )
            has_output = any(
                c.source_id == node_id for c in self.connections
            ) or self._is_output_node(node_id)
            
            if not (has_input or has_output):
                return f"节点 {node.name} 是孤立节点，请连接它"
        
        source_groups = {}
        for c in self.connections:
            if c.source_id not in source_groups:
                source_groups[c.source_id] = []
            source_groups[c.source_id].append(c)
        
        for source_id, conns in source_groups.items():
            total = sum(c.split_ratio for c in conns)
            
            if total > 1.0 + 0.01:
                source_name = self.nodes[source_id].name
                return f"节点 {source_name} 的输出分流比之和不能超过1.0 (当前: {total:.3f})"
        
        return None
    
    def _is_output_node(self, node_id):
        out_conns = [c for c in self.connections if c.source_id == node_id]
        return len(out_conns) == 0
    
    def _topological_sort(self):
        in_degree = {nid: 0 for nid in self.nodes}
        adjacency = {nid: [] for nid in self.nodes}
        
        for c in self.connections:
            adjacency[c.source_id].append(c.target_id)
            in_degree[c.target_id] += 1
        
        for nid in self.feed_targets:
            if nid in in_degree:
                pass
        
        queue = deque()
        for nid, deg in in_degree.items():
            if deg == 0:
                queue.append(nid)
        
        result = []
        while queue:
            nid = queue.popleft()
            result.append(nid)
            for neighbor in adjacency[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.nodes):
            return None
        return result
    
    def _has_cycle(self):
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self.nodes}
        
        adjacency = {nid: [] for nid in self.nodes}
        for c in self.connections:
            adjacency[c.source_id].append(c.target_id)
        
        def dfs(nid):
            color[nid] = GRAY
            for neighbor in adjacency[nid]:
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[nid] = BLACK
            return False
        
        for nid in self.nodes:
            if color[nid] == WHITE:
                if dfs(nid):
                    return True
        return False
    
    def get_incoming_connections(self, node_id):
        return [c for c in self.connections if c.target_id == node_id]
    
    def get_outgoing_connections(self, node_id):
        return [c for c in self.connections if c.source_id == node_id]
    
    def solve(self, reactions):
        error = self.validate_network()
        if error:
            self.solved = False
            self.solve_error = error
            return False
        
        try:
            if self._has_cycle():
                return self._solve_with_cycles(reactions)
            else:
                return self._solve_feedforward(reactions)
        except Exception as e:
            self.solved = False
            self.solve_error = f"求解错误: {str(e)}"
            return False
    
    def _solve_feedforward(self, reactions):
        topo_order = self._topological_sort()
        if topo_order is None:
            self.solved = False
            self.solve_error = "拓扑排序失败"
            return False
        
        for nid in topo_order:
            node = self.nodes[nid]
            
            C_mix = np.zeros(5)
            T_mix = 0.0
            F_total = 0.0
            
            if nid in self.feed_targets:
                ratio = self.feed_targets[nid]
                F_feed = self.F_feed * ratio
                C_mix += self.C_feed * F_feed
                T_mix += self.T_feed * F_feed
                F_total += F_feed
            
            for conn in self.get_incoming_connections(nid):
                source_node = self.nodes[conn.source_id]
                if source_node.C_out is not None and source_node.T_out is not None:
                    F_flow = conn.F_flow
                    if F_flow <= 0:
                        out_total = sum(c.split_ratio for c in self.get_outgoing_connections(conn.source_id))
                        if out_total > 0:
                            F_flow = self.F_feed * conn.split_ratio
                    
                    C_mix += source_node.C_out * F_flow
                    T_mix += source_node.T_out * F_flow
                    F_total += F_flow
                    
                    conn.C_flow = source_node.C_out.copy()
                    conn.T_flow = source_node.T_out
                    conn.F_flow = F_flow
            
            if F_total > 0:
                C_in = C_mix / F_total
                T_in = T_mix / F_total
            else:
                C_in = self.C_feed.copy()
                T_in = self.T_feed
            
            T_out, C_out = node.solve_steady(C_in, T_in, F_total, reactions)
        
        self._update_connection_flows()
        self.solved = True
        self.solve_error = None
        return True
    
    def _solve_with_cycles(self, reactions, max_iter=50, tol=1e-6):
        all_ids = list(self.nodes.keys())
        
        old_C = {nid: self.C_feed.copy() for nid in all_ids}
        old_T = {nid: self.T_feed for nid in all_ids}
        
        xs = []
        fs = []
        
        for iteration in range(max_iter):
            for nid in all_ids:
                node = self.nodes[nid]
                
                C_mix = np.zeros(5)
                T_mix = 0.0
                F_total = 0.0
                
                if nid in self.feed_targets:
                    ratio = self.feed_targets[nid]
                    F_feed = self.F_feed * ratio
                    C_mix += self.C_feed * F_feed
                    T_mix += self.T_feed * F_feed
                    F_total += F_feed
                
                for conn in self.get_incoming_connections(nid):
                    source_id = conn.source_id
                    F_est = self.F_feed * conn.split_ratio
                    
                    if old_C[source_id] is not None:
                        C_mix += old_C[source_id] * F_est
                        T_mix += old_T[source_id] * F_est
                        F_total += F_est
                    
                    conn.C_flow = old_C[source_id].copy() if old_C[source_id] is not None else self.C_feed.copy()
                    conn.T_flow = old_T[source_id] if old_T[source_id] is not None else self.T_feed
                    conn.F_flow = F_est
                
                if F_total > 0:
                    C_in = C_mix / F_total
                    T_in = T_mix / F_total
                else:
                    C_in = self.C_feed.copy()
                    T_in = self.T_feed
                
                T_out, C_out = node.solve_steady(C_in, T_in, F_total, reactions)
                old_C[nid] = C_out.copy()
                old_T[nid] = T_out
            
            if iteration > 0:
                f_new = np.concatenate([
                    np.concatenate([old_C[nid], [old_T[nid]]]) for nid in all_ids
                ])
                
                if iteration >= 2 and len(xs) >= 2:
                    try:
                        dx = xs[-1] - xs[-2]
                        df = fs[-1] - fs[-2]
                        
                        denom = np.dot(df, dx)
                        if abs(denom) > 1e-15:
                            wegstein_factor = np.dot(xs[-1] - fs[-1], dx) / denom
                            x_new = fs[-1] + wegstein_factor * (fs[-1] - fs[-2])
                            x_new = np.clip(x_new, 0, None)
                            
                            idx = 0
                            for nid in all_ids:
                                old_C[nid] = np.maximum(0, x_new[idx:idx+5])
                                old_T[nid] = max(200, min(800, x_new[idx+5]))
                                idx += 6
                    except:
                        pass
                
                fs.append(f_new.copy())
                
                max_diff = 0.0
                if len(fs) >= 2:
                    max_diff = np.max(np.abs(fs[-1] - fs[-2]))
                    if max_diff < tol:
                        self._update_connection_flows()
                        self.solved = True
                        self.solve_error = None
                        return True
                
                xs.append(f_new.copy())
            else:
                fs.append(np.concatenate([
                    np.concatenate([old_C[nid], [old_T[nid]]]) for nid in all_ids
                ]))
                xs.append(fs[-1].copy())
        
        self._update_connection_flows()
        self.solved = True
        self.solve_error = f"迭代达到最大次数 ({max_iter})，解可能不完全收敛"
        return True
    
    def _update_connection_flows(self):
        for conn in self.connections:
            source_node = self.nodes[conn.source_id]
            if source_node.C_out is not None:
                conn.C_flow = source_node.C_out.copy()
                conn.T_flow = source_node.T_out
    
    def get_output_streams(self):
        outputs = []
        for nid, node in self.nodes.items():
            out_conns = self.get_outgoing_connections(nid)
            if len(out_conns) == 0 and node.C_out is not None:
                outputs.append({
                    'node_id': nid,
                    'node_name': node.name,
                    'C': node.C_out,
                    'T': node.T_out,
                    'F': self.F_feed
                })
        return outputs
    
    def get_metrics(self):
        if not self.solved:
            return None
        
        total_heat = 0.0
        max_T = 0.0
        node_conversions = {}
        
        for nid, node in self.nodes.items():
            total_heat += node.Q_cooling
            max_T = max(max_T, node.T_out if node.T_out else 0)
            node_conversions[nid] = node.conversion
        
        out_flows = {}
        in_flows = {}
        for nid in self.nodes:
            out_flows[nid] = 0.0
            in_flows[nid] = 0.0
        
        for nid, ratio in self.feed_targets.items():
            in_flows[nid] = self.F_feed * ratio
        
        for iter_i in range(100):
            new_in_flows = {}
            for nid in self.nodes:
                new_in_flows[nid] = 0.0
            
            for nid, ratio in self.feed_targets.items():
                new_in_flows[nid] += self.F_feed * ratio
            
            for conn in self.connections:
                src_id = conn.source_id
                tgt_id = conn.target_id
                
                total_split = sum(c.split_ratio for c in self.get_outgoing_connections(src_id))
                
                if total_split > 0:
                    flow_from_src = in_flows[src_id] * (conn.split_ratio / total_split)
                else:
                    flow_from_src = 0.0
                
                new_in_flows[tgt_id] += flow_from_src
            
            max_change = 0.0
            for nid in self.nodes:
                max_change = max(max_change, abs(new_in_flows[nid] - in_flows[nid]))
                in_flows[nid] = new_in_flows[nid]
            
            if max_change < 1e-12:
                break
        
        total_A_in = self.C_feed[0] * self.F_feed
        total_A_out = 0.0
        total_B_out = 0.0
        total_C_out = 0.0
        
        total_product_split_all = 0.0
        for nid, node in self.nodes.items():
            total_out_split = sum(c.split_ratio for c in self.get_outgoing_connections(nid))
            product_split = max(0.0, 1.0 - total_out_split)
            if len(self.get_outgoing_connections(nid)) == 0:
                product_split = 1.0
            total_product_split_all += product_split
        
        total_product_split_all = max(total_product_split_all, 1.0)
        
        for nid, node in self.nodes.items():
            if node.C_out is None:
                continue
            
            total_out_split = sum(c.split_ratio for c in self.get_outgoing_connections(nid))
            product_split = max(0.0, 1.0 - total_out_split)
            if len(self.get_outgoing_connections(nid)) == 0:
                product_split = 1.0
            
            if product_split > 0.001 or len(self.get_outgoing_connections(nid)) == 0:
                product_F = self.F_feed * (product_split / total_product_split_all)
                
                total_A_out += node.C_out[0] * product_F
                total_B_out += node.C_out[1] * product_F
                total_C_out += node.C_out[2] * product_F
        
        if total_A_out < 1e-10 and len(self.nodes) > 0:
            for nid, node in self.nodes.items():
                if node.C_out is None:
                    continue
                if len(self.get_outgoing_connections(nid)) == 0:
                    total_A_out += node.C_out[0] * self.F_feed
                    total_B_out += node.C_out[1] * self.F_feed
                    total_C_out += node.C_out[2] * self.F_feed
            
            if total_A_out < 1e-10:
                avg_CA = sum(n.C_out[0] for n in self.nodes.values() if n.C_out is not None) / max(len(self.nodes), 1)
                avg_CB = sum(n.C_out[1] for n in self.nodes.values() if n.C_out is not None) / max(len(self.nodes), 1)
                avg_CC = sum(n.C_out[2] for n in self.nodes.values() if n.C_out is not None) / max(len(self.nodes), 1)
                total_A_out = avg_CA * self.F_feed
                total_B_out = avg_CB * self.F_feed
                total_C_out = avg_CC * self.F_feed
        
        total_A_consumed = max(0.0, total_A_in - total_A_out)
        total_B_produced = max(0.0, total_B_out)
        total_C_produced = max(0.0, total_C_out)
        
        total_conversion = max(0.0, min(1.0, total_A_consumed / max(total_A_in, 1e-10)))
        
        if total_A_consumed > 1e-10:
            total_selectivity = total_B_produced / max(total_A_consumed, 1e-10)
        else:
            total_selectivity = 0.0
        total_selectivity = max(0.0, min(1.0, total_selectivity))
        
        total_yield = total_conversion * total_selectivity
        
        path_contributions = self._calculate_path_contributions()
        
        return {
            'total_conversion': float(total_conversion),
            'total_selectivity': float(total_selectivity),
            'total_yield': float(total_yield),
            'total_heat_load': float(total_heat / 1000.0),
            'max_temperature': float(max_T),
            'node_conversions': node_conversions,
            'total_A_consumed': float(total_A_consumed),
            'total_B_produced': float(total_B_produced),
            'total_C_produced': float(total_C_produced),
            'C_feed_value': float(self.C_feed[0]),
            'F_feed_value': float(self.F_feed),
            'path_contributions': path_contributions,
        }
    
    def _calculate_path_contributions(self):
        contributions = {}
        for nid, node in self.nodes.items():
            in_conns = self.get_incoming_connections(nid)
            is_first = (nid in self.feed_targets)
            if is_first and not in_conns:
                contributions[nid] = 1.0
            elif in_conns:
                total = 0.0
                for conn in in_conns:
                    src_contrib = contributions.get(conn.source_id, 0)
                    total += src_contrib * conn.split_ratio
                if nid in self.feed_targets:
                    total += self.feed_targets[nid]
                contributions[nid] = total
            else:
                contributions[nid] = 0.0
        return contributions
    
    def optimize(self, reactions, objective="max_yield", 
                 T_max=473.15, X_min=0.3, V_total_max=10.0,
                 T_c_bounds=(260.0, 340.0), T_feed_bounds=(280.0, 380.0)):
        if not self.solved:
            ok = self.solve(reactions)
            if not ok:
                return None, "网络无法求解"
        
        base_metrics = self.get_metrics()
        if base_metrics is None:
            return None, "无法获取基准指标"
        
        node_ids = list(self.nodes.keys())
        n_nodes = len(node_ids)
        
        x0 = []
        bounds = []
        var_descriptions = []
        
        for nid in node_ids:
            node = self.nodes[nid]
            x0.append(node.T_c)
            bounds.append(T_c_bounds)
            var_descriptions.append(f"T_c_{node.name}")
            
            if node.feed_temperature is not None:
                x0.append(node.feed_temperature)
            else:
                x0.append(self.T_feed)
            bounds.append(T_feed_bounds)
            var_descriptions.append(f"T_feed_{node.name}")
            
            out_conns = self.get_outgoing_connections(nid)
            if len(out_conns) >= 2:
                for i, conn in enumerate(out_conns[:-1]):
                    x0.append(conn.split_ratio)
                    bounds.append((0.05, 0.95))
                    var_descriptions.append(f"分流比_{self.nodes[conn.source_id].name}→{self.nodes[conn.target_id].name}")
        
        if len(x0) == 0:
            return None, "没有可优化的变量"
        
        original_node_params = {}
        for nid in node_ids:
            node = self.nodes[nid]
            original_node_params[nid] = {
                'T_c': node.T_c,
                'feed_temperature': node.feed_temperature,
            }
        
        original_connections = []
        for c in self.connections:
            original_connections.append({
                'id': c.id,
                'split_ratio': c.split_ratio,
            })
        
        def apply_vars(x):
            idx = 0
            for i, nid in enumerate(node_ids):
                node = self.nodes[nid]
                node.T_c = float(x[idx])
                idx += 1
                
                node.feed_temperature = float(x[idx])
                idx += 1
                
                out_conns = self.get_outgoing_connections(nid)
                if len(out_conns) >= 2:
                    ratios = []
                    for j in range(len(out_conns) - 1):
                        ratios.append(max(0.01, min(0.99, float(x[idx]))))
                        idx += 1
                    
                    if ratios:
                        sum_ratios = sum(ratios)
                        last_ratio = max(0.01, 1.0 - sum_ratios)
                        ratios.append(last_ratio)
                        
                        sum_total = sum(ratios)
                        for j, conn in enumerate(out_conns):
                            conn.split_ratio = ratios[j] / sum_total
        
        def objective_func(x):
            try:
                apply_vars(x)
                
                for nid in node_ids:
                    node = self.nodes[nid]
                    node.C_out = None
                    node.T_out = None
                
                self.solve(reactions)
                metrics = self.get_metrics()
                
                if metrics is None:
                    return 1e6
                
                penalty = 0.0
                
                if metrics['max_temperature'] > T_max:
                    penalty += (metrics['max_temperature'] - T_max) * 10000
                
                if metrics['total_conversion'] < X_min:
                    penalty += (X_min - metrics['total_conversion']) * 100000
                
                total_V = sum(self.nodes[nid].V for nid in node_ids)
                if total_V > V_total_max:
                    penalty += (total_V - V_total_max) * 1000
                
                for nid in node_ids:
                    node = self.nodes[nid]
                    if node.T_out is not None:
                        if node.T_out > T_max:
                            penalty += (node.T_out - T_max) * 5000
                
                if objective == "max_yield":
                    return -metrics['total_yield'] + penalty
                elif objective == "min_heat":
                    return metrics['total_heat_load'] + penalty
                else:
                    return -metrics['total_conversion'] + penalty
                    
            except Exception as e:
                return 1e6
        
        try:
            result = minimize(
                objective_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 60, 'ftol': 1e-5, 'maxfun': 120, 'eps': 1e-3}
            )
            
            apply_vars(result.x)
            for nid in node_ids:
                node = self.nodes[nid]
                node.C_out = None
                node.T_out = None
            
            self.solve(reactions)
            opt_metrics = self.get_metrics()
            
            for nid in node_ids:
                node = self.nodes[nid]
                orig = original_node_params[nid]
                node.T_c = orig['T_c']
                node.feed_temperature = orig['feed_temperature']
                node.C_out = None
                node.T_out = None
            
            for c in self.connections:
                for orig_c in original_connections:
                    if c.id == orig_c['id']:
                        c.split_ratio = orig_c['split_ratio']
                        break
            
            self.solve(reactions)
            
            optimized_params = []
            idx = 0
            for i, nid in enumerate(node_ids):
                node = self.nodes[nid]
                
                optimized_params.append({
                    'description': f"冷却温度 {node.name}",
                    'before': float(original_node_params[nid]['T_c']),
                    'after': float(result.x[idx]),
                    'unit': 'K'
                })
                idx += 1
                
                optimized_params.append({
                    'description': f"进料温度 {node.name}",
                    'before': float(original_node_params[nid]['feed_temperature'] or self.T_feed),
                    'after': float(result.x[idx]),
                    'unit': 'K'
                })
                idx += 1
                
                out_conns = self.get_outgoing_connections(nid)
                if len(out_conns) >= 2:
                    ratios_after = []
                    for j in range(len(out_conns) - 1):
                        ratios_after.append(float(result.x[idx]))
                        idx += 1
                    if ratios_after:
                        sum_r = sum(ratios_after)
                        last_r = 1.0 - sum_r
                        ratios_after.append(last_r)
                        sum_total = sum(ratios_after)
                        
                        for j, conn in enumerate(out_conns):
                            src_name = self.nodes[conn.source_id].name
                            tgt_name = self.nodes[conn.target_id].name
                            optimized_params.append({
                                'description': f"分流比 {src_name}→{tgt_name}",
                                'before': float(conn.split_ratio),
                                'after': float(ratios_after[j] / sum_total),
                                'unit': ''
                            })
            
            return {
                'base_metrics': base_metrics,
                'optimized_metrics': opt_metrics,
                'params': optimized_params,
                'success': result.success,
                'message': result.message,
            }, None
            
        except Exception as e:
            for nid in node_ids:
                node = self.nodes[nid]
                orig = original_node_params[nid]
                node.T_c = orig['T_c']
                node.feed_temperature = orig['feed_temperature']
                node.C_out = None
                node.T_out = None
            
            for c in self.connections:
                for orig_c in original_connections:
                    if c.id == orig_c['id']:
                        c.split_ratio = orig_c['split_ratio']
                        break
            
            self.solve(reactions)
            return None, f"优化失败: {str(e)}"

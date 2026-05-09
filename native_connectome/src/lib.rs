use std::collections::BTreeSet;
use std::ptr;
use std::slice;

const LAMBDA: f64 = 1.0;
const EPS_BOND: f64 = 200.0;
const LAMBDA_BOND: f64 = 1.0;
const BETA_DEBT: f64 = 0.1;
const D_DIFF: f64 = 0.05;
const TAU: f64 = 2.0;
const C_SIGNAL: f64 = 0.15811388300841897;

#[derive(Clone, Copy)]
struct Edge {
    target: usize,
    psi_curr: f64,
    psi_prev: f64,
    geom_weight: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VdmMetrics {
    pub tick: i64,
    pub n_walkers: u64,
    pub n_active: u64,
    pub n_warm: u64,
    pub n_computed: u64,
    pub bonds_instantiated: u64,
    pub bonds_removed: u64,
    pub bonds_total: u64,
    pub k_t: f64,
    pub phi_mean: f64,
    pub phi_var: f64,
    pub phi_dot_max: f64,
    pub mean_degree: f64,
    pub stimulus_active: u64,
    pub n_observed: u64,
    pub n_unsettled: u64,
}

pub struct NativeConnectome {
    n: usize,
    adj: Vec<Vec<Edge>>,
    phi_curr: Vec<f64>,
    phi_prev: Vec<f64>,
    phi_dot: Vec<f64>,
    debt: Vec<f64>,
    last_visit: Vec<i32>,
    observed_nodes: BTreeSet<usize>,
    unsettled_nodes: BTreeSet<usize>,
    j_ext: Vec<f64>,
    j_ext_nonzero: BTreeSet<usize>,
    directed_edge_count: usize,
    phi_sum: f64,
    phi_sumsq: f64,
    k_t: f64,
    tick: i32,
    transport_renormalization: f64,
    dynamic_bond_geom_weight: f64,
}

fn py_round(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let floor = value.floor();
    let frac = value - floor;
    if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1.0
    } else {
        let floor_i = floor as i64;
        if floor_i % 2 == 0 { floor } else { floor + 1.0 }
    }
}

fn clip01(value: f64) -> f64 {
    if value < 0.0 {
        0.0
    } else if value > 1.0 {
        1.0
    } else {
        value
    }
}

fn bond_potential_derivative(psi: f64) -> f64 {
    2.0 * LAMBDA_BOND * psi * (1.0 - psi) * (1.0 - 2.0 * psi)
}

fn bond_gradient_source(phi_i: f64, phi_j: f64) -> f64 {
    0.5 * (phi_j - phi_i) * (phi_j - phi_i)
}

fn bond_decoherence_floor(k_t: f64) -> f64 {
    if k_t <= 0.0 {
        0.0
    } else {
        (2.0 * k_t / EPS_BOND).sqrt()
    }
}

impl NativeConnectome {
    fn new(
        n: usize,
        edges: &[(usize, usize, f64)],
        transport_renormalization: f64,
        dynamic_bond_geom_weight: f64,
    ) -> Self {
        let mut conn = Self {
            n,
            adj: vec![Vec::new(); n],
            phi_curr: vec![0.0; n],
            phi_prev: vec![0.0; n],
            phi_dot: vec![0.0; n],
            debt: vec![0.0; n],
            last_visit: vec![-1; n],
            observed_nodes: BTreeSet::new(),
            unsettled_nodes: BTreeSet::new(),
            j_ext: vec![0.0; n],
            j_ext_nonzero: BTreeSet::new(),
            directed_edge_count: 0,
            phi_sum: 0.0,
            phi_sumsq: 0.0,
            k_t: 0.0,
            tick: 0,
            transport_renormalization,
            dynamic_bond_geom_weight,
        };
        for &(u, v, geom) in edges {
            if u < n && v < n && u != v && !conn.has_bond(u, v) {
                conn.add_bond(u, v, 0.5, geom);
            }
        }

        // Preserve the current v9 neighbor-degree symmetry seed exactly:
        // phi_i(0)=deg(i)/max_deg, with boundary nodes immediately observed.
        let max_deg = conn.adj.iter().map(|xs| xs.len()).max().unwrap_or(1).max(1);
        for i in 0..n {
            let val = conn.adj[i].len() as f64 / max_deg as f64;
            conn.phi_curr[i] = val;
            conn.phi_prev[i] = val;
            if conn.adj[i].len() < max_deg {
                conn.last_visit[i] = 0;
                conn.observed_nodes.insert(i);
                if (val - py_round(val)).abs() > 0.0 {
                    conn.unsettled_nodes.insert(i);
                }
            }
        }
        conn.phi_sum = conn.phi_curr.iter().sum();
        conn.phi_sumsq = conn.phi_curr.iter().map(|x| x * x).sum();
        conn
    }

    fn has_bond(&self, u: usize, v: usize) -> bool {
        self.adj[u].iter().any(|edge| edge.target == v)
    }

    fn add_bond(&mut self, u: usize, v: usize, psi_init: f64, geom_weight: f64) {
        self.adj[u].push(Edge {
            target: v,
            psi_curr: psi_init,
            psi_prev: psi_init,
            geom_weight,
        });
        self.adj[v].push(Edge {
            target: u,
            psi_curr: psi_init,
            psi_prev: psi_init,
            geom_weight,
        });
        self.directed_edge_count += 2;
    }

    fn stimulate(&mut self, indices: &[i32], amplitudes: &[f64]) {
        for (&idx, &amp) in indices.iter().zip(amplitudes.iter()) {
            if idx >= 0 {
                let i = idx as usize;
                if i < self.n {
                    self.j_ext[i] = amp;
                    if amp.abs() > 1e-15 {
                        self.j_ext_nonzero.insert(i);
                    }
                }
            }
        }
    }

    fn select_neighbor(&self, node: usize, emit_index: usize) -> Option<usize> {
        let nbrs = &self.adj[node];
        if nbrs.is_empty() {
            return None;
        }
        let w_sum: f64 = nbrs.iter().map(|e| e.psi_curr).sum();
        if w_sum < 1e-30 {
            return None;
        }
        let mut u = self.phi_dot[node].abs() * (1.0 + emit_index as f64);
        u -= u as i64 as f64;
        let mut cdf = 0.0;
        for edge in nbrs {
            cdf += edge.psi_curr / w_sum;
            if cdf >= u {
                return Some(edge.target);
            }
        }
        nbrs.last().map(|edge| edge.target)
    }

    fn propagate_one(
        &self,
        source: usize,
        emit_index: usize,
        v_th: f64,
        h_max: usize,
        active_set: &mut BTreeSet<usize>,
        bond_pairs: &mut BTreeSet<(usize, usize)>,
    ) -> usize {
        let mut current = source;
        let mut event_count = 0usize;
        for hop in 0..h_max {
            if hop > 0 && self.phi_dot[current].abs() <= v_th {
                break;
            }
            let Some(target) = self.select_neighbor(current, emit_index + hop) else {
                break;
            };
            event_count += 1;
            active_set.insert(current);
            active_set.insert(target);

            if self.phi_dot[current].abs() > v_th && self.phi_dot[target].abs() > v_th {
                if !self.has_bond(current, target) {
                    let pair = if current < target {
                        (current, target)
                    } else {
                        (target, current)
                    };
                    bond_pairs.insert(pair);
                }
            }
            if self.phi_dot[current].abs() > v_th {
                for edge in &self.adj[target] {
                    let k = edge.target;
                    if k != current && self.phi_dot[k].abs() > v_th && !self.has_bond(current, k) {
                        let pair = if current < k {
                            (current, k)
                        } else {
                            (k, current)
                        };
                        bond_pairs.insert(pair);
                    }
                }
            }
            current = target;
        }
        event_count
    }

    fn run_gauge_step(
        &self,
        candidate_nodes: &BTreeSet<usize>,
    ) -> (
        usize,
        BTreeSet<usize>,
        BTreeSet<usize>,
        BTreeSet<(usize, usize)>,
    ) {
        let v_th = if self.k_t <= 0.0 {
            0.0
        } else {
            (2.0 * self.k_t).sqrt()
        };
        let mut n_walkers = 0usize;
        let mut active_set = BTreeSet::new();
        let mut warm_set = BTreeSet::new();
        let mut bond_pairs = BTreeSet::new();
        if v_th <= 0.0 {
            return (n_walkers, active_set, warm_set, bond_pairs);
        }
        let h_max = ((C_SIGNAL / v_th).floor() as usize).max(1);
        let two_k_t = 2.0 * self.k_t;
        for &i in candidate_nodes {
            if i >= self.n {
                continue;
            }
            let ratio = self.phi_dot[i] * self.phi_dot[i] / two_k_t - 1.0;
            let count = ratio.max(0.0).floor() as usize;
            if count == 0 {
                continue;
            }
            active_set.insert(i);
            for ei in 0..count {
                n_walkers +=
                    self.propagate_one(i, ei, v_th, h_max, &mut active_set, &mut bond_pairs);
            }
        }
        for &i in &active_set {
            for edge in &self.adj[i] {
                if !active_set.contains(&edge.target) {
                    warm_set.insert(edge.target);
                }
            }
        }
        (n_walkers, active_set, warm_set, bond_pairs)
    }

    fn local_laplacian(&self, node: usize) -> f64 {
        let mut acc = 0.0;
        let phi_i = self.phi_curr[node];
        for edge in &self.adj[node] {
            acc += edge.psi_curr * edge.geom_weight * (self.phi_curr[edge.target] - phi_i);
        }
        self.transport_renormalization * acc
    }

    fn measure_cold_node(&mut self, node: usize, tick_now: i32) {
        let t_last = self.last_visit[node].max(0);
        let dt_gap = tick_now - t_last;
        if dt_gap <= 1 {
            return;
        }
        let tau_eff = TAU * (BETA_DEBT * self.debt[node]).exp();
        let old_phi = self.phi_curr[node];
        let phi_well = py_round(self.phi_curr[node]);
        let decay = (-(dt_gap as f64) / tau_eff).exp();
        self.phi_curr[node] = phi_well + (self.phi_curr[node] - phi_well) * decay;
        self.phi_prev[node] = phi_well + (self.phi_prev[node] - phi_well) * decay;
        let new_phi = self.phi_curr[node];
        self.phi_sum += new_phi - old_phi;
        self.phi_sumsq += new_phi * new_phi - old_phi * old_phi;
        let bond_decay = (-(dt_gap as f64) / EPS_BOND).exp();
        for edge in &mut self.adj[node] {
            let psi_well = py_round(edge.psi_curr);
            edge.psi_curr = psi_well + (edge.psi_curr - psi_well) * bond_decay;
        }
        let tau_eff_debt = TAU * (BETA_DEBT * self.debt[node]).exp();
        self.debt[node] *= (-(dt_gap as f64) / tau_eff_debt).exp();
    }

    fn decohere_bonds(&mut self, compute_list: &[usize]) -> usize {
        if self.k_t <= 0.0 {
            return 0;
        }
        let eta_floor = bond_decoherence_floor(self.k_t);
        let mut removed = 0usize;
        for &i in compute_list {
            if i >= self.n || self.adj[i].is_empty() {
                continue;
            }
            let mut dead = Vec::new();
            let mut alive = Vec::with_capacity(self.adj[i].len());
            for edge in self.adj[i].iter().copied() {
                if edge.psi_curr >= eta_floor {
                    alive.push(edge);
                } else {
                    dead.push(edge.target);
                }
            }
            if dead.is_empty() {
                continue;
            }
            removed += dead.len();
            self.adj[i] = alive;
            for j in dead {
                let before = self.adj[j].len();
                self.adj[j].retain(|edge| edge.target != i);
                let removed_mirror = before - self.adj[j].len();
                self.directed_edge_count =
                    self.directed_edge_count.saturating_sub(1 + removed_mirror);
            }
        }
        removed
    }

    fn step(&mut self, tick: i32) -> VdmMetrics {
        self.tick = tick;
        let stim_nodes: Vec<usize> = self
            .j_ext_nonzero
            .iter()
            .copied()
            .filter(|&i| i < self.n && self.j_ext[i].abs() > 1e-15)
            .collect();
        let mut emitter_candidates = self.unsettled_nodes.clone();
        for &i in &stim_nodes {
            emitter_candidates.insert(i);
        }

        let (n_walkers, mut active_set, mut warm_set, bond_pairs) =
            self.run_gauge_step(&emitter_candidates);

        for &idx in &stim_nodes {
            active_set.insert(idx);
            for edge in &self.adj[idx] {
                if !active_set.contains(&edge.target) {
                    warm_set.insert(edge.target);
                }
            }
        }
        for &i in &stim_nodes {
            self.last_visit[i] = tick;
            self.observed_nodes.insert(i);
        }
        for &i in &warm_set {
            if self.last_visit[i] < 0 {
                self.last_visit[i] = tick;
            }
            self.observed_nodes.insert(i);
        }

        let mut compute_set = active_set.clone();
        compute_set.extend(warm_set.iter().copied());
        compute_set.extend(self.unsettled_nodes.iter().copied());
        let compute_list: Vec<usize> = compute_set.iter().copied().collect();

        for &i in &active_set {
            if self.last_visit[i] >= 0 && self.last_visit[i] < tick - 1 {
                self.measure_cold_node(i, tick);
            }
            self.last_visit[i] = tick;
            self.observed_nodes.insert(i);
        }

        let mut phi_new = self.phi_curr.clone();
        if !compute_list.is_empty() {
            let old_values: Vec<f64> = compute_list.iter().map(|&i| self.phi_curr[i]).collect();
            for &i in &compute_list {
                let tau_eff = TAU * (BETA_DEBT * self.debt[i]).exp();
                let phi_i = self.phi_curr[i];
                let d_v_i = 2.0 * LAMBDA * phi_i * (1.0 - phi_i) * (1.0 - 2.0 * phi_i);
                let rhs = D_DIFF * self.local_laplacian(i) - d_v_i + self.j_ext[i];
                phi_new[i] = (rhs + (2.0 * tau_eff + 1.0) * self.phi_curr[i]
                    - tau_eff * self.phi_prev[i])
                    / (tau_eff + 1.0);
            }
            for &i in &compute_list {
                phi_new[i] = clip01(phi_new[i]);
            }
            let mut delta_sum = 0.0;
            let mut delta_sumsq = 0.0;
            for (pos, &i) in compute_list.iter().enumerate() {
                delta_sum += phi_new[i] - old_values[pos];
                delta_sumsq += phi_new[i] * phi_new[i] - old_values[pos] * old_values[pos];
            }
            self.phi_sum += delta_sum;
            self.phi_sumsq += delta_sumsq;
        }

        for &i in &compute_list {
            let phi_i = self.phi_curr[i];
            let mut next_edges = self.adj[i].clone();
            for edge in &mut next_edges {
                let psi_c = edge.psi_curr;
                let psi_p = edge.psi_prev;
                let rhs_bond = -bond_potential_derivative(psi_c)
                    + bond_gradient_source(phi_i, self.phi_curr[edge.target]);
                let psi_new = (rhs_bond + (2.0 * EPS_BOND + 1.0) * psi_c - EPS_BOND * psi_p)
                    / (EPS_BOND + 1.0);
                edge.psi_prev = psi_c;
                edge.psi_curr = clip01(psi_new);
            }
            self.adj[i] = next_edges;
        }

        let mut instantiated = 0usize;
        for (u, v) in bond_pairs {
            if u < self.n && v < self.n && !self.has_bond(u, v) {
                self.add_bond(u, v, 0.0, self.dynamic_bond_geom_weight);
                instantiated += 1;
            }
        }

        let removed = self.decohere_bonds(&compute_list);

        if compute_list.len() > 1 {
            let dots: Vec<f64> = compute_list
                .iter()
                .map(|&i| phi_new[i] - self.phi_curr[i])
                .collect();
            let mean = dots.iter().sum::<f64>() / dots.len() as f64;
            let var = dots
                .iter()
                .map(|x| {
                    let d = *x - mean;
                    d * d
                })
                .sum::<f64>()
                / dots.len() as f64;
            self.k_t = (0.5 * var).max(0.0);
        }

        if !compute_list.is_empty() {
            for &i in &compute_list {
                let speed = (phi_new[i] - self.phi_curr[i]).abs();
                self.debt[i] += speed;
            }
            for &i in &compute_list {
                let tau_eff_i = TAU * (BETA_DEBT * self.debt[i]).exp();
                self.debt[i] *= (-1.0 / tau_eff_i).exp();
            }
        }

        let next_pos_thresh = (self.k_t.max(0.0) / LAMBDA).sqrt();
        let next_vel_thresh = (2.0 * self.k_t.max(0.0)).sqrt();
        if !compute_list.is_empty() {
            let mut next_unsettled = BTreeSet::new();
            for &node in &compute_list {
                let dot_value = phi_new[node] - self.phi_curr[node];
                let nearest_well = py_round(phi_new[node]);
                if (phi_new[node] - nearest_well).abs() > next_pos_thresh
                    || dot_value.abs() > next_vel_thresh
                {
                    next_unsettled.insert(node);
                }
            }
            for &node in self.unsettled_nodes.difference(&compute_set) {
                next_unsettled.insert(node);
            }
            self.unsettled_nodes = next_unsettled;
        }

        self.phi_prev = self.phi_curr.clone();
        self.phi_curr = phi_new;
        for i in 0..self.n {
            self.phi_dot[i] = self.phi_curr[i] - self.phi_prev[i];
        }

        let zero_nodes: Vec<usize> = self.j_ext_nonzero.iter().copied().collect();
        for idx in zero_nodes {
            if idx < self.n {
                self.j_ext[idx] = 0.0;
            }
        }
        self.j_ext_nonzero.clear();

        let phi_mean = if self.n > 0 {
            self.phi_sum / self.n as f64
        } else {
            0.0
        };
        let phi_var = if self.n > 0 {
            (self.phi_sumsq / self.n as f64 - phi_mean * phi_mean).max(0.0)
        } else {
            0.0
        };
        let mut phi_dot_max = 0.0f64;
        for &i in &compute_list {
            phi_dot_max = phi_dot_max.max(self.phi_dot[i].abs());
        }

        VdmMetrics {
            tick: tick as i64,
            n_walkers: n_walkers as u64,
            n_active: active_set.len() as u64,
            n_warm: warm_set.len() as u64,
            n_computed: compute_list.len() as u64,
            bonds_instantiated: instantiated as u64,
            bonds_removed: removed as u64,
            bonds_total: (self.directed_edge_count / 2) as u64,
            k_t: self.k_t,
            phi_mean,
            phi_var,
            phi_dot_max,
            mean_degree: if self.n > 0 {
                self.directed_edge_count as f64 / self.n as f64
            } else {
                0.0
            },
            stimulus_active: stim_nodes.len() as u64,
            n_observed: self.observed_nodes.len() as u64,
            n_unsettled: self.unsettled_nodes.len() as u64,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_new(
    n: usize,
    u_ptr: *const i32,
    v_ptr: *const i32,
    geom_ptr: *const f64,
    edge_count: usize,
    transport_renormalization: f64,
    dynamic_bond_geom_weight: f64,
) -> *mut NativeConnectome {
    if n == 0 {
        return ptr::null_mut();
    }
    let us = unsafe { slice::from_raw_parts(u_ptr, edge_count) };
    let vs = unsafe { slice::from_raw_parts(v_ptr, edge_count) };
    let gs = unsafe { slice::from_raw_parts(geom_ptr, edge_count) };
    let mut edges = Vec::with_capacity(edge_count);
    for idx in 0..edge_count {
        if us[idx] >= 0 && vs[idx] >= 0 {
            edges.push((us[idx] as usize, vs[idx] as usize, gs[idx]));
        }
    }
    Box::into_raw(Box::new(NativeConnectome::new(
        n,
        &edges,
        transport_renormalization,
        dynamic_bond_geom_weight,
    )))
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_free(ptr_conn: *mut NativeConnectome) {
    if !ptr_conn.is_null() {
        unsafe {
            drop(Box::from_raw(ptr_conn));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_stimulate(
    ptr_conn: *mut NativeConnectome,
    idx_ptr: *const i32,
    amp_ptr: *const f64,
    count: usize,
) -> i32 {
    if ptr_conn.is_null() {
        return -1;
    }
    let conn = unsafe { &mut *ptr_conn };
    let idxs = unsafe { slice::from_raw_parts(idx_ptr, count) };
    let amps = unsafe { slice::from_raw_parts(amp_ptr, count) };
    conn.stimulate(idxs, amps);
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_step(
    ptr_conn: *mut NativeConnectome,
    tick: i32,
    out_metrics: *mut VdmMetrics,
) -> i32 {
    if ptr_conn.is_null() || out_metrics.is_null() {
        return -1;
    }
    let conn = unsafe { &mut *ptr_conn };
    let metrics = conn.step(tick);
    unsafe {
        *out_metrics = metrics;
    }
    0
}

fn copy_f64_slice(src: &[f64], out: *mut f64, cap: usize) -> usize {
    let n = src.len().min(cap);
    if n > 0 {
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), out, n);
        }
    }
    n
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_copy_phi_curr(
    ptr_conn: *const NativeConnectome,
    out: *mut f64,
    cap: usize,
) -> usize {
    if ptr_conn.is_null() {
        return 0;
    }
    copy_f64_slice(&unsafe { &*ptr_conn }.phi_curr, out, cap)
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_copy_phi_prev(
    ptr_conn: *const NativeConnectome,
    out: *mut f64,
    cap: usize,
) -> usize {
    if ptr_conn.is_null() {
        return 0;
    }
    copy_f64_slice(&unsafe { &*ptr_conn }.phi_prev, out, cap)
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_copy_phi_dot(
    ptr_conn: *const NativeConnectome,
    out: *mut f64,
    cap: usize,
) -> usize {
    if ptr_conn.is_null() {
        return 0;
    }
    copy_f64_slice(&unsafe { &*ptr_conn }.phi_dot, out, cap)
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_copy_debt(
    ptr_conn: *const NativeConnectome,
    out: *mut f64,
    cap: usize,
) -> usize {
    if ptr_conn.is_null() {
        return 0;
    }
    copy_f64_slice(&unsafe { &*ptr_conn }.debt, out, cap)
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_copy_last_visit(
    ptr_conn: *const NativeConnectome,
    out: *mut i32,
    cap: usize,
) -> usize {
    if ptr_conn.is_null() {
        return 0;
    }
    let src = &unsafe { &*ptr_conn }.last_visit;
    let n = src.len().min(cap);
    if n > 0 {
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), out, n);
        }
    }
    n
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_node_degree(
    ptr_conn: *const NativeConnectome,
    node: usize,
) -> usize {
    if ptr_conn.is_null() {
        return 0;
    }
    let conn = unsafe { &*ptr_conn };
    if node >= conn.n {
        0
    } else {
        conn.adj[node].len()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_copy_adj(
    ptr_conn: *const NativeConnectome,
    node: usize,
    out_targets: *mut i32,
    out_psi_curr: *mut f64,
    out_psi_prev: *mut f64,
    out_geom: *mut f64,
    cap: usize,
) -> usize {
    if ptr_conn.is_null() {
        return 0;
    }
    let conn = unsafe { &*ptr_conn };
    if node >= conn.n {
        return 0;
    }
    let n = conn.adj[node].len().min(cap);
    for k in 0..n {
        let edge = conn.adj[node][k];
        unsafe {
            *out_targets.add(k) = edge.target as i32;
            *out_psi_curr.add(k) = edge.psi_curr;
            *out_psi_prev.add(k) = edge.psi_prev;
            *out_geom.add(k) = edge.geom_weight;
        }
    }
    n
}

fn copy_set(src: &BTreeSet<usize>, out: *mut i32, cap: usize) -> usize {
    let n = src.len().min(cap);
    for (k, value) in src.iter().take(n).enumerate() {
        unsafe {
            *out.add(k) = *value as i32;
        }
    }
    n
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_observed_count(ptr_conn: *const NativeConnectome) -> usize {
    if ptr_conn.is_null() {
        0
    } else {
        unsafe { &*ptr_conn }.observed_nodes.len()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_unsettled_count(ptr_conn: *const NativeConnectome) -> usize {
    if ptr_conn.is_null() {
        0
    } else {
        unsafe { &*ptr_conn }.unsettled_nodes.len()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_copy_observed(
    ptr_conn: *const NativeConnectome,
    out: *mut i32,
    cap: usize,
) -> usize {
    if ptr_conn.is_null() {
        return 0;
    }
    copy_set(&unsafe { &*ptr_conn }.observed_nodes, out, cap)
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_copy_unsettled(
    ptr_conn: *const NativeConnectome,
    out: *mut i32,
    cap: usize,
) -> usize {
    if ptr_conn.is_null() {
        return 0;
    }
    copy_set(&unsafe { &*ptr_conn }.unsettled_nodes, out, cap)
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_k_t(ptr_conn: *const NativeConnectome) -> f64 {
    if ptr_conn.is_null() {
        0.0
    } else {
        unsafe { &*ptr_conn }.k_t
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_phi_sum(ptr_conn: *const NativeConnectome) -> f64 {
    if ptr_conn.is_null() {
        0.0
    } else {
        unsafe { &*ptr_conn }.phi_sum
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_phi_sumsq(ptr_conn: *const NativeConnectome) -> f64 {
    if ptr_conn.is_null() {
        0.0
    } else {
        unsafe { &*ptr_conn }.phi_sumsq
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vdm_connectome_directed_edge_count(ptr_conn: *const NativeConnectome) -> usize {
    if ptr_conn.is_null() {
        0
    } else {
        unsafe { &*ptr_conn }.directed_edge_count
    }
}

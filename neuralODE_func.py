import torch
import torch.nn as nn
from typing import List
from new_3D_strains import rod_strains_from_local_qs

class neuralODE(nn.Module):
    def __init__(self, edges: torch.Tensor, l0_edges: List[float], n_nodes: int, ndof: int, m_per_dof: float = 1.0, c_per_dof: float = 0.1, \
                 g: float = 9.81, freeDOF: List[int]=None, \
                    energy_nn: nn.Module = None, bend_twist_springs: torch.Tensor = None, \
                    d1_init: torch.Tensor = None, d2_init: torch.Tensor = None, dtype=torch.float32):
        super().__init__()
        self.ndof = ndof
        self.n_nodes = n_nodes
        self.freeDOF = freeDOF
        self.call_count = 0
        # Neural elastic energy
        assert energy_nn is not None, "Pass an EnergyNN instance"
        self.energy_model = energy_nn

        # edges, directors 
        self.register_buffer("edges", edges)  # (E,2)
        self.register_buffer("l0_edges", torch.tensor(l0_edges, dtype=dtype))  # (E,)
        self.n_edges = edges.shape[0]
        self.register_buffer("d1_init", d1_init)  # (E,3)
        self.register_buffer("d2_init", d2_init)  # (E,3
        
        # Indices / masks
        if freeDOF is None:
            freeDOF = list(range(ndof))
        free_idx = torch.as_tensor(freeDOF, dtype=torch.long)
        fixed_mask = torch.ones(ndof, dtype=torch.bool)
        fixed_mask[free_idx] = False
        self.register_buffer("free_idx", free_idx)
        self.register_buffer("fixed_mask", fixed_mask)
        self.register_buffer("free_mask", ~fixed_mask)  # same shape

        # Diagonal mass and damping for now (easy to replace with full matrices later)
        M = torch.eye(ndof) * m_per_dof       # (ndof, ndof)
        C = torch.eye(ndof) * c_per_dof       # (ndof, ndof)

        # Gravity as external force: acts in z only (every 3rd DOF starting from index 2)
        f_ext = torch.zeros(ndof)
        f_ext[2::3] = -m_per_dof * g          # (ndof,)

        # Register as buffers so dtype/device track the module (and no grads)
        self.register_buffer("M", M)
        self.register_buffer("C", C)
        self.register_buffer("f_ext", f_ext)

        # Pre-slice free blocks once (for diagonal M this is overkill but scales)
        self.register_buffer("M_ff", M.index_select(0, free_idx).index_select(1, free_idx))
        self.register_buffer("C_ff", C.index_select(0, free_idx).index_select(1, free_idx))

        # stencils
        self.register_buffer("bend_twist_springs", bend_twist_springs)

            # ----------------------------------------------------------
            # Vectorized elastic force using bend_twist_springs
            # Local order: [xi,xj,xk, theta_prev, theta_next]
            # Global order: [all nodes xyz..., all edge thetas...]
            # ----------------------------------------------------------

    def elastic_force(self, q: torch.Tensor, eps=1e-12) -> torch.Tensor:
        device = q.device
        batch_shape = q.shape[:-1]
        ndof = q.shape[-1]

        N = self.n_nodes
        nodal_ndof = 3 * N
        theta_offset = nodal_ndof

        # infer E from q (robust)
        E = ndof - nodal_ndof
        if E <= 0:
            raise RuntimeError(f"Need edge DOFs in q. Got ndof={ndof}, 3*n_nodes={nodal_ndof} => E={E}")

        springs_all = self.bend_twist_springs.to(device).long()  # (S,5) (i,j,k,e_prev,e_next)
        edges = self.edges.to(device).long()

        # filter springs if q has fewer edge dofs than mesh expects (debug-friendly)
        e_prev_all = springs_all[:, 3]
        e_next_all = springs_all[:, 4]
        mask = (e_prev_all < E) & (e_next_all < E)
        springs = springs_all[mask]
        S = springs.shape[0]
        if S == 0:
            return torch.zeros_like(q)

        # flatten batch
        Btot = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) else 1
        q_flat = q.reshape(Btot, ndof)

        # split
        x = q_flat[:, :nodal_ndof].reshape(Btot, N, 3)     # (B,N,3)
        theta = q_flat[:, theta_offset:].reshape(Btot, E)  # (B,E)

        # spring indices
        i = springs[:, 0]; j = springs[:, 1]; k = springs[:, 2]
        e_prev = springs[:, 3]; e_next = springs[:, 4]

        # gather nodal coords
        x_i = x[:, i, :]
        x_j = x[:, j, :]
        x_k = x[:, k, :]

        # gather thetas
        th_prev = theta[:, e_prev]
        th_next = theta[:, e_next]

        # local dof vector (B,S,11) and make it the differentiation input
        q_s = torch.cat([x_i, x_j, x_k, th_prev[..., None], th_next[..., None]], dim=-1)
        # q_s = q_s.clone().requires_grad_(True)
        q_s.requires_grad_(True)


        # gather per-edge rest lengths for the two incident edges
        l0 = torch.as_tensor(self.l0_edges, device=device, dtype=q.dtype)
        if l0.dim() == 0:
            l0 = l0.expand(E)
        l0_prev = l0[e_prev].view(1, S).expand(Btot, S)
        l0_next = l0[e_next].view(1, S).expand(Btot, S)

        # gather reference directors for the two incident edges
        d1_all = torch.as_tensor(self.d1_init, device=device, dtype=q.dtype)  # (E,3)
        d2_all = torch.as_tensor(self.d2_init, device=device, dtype=q.dtype)  # (E,3)
        d1_prev = d1_all[e_prev].view(1, S, 3).expand(Btot, S, 3)
        d2_prev = d2_all[e_prev].view(1, S, 3).expand(Btot, S, 3)
        d1_next = d1_all[e_next].view(1, S, 3).expand(Btot, S, 3)
        d2_next = d2_all[e_next].view(1, S, 3).expand(Btot, S, 3)

        # strains computed FROM q_s => graph is connected
        strains = rod_strains_from_local_qs(
            q_s, l0_prev, l0_next,
            d1_prev, d2_prev, d1_next, d2_next,
            eps=eps
        )  # (B,S,4)

        # use only stretching and bending strains to compute energy
        # strains_to_use = strains[:, :, [0,1,2]]  # (B,S,3)

        # energy + grad wrt q_s
        E_s = self.energy_model(strains)  # (B,S,1) or (B,S)
        (dE_dqs,) = torch.autograd.grad(E_s.sum(), q_s, create_graph=True)
        f_s = -dE_dqs  # (B,S,11)

        # scatter-add into global force
        f_full = torch.zeros_like(q_flat)

        def xyz_idx(node_ids):
            base = (3 * node_ids).unsqueeze(-1)  # (S,1)
            off = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
            return base + off

        idx_x_i = xyz_idx(i)
        idx_x_j = xyz_idx(j)
        idx_x_k = xyz_idx(k)
        idx_th_prev = (theta_offset + e_prev).unsqueeze(-1)
        idx_th_next = (theta_offset + e_next).unsqueeze(-1)

        idx11 = torch.cat([idx_x_i, idx_x_j, idx_x_k, idx_th_prev, idx_th_next], dim=-1)  # (S,11)

        idx_flat = idx11.reshape(1, S * 11).expand(Btot, S * 11)
        src_flat = f_s.reshape(Btot, S * 11)
        f_full.scatter_add_(dim=1, index=idx_flat, src=src_flat)

        return f_full.reshape(*batch_shape, ndof)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (..., 2*ndof)
        Returns dx/dt with same shape.
        """
        ndof = self.ndof
        q = x[..., :ndof]        # (..., ndof)
        v = x[..., ndof:]        # (..., ndof)

        # Forces
        f_el = self.elastic_force(q)          # (..., ndof)
        f_damp = torch.matmul(v, self.C.T)    # (..., ndof)

        # External + elastic + damping
        rhs = -f_damp + f_el + self.f_ext     # (..., ndof)

        # Free-DOF solve
        idx = self.free_idx
        rhs_f = rhs.index_select(-1, idx)     # (..., n_free)
        v_f   = v.index_select(-1, idx)       # (..., n_free)

        M_ff = self.M_ff
        a_f = torch.linalg.solve(M_ff, rhs_f.unsqueeze(-1)).squeeze(-1)  # (..., n_free)

        # Scatter back to full
        a_full = torch.zeros_like(v)
        v_full = torch.zeros_like(v)
        a_full[..., idx] = a_f
        v_full[..., idx] = v_f

        dxdt = torch.cat([v_full, a_full], dim=-1)

        # Logging
        self.call_count += 1
        if self.call_count % 1000 == 0:
            ke = 0.5 * torch.sum(v_full**2)
            print(f"[{self.call_count}] t={t.item():.3f}, KE={ke.item():.3e}")

        return dxdt

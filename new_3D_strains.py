import torch

# -------------------------
# Vectorized helpers (Torch)
# -------------------------

def _safe_norm(v, dim=-1, keepdim=False, eps=1e-12):
    return torch.linalg.norm(v, dim=dim, keepdim=keepdim).clamp_min(eps)

def _material_directors(d1, d2, theta):
    c = torch.cos(theta)[..., None]
    s = torch.sin(theta)[..., None]
    m1 = c * d1 + s * d2
    m2 = -s * d1 + c * d2
    return m1, m2

def _parallel_transport(u, t1, t2, eps=1e-12):
    b = torch.cross(t1, t2, dim=-1)
    bnorm = _safe_norm(b, dim=-1, keepdim=True, eps=eps)
    parallel = (bnorm <= (10.0 * eps))

    b = b / bnorm
    # stabilizing steps (same spirit as your numpy)
    b = b - (b * t1).sum(dim=-1, keepdim=True) * t1
    b = b / _safe_norm(b, dim=-1, keepdim=True, eps=eps)
    b = b - (b * t2).sum(dim=-1, keepdim=True) * t2
    b = b / _safe_norm(b, dim=-1, keepdim=True, eps=eps)

    n1 = torch.cross(t1, b, dim=-1)
    n2 = torch.cross(t2, b, dim=-1)

    d = (u * t1).sum(dim=-1, keepdim=True) * t2 \
        + (u * n1).sum(dim=-1, keepdim=True) * n2 \
        + (u * b ).sum(dim=-1, keepdim=True) * b

    return torch.where(parallel, u, d)

def _signed_angle(u, v, normal, eps=1e-12):
    # project to plane orthogonal to normal
    u = u - (u * normal).sum(dim=-1, keepdim=True) * normal
    v = v - (v * normal).sum(dim=-1, keepdim=True) * normal

    u = u / _safe_norm(u, dim=-1, keepdim=True, eps=eps)
    v = v / _safe_norm(v, dim=-1, keepdim=True, eps=eps)

    cosang = (u * v).sum(dim=-1).clamp(-1.0, 1.0)
    sinang = (torch.cross(u, v, dim=-1) * normal).sum(dim=-1)
    return torch.atan2(sinang, cosang)


# ----------------------------------------------------------
# Strains for GENERAL ROD NETWORK using bend_twist_springs
# ----------------------------------------------------------

def rod_strains_vectorized(
    q: torch.Tensor,
    n_nodes: int,
    n_edges: int,
    edges: torch.Tensor,                 # (E,2)
    bend_twist_springs: torch.Tensor,    # (S,5) rows (i,j,k,e_prev,e_next)
    l0_edges,                            # float or (E,)
    d1_edges: torch.Tensor,              # (E,3)
    d2_edges: torch.Tensor,              # (E,3)
    eps: float = 1e-12,
):
    """
    Global q layout:
      q = [... nodal xyz (3*n_nodes) ..., ... edge thetas (n_edges) ...]

    Returns:
      strains: (..., S, 4) with columns [stretch, kappa1, kappa2, twist]
        - stretch is node-based at the middle node j:
            0.5*(stretch(e_prev) + stretch(e_next))
        - (kappa1,kappa2) are curvature components using material directors
        - twist is signed angle between transported m1e and m1f around t_next
    """
    device = q.device
    batch_shape = q.shape[:-1]
    ndof = q.shape[-1]

    edges = edges.to(device).long()
    springs = bend_twist_springs.to(device).long()
    S = springs.shape[0]

    nodal_ndof = 3 * n_nodes
    theta_offset = nodal_ndof

    # flatten batch
    Btot = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) else 1
    q_flat = q.reshape(Btot, ndof)

    # positions and edge thetas
    x = q_flat[:, :nodal_ndof].reshape(Btot, n_nodes, 3)                      # (B,N,3)
    theta = q_flat[:, theta_offset:theta_offset + n_edges].reshape(Btot, n_edges)  # (B,E)

    # ---------- compute edge tangents + edge stretch for ALL edges ----------
    a = edges[:, 0]  # (E,)
    b = edges[:, 1]  # (E,)
    xa = x[:, a, :]  # (B,E,3)
    xb = x[:, b, :]  # (B,E,3)

    ee = xb - xa
    le = _safe_norm(ee, dim=-1, keepdim=False, eps=eps)   # (B,E)
    te = ee / le[..., None]                                # (B,E,3)

    l0 = torch.as_tensor(l0_edges, device=device, dtype=le.dtype)
    if l0.dim() == 0:
        l0 = l0.expand(n_edges)
    l0 = l0.view(1, n_edges)  # (1,E)
    stretch_edge = le / (l0 + eps) - 1.0                   # (B,E)

    # ---------- material directors on ALL edges ----------
    d1 = torch.as_tensor(d1_edges, device=device, dtype=le.dtype).view(1, n_edges, 3).expand(Btot, -1, -1)
    d2 = torch.as_tensor(d2_edges, device=device, dtype=le.dtype).view(1, n_edges, 3).expand(Btot, -1, -1)
    m1, m2 = _material_directors(d1, d2, theta)            # (B,E,3)

    # ---------- gather spring data ----------
    i = springs[:, 0]
    j = springs[:, 1]
    k = springs[:, 2]
    e_prev = springs[:, 3]
    e_next = springs[:, 4]

    # edge tangents + frames at the two edges incident to the turning node
    t_prev = te[:, e_prev, :]  # (B,S,3)
    t_next = te[:, e_next, :]  # (B,S,3)

    m1e = m1[:, e_prev, :]
    m2e = m2[:, e_prev, :]
    m1f = m1[:, e_next, :]
    m2f = m2[:, e_next, :]

    # stretch at middle node j: average of stretches on the two incident edges
    stretch_s = 0.5 * (stretch_edge[:, e_prev] + stretch_edge[:, e_next])  # (B,S)

    # curvature binormal
    chi = (1.0 + (t_prev * t_next).sum(dim=-1, keepdim=True)).clamp_min(eps)
    kb = 2.0 * torch.cross(t_prev, t_next, dim=-1) / chi                   # (B,S,3)

    kappa1 = 0.5 * (kb * (m2e + m2f)).sum(dim=-1)                          # (B,S)
    kappa2 = -0.5 * (kb * (m1e + m1f)).sum(dim=-1)                         # (B,S)

    # twist
    m1e_pt = _parallel_transport(m1e, t_prev, t_next, eps=eps)
    m1e_pt = m1e_pt - (m1e_pt * t_next).sum(dim=-1, keepdim=True) * t_next
    m1e_pt = m1e_pt / _safe_norm(m1e_pt, dim=-1, keepdim=True, eps=eps)

    twist = _signed_angle(m1e_pt, m1f, t_next, eps=eps)                    # (B,S)

    strains = torch.stack([stretch_s, kappa1, kappa2, twist], dim=-1)       # (B,S,4)
    return strains.reshape(*batch_shape, S, 4)


def rod_strains_from_local_qs(
    q_s: torch.Tensor,          # (B,S,11) = [xi(3), xj(3), xk(3), theta_prev, theta_next]
    l0_prev: torch.Tensor,      # (S,) or (1,S) or (B,S)
    l0_next: torch.Tensor,      # (S,) or (1,S) or (B,S)
    d1_prev: torch.Tensor,      # (S,3) or (1,S,3) or (B,S,3)
    d2_prev: torch.Tensor,      # same
    d1_next: torch.Tensor,      # same
    d2_next: torch.Tensor,      # same
    eps: float = 1e-12,
):
    """
    Returns (B,S,4): [stretch, kappa1, kappa2, twist]
    """
    # unpack local dofs
    xi = q_s[..., 0:3]
    xj = q_s[..., 3:6]
    xk = q_s[..., 6:9]
    th_prev = q_s[..., 9]
    th_next = q_s[..., 10]

    # edge vectors and tangents
    e_prev = xj - xi
    e_next = xk - xj
    l_prev = torch.linalg.norm(e_prev, dim=-1).clamp_min(eps)
    l_next = torch.linalg.norm(e_next, dim=-1).clamp_min(eps)
    t_prev = e_prev / l_prev[..., None]
    t_next = e_next / l_next[..., None]

    # stretch (node-based) = avg(edge stretches)
    stretch_prev = l_prev / (l0_prev + eps) - 1.0
    stretch_next = l_next / (l0_next + eps) - 1.0
    stretch = 0.5 * (stretch_prev + stretch_next)

    # material directors on each edge
    c0 = torch.cos(th_prev)[..., None]
    s0 = torch.sin(th_prev)[..., None]
    m1e = c0 * d1_prev + s0 * d2_prev
    m2e = -s0 * d1_prev + c0 * d2_prev

    c1 = torch.cos(th_next)[..., None]
    s1 = torch.sin(th_next)[..., None]
    m1f = c1 * d1_next + s1 * d2_next
    m2f = -s1 * d1_next + c1 * d2_next

    # curvature binormal
    chi = (1.0 + (t_prev * t_next).sum(dim=-1, keepdim=True)).clamp_min(eps)
    kb = 2.0 * torch.cross(t_prev, t_next, dim=-1) / chi  # (B,S,3)

    kappa1 = 0.5 * (kb * (m2e + m2f)).sum(dim=-1)
    kappa2 = -0.5 * (kb * (m1e + m1f)).sum(dim=-1)

    # twist: signed angle between transported m1e and m1f around t_next
    m1e_pt = _parallel_transport(m1e, t_prev, t_next, eps=eps)
    m1e_pt = m1e_pt - (m1e_pt * t_next).sum(dim=-1, keepdim=True) * t_next
    m1e_pt = m1e_pt / torch.linalg.norm(m1e_pt, dim=-1, keepdim=True).clamp_min(eps)

    twist = _signed_angle(m1e_pt, m1f, t_next, eps=eps)

    return torch.stack([stretch, kappa1, kappa2, twist], dim=-1)

# -------------------------

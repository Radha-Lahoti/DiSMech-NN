import torch
from typing import Tuple

# Small threshold used only to "snap" extremely tiny tangent components to 0
_TANGENT_THRESHOLD = 1e-10


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """
    Differentiable norm that avoids NaN gradients at x=0 by computing sqrt(sum(x^2) + eps).
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    Safe normalization: x / sqrt(sum(x^2) + eps)
    """
    return x / safe_norm(x, dim=dim, keepdim=True, eps=eps)


def _ensure_edges_torch(edges, device, *, dtype=torch.long) -> torch.Tensor:
    if isinstance(edges, torch.Tensor):
        return edges.to(device=device, dtype=dtype)
    return torch.as_tensor(edges, device=device, dtype=dtype)


def rotate_axis_angle(v: torch.Tensor, axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    v:    (..., 3)
    axis: (..., 3)  (assumed unit or near-unit; works either way)
    theta:(...,) or (..., 1)
    """
    if theta.ndim == v.ndim - 1:
        theta = theta.unsqueeze(-1)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    dot_av = torch.sum(axis * v, dim=-1, keepdim=True)

    # Rodrigues rotation formula
    return cos_t * v + sin_t * torch.cross(axis, v, dim=-1) + (1.0 - cos_t) * dot_av * axis


def signed_angle(u: torch.Tensor, v: torch.Tensor, n: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Stable signed angle from u to v about normal n.

    u, v, n: (..., 3)
    returns: (...) signed angle in radians

    This avoids:
      - norm(cross(u,v)) gradients at 0
      - sign(...) nondifferentiability
      - atan2(0,0) instability

    by computing signed sin/cos and normalizing the (sin, cos) pair.
    """
    w = torch.cross(u, v, dim=-1)                  # (..., 3)
    sin = torch.sum(n * w, dim=-1)                 # (...) signed sine
    cos = torch.sum(u * v, dim=-1)                 # (...) cosine

    denom = sin * sin + cos * cos                  # (...)
    inv = torch.rsqrt(denom + eps)                 # (...)

    sin_n = sin * inv
    cos_n = cos * inv

    return torch.atan2(sin_n, cos_n)


def parallel_transport(u: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Parallel transport vector u from tangent t_start to tangent t_end.

    u, t_start, t_end: (..., 3)
    returns: (..., 3)

    Uses safe_norm to avoid NaN gradients when cross products vanish.
    Also avoids dividing on degenerate entries via boolean indexing.
    """
    b = torch.cross(t_start, t_end, dim=-1)                    # (..., 3)
    b_norm = safe_norm(b, dim=-1, keepdim=True, eps=1e-12)     # (..., 1)
    mask = (b_norm.squeeze(-1) < eps)                          # (...)

    # Avoid division on degenerate entries
    b_hat = torch.zeros_like(b)
    b_hat[~mask] = b[~mask] / b_norm[~mask]

    # orthogonalize against t_start
    dot_bt = torch.sum(b_hat * t_start, dim=-1, keepdim=True)  # (..., 1)
    b_ortho = b_hat - dot_bt * t_start

    b_ortho_norm = safe_norm(b_ortho, dim=-1, keepdim=True, eps=1e-12)  # (..., 1)
    mask2 = mask | (b_ortho_norm.squeeze(-1) < eps)

    b_ortho_hat = torch.zeros_like(b_ortho)
    b_ortho_hat[~mask2] = b_ortho[~mask2] / b_ortho_norm[~mask2]

    n1 = torch.cross(t_start, b_ortho_hat, dim=-1)
    n2 = torch.cross(t_end,   b_ortho_hat, dim=-1)

    comp = (
        torch.sum(u * t_start, dim=-1, keepdim=True) * t_end +
        torch.sum(u * n1,      dim=-1, keepdim=True) * n2 +
        torch.sum(u * b_ortho_hat, dim=-1, keepdim=True) * b_ortho_hat
    )

    return torch.where(mask.unsqueeze(-1), u, comp)


def compute_tangent(q: torch.Tensor, edges) -> torch.Tensor:
    """
    q: (..., 3*num_nodes) or (..., num_nodes, 3)
    edges: (E, 2) indices
    returns: tangents (..., E, 3)
    """
    if q.ndim >= 2 and q.shape[-1] == 3:
        X = q
    else:
        X = q.reshape(*q.shape[:-1], -1, 3)  # (..., num_nodes, 3)

    edges_t = _ensure_edges_torch(edges, device=X.device, dtype=torch.long)
    n0 = edges_t[:, 0]
    n1 = edges_t[:, 1]

    pos0 = X.index_select(dim=-2, index=n0)  # (..., E, 3)
    pos1 = X.index_select(dim=-2, index=n1)  # (..., E, 3)

    vecs = pos1 - pos0
    tangent = safe_normalize(vecs, dim=-1, eps=1e-12)

    # Optional: snap tiny components to exactly zero (kept from your original)
    tangent = torch.where(tangent.abs() < _TANGENT_THRESHOLD, torch.zeros_like(tangent), tangent)
    return tangent


def compute_time_parallel(a1_old: torch.Tensor, q0: torch.Tensor, q: torch.Tensor, edges) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    a1_old: (..., E, 3) typically per-edge director (must match tangent shape)
    q0, q:  (..., 3*num_nodes) or (..., num_nodes, 3)
    returns: (a1, a2) with shape (..., E, 3)
    """
    tangent0 = compute_tangent(q0, edges)  # (..., E, 3)
    tangent  = compute_tangent(q,  edges)  # (..., E, 3)

    a1_transported = parallel_transport(a1_old, tangent0, tangent)

    # Orthonormalize: remove component along tangent, then normalize safely
    t_dot = torch.sum(a1_transported * tangent, dim=-1, keepdim=True)
    a1 = a1_transported - tangent * t_dot
    a1 = safe_normalize(a1, dim=-1, eps=1e-12)

    a2 = torch.cross(tangent, a1, dim=-1)
    return a1, a2


def compute_material_directors(theta: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    theta: (...,) or (..., 1)
    a1,a2: (..., 3)
    """
    if theta.ndim == a1.ndim - 1:
        theta = theta.unsqueeze(-1)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    m1 = cos_t * a1 + sin_t * a2
    m2 = -sin_t * a1 + cos_t * a2
    return m1, m2


def compute_reference_twist(edges, a1: torch.Tensor, tangent: torch.Tensor, ref_twist: torch.Tensor = None) -> torch.Tensor:
    """
    a1, tangent: (..., n_edges, 3)
    returns: (..., n_edges-1)
    """
    t0 = tangent[..., :-1, :]
    t1 = tangent[..., 1:, :]
    u0 = a1[..., :-1, :]
    u1 = a1[..., 1:, :]

    if ref_twist is None:
        ref_twist = torch.zeros(
            t0.shape[:-1],
            device=a1.device,
            dtype=a1.dtype
        )

    ut = parallel_transport(u0, t0, t1)
    ut = rotate_axis_angle(ut, t1, ref_twist)
    angles = signed_angle(ut, u1, t1)

    return ref_twist + angles

import torch

def curvature_y_bergou(node0, node1, node2, eps=1e-12):
    """
    node0,node1,node2: (..., 3)
    returns: (...,)  y-component of discrete curvature binormal
    """
    ee = node1 - node0
    ef = node2 - node1

    ne = torch.linalg.norm(ee, dim=-1, keepdim=True) + eps
    nf = torch.linalg.norm(ef, dim=-1, keepdim=True) + eps

    te = ee / ne
    tf = ef / nf

    chi = 1.0 + torch.sum(te * tf, dim=-1, keepdim=True) + eps
    kb = 2.0 * torch.cross(te, tf, dim=-1) / chi
    return kb[..., 1]


def longitudinal_strain_node_avg(node0, node1, node2, l_eff, eps=1e-12):
    """
    node0,node1,node2: (..., 3)
    l_eff: scalar float or tensor broadcastable to (...)
    returns: (...,) average axial strain of edges (0-1) and (1-2)
    """
    l_eff = torch.as_tensor(l_eff, device=node0.device, dtype=node0.dtype)
    s01 = torch.linalg.norm(node1 - node0, dim=-1) / (l_eff + eps) - 1.0
    s12 = torch.linalg.norm(node2 - node1, dim=-1) / (l_eff + eps) - 1.0
    return 0.5 * (s01 + s12)


def q_to_x_nodal(q: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """
    q: (3*n_nodes + n_edges, 1) or (3*n_nodes + n_edges,)
       First 3*n_nodes entries are nodal positions (stacked xyz per node).
    Returns:
       x: (n_nodes, 3)
    """
    q = q.view(-1)                       # (3*n_nodes + n_edges,)
    q_nodal = q[: 3 * n_nodes]           # (3*n_nodes,)
    x = q_nodal.view(n_nodes, 3)         # (n_nodes, 3)
    return x


def edge_stretch_from_q(q: torch.Tensor, n_nodes: int, l0_edges, eps=1e-12) -> torch.Tensor:
    """
    Vectorized stretch for all chain edges (i,i+1), i=0..n_nodes-2.

    l0_edges: float or (n_nodes-1,) tensor of rest lengths
    Returns: (n_nodes-1,) tensor
    """
    x = q_to_x_nodal(q, n_nodes)         # (N,3)

    x_i = x[:-1, :]                      # (E,3)
    x_j = x[1:, :]                       # (E,3)

    diff = x_j - x_i
    length = torch.linalg.norm(diff, dim=-1)  # (E,)

    l0 = torch.as_tensor(l0_edges, device=q.device, dtype=length.dtype)
    if l0.dim() == 0:
        l0 = l0.expand(n_nodes - 1)      # (E,)

    return length / (l0 + eps) - 1.0     # (E,)


def node_axial_stretch_from_q(q: torch.Tensor, n_nodes: int, l0_edges, eps=1e-12) -> torch.Tensor:
    """
    Axial stretch per node for a chain:
      node 0:     0.5 * stretch(edge 0)
      node N-1:   0.5 * stretch(edge N-2)
      internal i: 0.5*(stretch(edge i-1) + stretch(edge i))

    Returns: (n_nodes,) tensor
    """
    s_edge = edge_stretch_from_q(q, n_nodes, l0_edges, eps=eps)  # (E,) where E=N-1

    out = torch.empty(n_nodes, device=q.device, dtype=s_edge.dtype)
    out[0] = 0.5 * s_edge[0]
    out[-1] = 0.5 * s_edge[-1]
    out[1:-1] = 0.5 * (s_edge[:-1] + s_edge[1:])
    return out


def node_curvature_y_from_q(q: torch.Tensor, n_nodes: int, eps=1e-12) -> torch.Tensor:
    """
    Bergou discrete curvature binormal y-component for each triple (i-1,i,i+1).
    Defined for i=1..N-2.

    Returns: (n_nodes-2,) tensor corresponding to nodes 1..N-2
    """
    x = q_to_x_nodal(q, n_nodes)         # (N,3)

    x0 = x[:-2, :]                       # (T,3) where T=N-2
    x1 = x[1:-1, :]
    x2 = x[2:, :]

    ee = x1 - x0
    ef = x2 - x1

    ne = torch.linalg.norm(ee, dim=-1, keepdim=True) + eps
    nf = torch.linalg.norm(ef, dim=-1, keepdim=True) + eps

    te = ee / ne
    tf = ef / nf

    chi = 1.0 + torch.sum(te * tf, dim=-1, keepdim=True) + eps
    kb = 2.0 * torch.cross(te, tf, dim=-1) / chi        # (T,3)

    return kb[:, 1]                                     # (T,)

import torch

def get_strain_stretch_edge2D_torch(nodeA, nodeB, l0=1.0, eps=1e-12):
    """
    nodeA, nodeB: (..., D) torch tensors (D = 2 or 3).
    l0: reference length (float)
    """
    diff = nodeB - nodeA                       # (..., D)
    length = torch.linalg.norm(diff, dim=-1)   # (...,)
    return length / (l0 + eps) - 1.0


def get_strain_stretch2D_torch(
    node0,
    node1,
    node2,
    l_0=1.0,
    l_1=1.0,
    is_boundary=None,   # bool tensor broadcastable to (...,)
    eps=1e-12,
):
    """
    node0, node1, node2: (..., D) tensors
      - For boundary stencils, you may pass any dummy node2; it will be ignored.
    l_0, l_1: float or tensor broadcastable to batch shape (...,)
    is_boundary: bool tensor broadcastable to (...,)
      - True  => boundary formula: 0.5*stretch_first
      - False => interior formula: 0.5*(stretch_first+stretch_second)
    """
    stretch_first = get_strain_stretch_edge2D_torch(node0, node1, l_0, eps)

    # If caller doesn't provide, assume all interior (no branching)
    if is_boundary is None:
        stretch_second = get_strain_stretch_edge2D_torch(node1, node2, l_1, eps)
        return 0.5 * (stretch_first + stretch_second)

    stretch_second = get_strain_stretch_edge2D_torch(node1, node2, l_1, eps)

    # internal branching happens *inside* this function
    return torch.where(is_boundary, 0.5 * stretch_first, 0.5 * (stretch_first + stretch_second))



def get_strain_curvature_3D_torch(node0, node1, node2, m1e, m2e, m1f, m2f, eps=1e-12):
    """
    node0, node1, node2: (..., 3) torch tensors
    Computes curvature using Bergou's discrete curvature binormal formula.
    Returns scalar curvature kappa_y = (kb)[..., 1]
    """

    ee = node1 - node0                         # (..., 3)
    ef = node2 - node1                         # (..., 3)

    ne = torch.linalg.norm(ee, dim=-1, keepdim=True) + eps
    nf = torch.linalg.norm(ef, dim=-1, keepdim=True) + eps

    te = ee / ne                               # (..., 3)
    tf = ef / nf                               # (..., 3)

    chi = 1.0 + torch.sum(te * tf, dim=-1, keepdim=True) + eps

    kb = 2.0 * torch.cross(te, tf, dim=-1) / chi    # (..., 3)

    kappa1 = 0.5 * torch.einsum('...i,...i->...', kb, m2e + m2f)
    kappa2 = -0.5 * torch.einsum('...i,...i->...', kb, m1e + m1f)

    return kappa1, kappa2                          # (...)

def get_strain_twist_3D_torch(theta0, theta1, reftwist):
    """
    Returns scalar twist strain tau (...)
    """
    tau = reftwist + theta1 - theta0                 # (...)

    return tau
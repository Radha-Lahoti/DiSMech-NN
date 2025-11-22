import torch

def get_strain_stretch_edge2D_torch(nodeA, nodeB, l0, eps=1e-12):
    """
    nodeA, nodeB: (..., D) torch tensors (D = 2 or 3).
    l0: reference length (float)
    """
    diff = nodeB - nodeA                       # (..., D)
    length = torch.linalg.norm(diff, dim=-1)   # (...,)
    return length / (l0 + eps) - 1.0


def get_strain_stretch2D_torch(node0, node1, node2=None, l_0=1.0, l_1=1.0, eps=1e-12):
    """
    node0, node1, node2: (..., D) torch tensors
    l_0, l_1: floats (reference lengths)

    Returns: axial stretch at node1
    """
    # Stretch of first edge (node0 -> node1)
    stretch_first = get_strain_stretch_edge2D_torch(node0, node1, l_0, eps)

    if node2 is None:
        # Boundary node case: only half of first-edge stretch
        return 0.5 * stretch_first

    # Stretch of second edge (node1 -> node2)
    stretch_second = get_strain_stretch_edge2D_torch(node1, node2, l_1, eps)

    # Internal node: average of the two
    return 0.5 * (stretch_first + stretch_second)


def get_strain_curvature_3D_torch(node0, node1, node2, eps=1e-12):
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

    # We only want bending in x–z plane -> y-component curvature
    return kb[..., 1]                           # (...,)

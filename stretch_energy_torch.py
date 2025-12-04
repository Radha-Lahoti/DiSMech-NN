import torch

# ------------------------------
# Edge stretch (2 nodes, 3D coords)
# ------------------------------

def get_strain_stretch_edge2D_torch(node0, node1, l_k):
    """
    Compute axial stretch of an edge connecting node0 and node1.

    node0, node1: (3,) tensors
    l_k: scalar or 0D tensor (reference length)
    """
    device = node0.device
    dtype = node0.dtype

    l_k = torch.as_tensor(l_k, device=device, dtype=dtype)

    edge = node1 - node0               # (3,)
    edgeLen = torch.linalg.norm(edge)  # scalar
    epsX = edgeLen / l_k - 1.0
    return epsX


def grad_and_hess_strain_stretch_edge2D_torch(node0, node1, l_k):
    """
    Gradient and Hessian of axial stretch wrt the 6 DOFs:
    [x0,y0,z0, x1,y1,z1].

    Returns:
      dF: (6,) tensor
      dJ: (6,6) tensor
    """
    device = node0.device
    dtype = node0.dtype

    l_k = torch.as_tensor(l_k, device=device, dtype=dtype)

    edge = node1 - node0                       # (3,)
    edgeLen = torch.linalg.norm(edge)          # scalar
    tangent = edge / edgeLen                   # (3,)
    epsX = get_strain_stretch_edge2D_torch(node0, node1, l_k)

    # gradient of stretch wrt edge vector
    dF_unit = tangent / l_k                    # (3,)

    dF = torch.zeros(6, device=device, dtype=dtype)
    dF[0:3] = -dF_unit
    dF[3:6] =  dF_unit

    # Hessian of square(stretch) wrt edge vector
    Id3 = torch.eye(3, device=device, dtype=dtype)
    edge_outer = torch.outer(edge, edge)       # (3,3)

    M = 2.0 / l_k * (
        (1.0 / l_k - 1.0 / edgeLen) * Id3
        + (1.0 / edgeLen) * edge_outer / (edgeLen ** 2)
    )

    # Convert to Hessian of stretch wrt edge vector
    if torch.abs(epsX) < 1e-14:  # Edge case
        M2 = torch.zeros_like(M)
    else:
        M2 = 1.0 / (2.0 * epsX) * (M - 2.0 * torch.outer(dF_unit, dF_unit))

    dJ = torch.zeros((6, 6), device=device, dtype=dtype)
    dJ[0:3, 0:3] =  M2
    dJ[3:6, 3:6] =  M2
    dJ[0:3, 3:6] = -M2
    dJ[3:6, 0:3] = -M2

    return dF, dJ


# ------------------------------
# Node stretch (3 nodes, 3D coords)
# ------------------------------

def get_strain_stretch2D_torch(node0, node1, node2=None, l_0=1.0, l_1=1.0):
    """
    Axial stretch at node1 from the two incident edges.
    node0, node1, node2: (3,) tensors (3D coords)
    l_0, l_1: reference edge lengths
    """
    device = node0.device
    dtype = node0.dtype
    l_0 = torch.as_tensor(l_0, device=device, dtype=dtype)
    l_1 = torch.as_tensor(l_1, device=device, dtype=dtype)

    if node2 is not None:
        stretch_first  = get_strain_stretch_edge2D_torch(node0, node1, l_0)
        stretch_second = get_strain_stretch_edge2D_torch(node1, node2, l_1)
        epsilon_1 = 0.5 * stretch_first + 0.5 * stretch_second
    else:
        epsilon_1 = 0.5 * get_strain_stretch_edge2D_torch(node0, node1, l_0)

    return epsilon_1


def grad_and_hess_strain_stretch2D_torch(node0, node1, node2=None, l_0=1.0, l_1=1.0):
    """
    Gradient and Hessian of axial stretch at node1 wrt DOFs:

      • If node2 is not None: 9 DOFs [x0,y0,z0, x1,y1,z1, x2,y2,z2]
      • If node2 is None:      6 DOFs [x0,y0,z0, x1,y1,z1]

    Returns:
      G_varepsilon: (9,) or (6,) tensor
      H_varepsilon: (9,9) or (6,6) tensor
    """
    device = node0.device
    dtype = node0.dtype

    l_0 = torch.as_tensor(l_0, device=device, dtype=dtype)
    l_1 = torch.as_tensor(l_1, device=device, dtype=dtype)

    if node2 is not None:
        G_varepsilon = torch.zeros(9, device=device, dtype=dtype)
        H_varepsilon = torch.zeros((9, 9), device=device, dtype=dtype)

        G1, H1 = grad_and_hess_strain_stretch_edge2D_torch(node0, node1, l_0)
        G2, H2 = grad_and_hess_strain_stretch_edge2D_torch(node1, node2, l_1)

        # Embed edge gradients into 9-DOF structure
        G_varepsilon[0:6] += 0.5 * G1            # nodes 0 & 1
        G_varepsilon[3:9] += 0.5 * G2            # nodes 1 & 2

        H_varepsilon[0:6, 0:6] += 0.5 * H1
        H_varepsilon[3:9, 3:9] += 0.5 * H2
    else:
        G1, H1 = grad_and_hess_strain_stretch_edge2D_torch(node0, node1, l_0)
        G_varepsilon = 0.5 * G1
        H_varepsilon = 0.5 * H1

    return G_varepsilon, H_varepsilon


# ------------------------------
# Stretch energy (linear elastic)
# ------------------------------

def get_energy_stretch_linear_elastic_torch(node0, node1, node2=None, l_0=1.0, l_1=None, EA=None):
    """
    E_s = 0.5 * EA * epsilon^2 * l_1   (same as your NumPy version)
    """
    if EA is None or l_1 is None:
        raise ValueError("EA and l_1 must be provided (mirrors NumPy behavior).")

    device = node0.device
    dtype = node0.dtype

    l_0 = torch.as_tensor(l_0, device=device, dtype=dtype)
    l_1 = torch.as_tensor(l_1, device=device, dtype=dtype)
    EA  = torch.as_tensor(EA,  device=device, dtype=dtype)

    strain_stretch = get_strain_stretch2D_torch(node0, node1, node2, l_0, l_1)
    E_s = 0.5 * EA * strain_stretch**2.0 * l_1
    return E_s


def grad_and_hess_energy_stretch_linear_elastic_torch(node0, node1, node2=None, l_0=1.0, l_1=None, EA=None):
    """
    Gradient & Hessian of stretch energy wrt nodal DOFs.
    Shapes:
      • If internal node (node2 not None): G: (9,), H: (9,9)
      • If boundary (node2 is None):       G: (6,), H: (6,6)
    """
    if EA is None:
        raise ValueError("EA must be provided.")

    device = node0.device
    dtype = node0.dtype

    l_0 = torch.as_tensor(l_0, device=device, dtype=dtype)
    EA  = torch.as_tensor(EA,  device=device, dtype=dtype)

    strain_stretch = get_strain_stretch2D_torch(node0, node1, node2, l_0, l_1 if l_1 is not None else l_0)
    G_strain, H_strain = grad_and_hess_strain_stretch2D_torch(node0, node1, node2, l_0, l_1 if l_1 is not None else l_0)

    if l_1 is None:
        l_eff = l_0
    else:
        l_1 = torch.as_tensor(l_1, device=device, dtype=dtype)
        l_eff = 0.5 * (l_0 + l_1)

    gradE_strain = EA * strain_stretch * l_eff
    hessE_strain = EA * l_eff

    # Outer product: (n,) x (n,) → (n,n)
    G = gradE_strain * G_strain
    H = gradE_strain * H_strain + hessE_strain * (G_strain.unsqueeze(-1) @ G_strain.unsqueeze(0))

    return G, H


# ------------------------------
# Global stretch force assembly
# ------------------------------

def get_stretch_force_torch(q, stretch_springs, EA):
    """
    Global stretch force assembly.

    q: (3N,) tensor – global DOF vector [x0,y0,z0, x1,y1,z1, ...]
    stretch_springs: list of (node0_idx, node1_idx, node2_idx, l_0, l_1)
                     node2_idx can be None for boundary springs.
    EA: axial stiffness
    """
    device = q.device
    dtype = q.dtype
    EA = torch.as_tensor(EA, device=device, dtype=dtype)

    num_dofs = q.shape[0]
    F_stretch = torch.zeros(num_dofs, device=device, dtype=dtype)

    for spring in stretch_springs:
        node0_idx, node1_idx, node2_idx, l_0, l_1 = spring

        node0 = q[3 * node0_idx : 3 * node0_idx + 3]
        node1 = q[3 * node1_idx : 3 * node1_idx + 3]

        if node2_idx is not None:
            node2 = q[3 * node2_idx : 3 * node2_idx + 3]
        else:
            node2 = None

        G_stretch, _ = grad_and_hess_energy_stretch_linear_elastic_torch(
            node0, node1, node2, l_0=l_0, l_1=l_1, EA=EA
        )

        # Apply negative gradient as force
        F_stretch[3 * node0_idx : 3 * node0_idx + 3] -= G_stretch[0:3]
        F_stretch[3 * node1_idx : 3 * node1_idx + 3] -= G_stretch[3:6]
        if node2_idx is not None:
            F_stretch[3 * node2_idx : 3 * node2_idx + 3] -= G_stretch[6:9]

    return F_stretch

import torch

_EPS = 1e-12

# ------------------------------
# Discrete curvature (3D) – PyTorch
# ------------------------------

def get_strain_curvature_3D_torch(node0, node1, node2):
    """
    Curvature at node1 from three 3D nodes, with bending in the x–z plane.
    m2 vectors are along +y, so we use kappa[1] (y-component).

    Inputs:
      node0, node1, node2: (..., 3) or (3,) tensors on same device

    Output:
      kappa1: scalar or tensor (...,) curvature (y-component of curvature binormal)
    """
    n0 = node0
    n1 = node1
    n2 = node2

    # Edges and unit tangents
    ee = n1 - n0
    ef = n2 - n1

    # keepdim=True so we can safely broadcast divide
    ne = torch.linalg.norm(ee, dim=-1, keepdim=True) + _EPS
    nf = torch.linalg.norm(ef, dim=-1, keepdim=True) + _EPS

    te = ee / ne
    tf = ef / nf

    # Curvature binormal (Bergou et al.)
    chi = 1.0 + (te * tf).sum(dim=-1, keepdim=True) + _EPS
    kb = 2.0 * torch.cross(te, tf, dim=-1) / chi  # vector curvature

    # For x–z bending, only y-component should remain nonzero
    kappa1 = kb[..., 1]   # y-component
    return kappa1


def grad_hess_strain_curvature_3D_torch(node0, node1, node2):
    """
    Gradient of kappa1 (y-component of curvature binormal) wrt the 9 DOFs:
    [x0,y0,z0, x1,y1,z1, x2,y2,z2].

    Uses m2 vectors along +y: m2e = m2f = [0,1,0].

    Inputs:
      node0, node1, node2: (3,) tensors

    Returns:
      gradKappa: (9,) tensor
      hessKappa: (9, 9) tensor (zeros, Gauss–Newton approx)
    """
    device = node0.device
    dtype = node0.dtype

    n0 = node0
    n1 = node1
    n2 = node2

    m2e = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    m2f = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    ee = n1 - n0
    ef = n2 - n1

    ne = torch.linalg.norm(ee) + _EPS
    nf = torch.linalg.norm(ef) + _EPS

    te = ee / ne
    tf = ef / nf

    chi = 1.0 + torch.dot(te, tf) + _EPS
    tilde_t  = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi  # stays along y

    # scalar curvature (y-component)
    kappa1 = get_strain_curvature_3D_torch(n0, n1, n2)

    # Vector derivatives wrt edge vectors (same structure as NumPy code)
    Dkappa1De = ( -kappa1 * tilde_t + torch.cross(tf, tilde_d2, dim=-1) ) / ne
    Dkappa1Df = ( -kappa1 * tilde_t - torch.cross(te, tilde_d2, dim=-1) ) / nf

    # Assemble gradient wrt node positions:
    # d/dn0 = -Dkappa1De
    # d/dn1 =  Dkappa1De - Dkappa1Df
    # d/dn2 =  Dkappa1Df
    gradKappa = torch.zeros(9, device=device, dtype=dtype)
    gradKappa[0:3] = -Dkappa1De
    gradKappa[3:6] =  Dkappa1De - Dkappa1Df
    gradKappa[6:9] =  Dkappa1Df

    # Full Hessian is lengthy; return zeros (use Gauss–Newton in energy assembly).
    hessKappa = torch.zeros(9, 9, device=device, dtype=dtype)
    return gradKappa, hessKappa


# ------------------------------
# Bending energy (linear elastic) – PyTorch
# ------------------------------

def get_energy_bending_linear_elastic_torch(node0, node1, node2=None, l_eff=None, EI=None):
    """
    E_b = 0.5 * (EI / l_eff) * kappa^2

    Inputs:
      node0, node1, node2: (3,) tensors
      l_eff, EI: scalars or 0D tensors (same device/dtype)

    Returns:
      scalar tensor (energy)
    """
    if node2 is None:
        return torch.zeros((), device=node0.device, dtype=node0.dtype)
    if l_eff is None or EI is None:
        raise ValueError("l_eff and EI must be provided.")

    # make l_eff, EI tensors on the right device
    device = node0.device
    dtype = node0.dtype
    l_eff = torch.as_tensor(l_eff, device=device, dtype=dtype)
    EI    = torch.as_tensor(EI,    device=device, dtype=dtype)

    kappa = get_strain_curvature_3D_torch(node0, node1, node2)
    return 0.5 * EI / l_eff * (kappa ** 2.0)


def grad_and_hess_energy_bending_linear_elastic_torch(node0, node1, node2=None, l_eff=None, EI=None):
    """
    Returns gradient (9,) and Hessian (9,9). Uses:
      ∇E = (EI/l_eff) * kappa * ∇kappa
      H  = (EI/l_eff) * [ kappa * H_kappa + (∇kappa)(∇kappa)^T ]
    We set H_kappa ≈ 0 (Gauss–Newton approximation). 
    """
    device = node0.device
    dtype = node0.dtype

    if node2 is None:
        return (
            torch.zeros(9, device=device, dtype=dtype),
            torch.zeros((9, 9), device=device, dtype=dtype),
        )
    if l_eff is None or EI is None:
        raise ValueError("l_eff and EI must be provided.")

    l_eff = torch.as_tensor(l_eff, device=device, dtype=dtype)
    EI    = torch.as_tensor(EI,    device=device, dtype=dtype)

    kappa = get_strain_curvature_3D_torch(node0, node1, node2)
    G_kappa, H_kappa = grad_hess_strain_curvature_3D_torch(node0, node1, node2)  # H_kappa = 0 here

    coeff = EI / l_eff
    G = coeff * kappa * G_kappa
    H = coeff * (G_kappa.unsqueeze(-1) @ G_kappa.unsqueeze(0))  # outer product

    return G, H


# ------------------------------
# Global bend force assembly (3D) – PyTorch
# ------------------------------

def get_bend_force_torch(q, bend_springs, EI):
    """
    Assemble global bending forces.

    q: (3N,) global DOF tensor [x0,y0,z0, x1,y1,z1, ...]
       (requires_grad can be True if you want autograd through forces)
    bend_springs: list of tuples (i0, i1, i2, l_eff)
    EI: bending stiffness (scalar or 0D tensor)

    Returns:
      F_bend: (3N,) tensor on same device as q
    """
    device = q.device
    dtype = q.dtype
    EI = torch.as_tensor(EI, device=device, dtype=dtype)

    ndofs = q.shape[0]
    F_bend = torch.zeros_like(q)

    for (i0, i1, i2, leff) in bend_springs:
        n0 = q[3 * i0 : 3 * i0 + 3]
        n1 = q[3 * i1 : 3 * i1 + 3]
        n2 = q[3 * i2 : 3 * i2 + 3]

        G_bend, _ = grad_and_hess_energy_bending_linear_elastic_torch(
            n0, n1, n2, l_eff=leff, EI=EI
        )

        # Negative gradient (force)
        F_bend[3 * i0 : 3 * i0 + 3] -= G_bend[0:3]
        F_bend[3 * i1 : 3 * i1 + 3] -= G_bend[3:6]
        F_bend[3 * i2 : 3 * i2 + 3] -= G_bend[6:9]

    return F_bend

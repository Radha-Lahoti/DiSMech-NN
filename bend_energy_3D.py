import numpy as np

# ------------------------------
# Discrete curvature (3D)
# ------------------------------

_EPS = 1e-12

def get_strain_curvature_3D(node0, node1, node2): # xz plane
    """
    Curvature at node1 from three 3D nodes, with bending in the x–z plane.
    m2 vectors are along +y, so we use kappa[1] (y-component).

    Inputs:
      node0, node1, node2: (3,) arrays

    Output:
      kappa1: scalar curvature (y-component of curvature binormal)
    """
    n0 = np.asarray(node0, dtype=float)
    n1 = np.asarray(node1, dtype=float)
    n2 = np.asarray(node2, dtype=float)

    # Edges and unit tangents
    ee = n1 - n0
    ef = n2 - n1
    ne = np.linalg.norm(ee) + _EPS
    nf = np.linalg.norm(ef) + _EPS
    te = ee / ne
    tf = ef / nf

    # Curvature binormal (Bergou et al.)
    chi = 1.0 + np.dot(te, tf) + _EPS
    kb = 2.0 * np.cross(te, tf) / chi  # vector curvature

    # For x–z bending, only y-component should remain nonzero
    kappa1 = kb[1]
    return kappa1


def grad_hess_strain_curvature_3D(node0, node1, node2): # xz plane
    """
    Gradient of kappa1 (y-component of curvature binormal) wrt the 9 DOFs:
    [x0,y0,z0, x1,y1,z1, x2,y2,z2].

    Uses m2 vectors along +y: m2e = m2f = [0,1,0].
    Returns:
      gradKappa: (9,) array
    """
    n0 = np.asarray(node0, dtype=float)
    n1 = np.asarray(node1, dtype=float)
    n2 = np.asarray(node2, dtype=float)

    m2e = np.array([0.0, 1.0, 0.0])
    m2f = np.array([0.0, 1.0, 0.0])

    ee = n1 - n0
    ef = n2 - n1
    ne = np.linalg.norm(ee) + _EPS
    nf = np.linalg.norm(ef) + _EPS
    te = ee / ne
    tf = ef / nf

    chi = 1.0 + np.dot(te, tf) + _EPS
    tilde_t  = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi  # stays along y

    # scalar curvature (y-component)
    kappa1 = get_strain_curvature_3D(n0, n1, n2)

    # Vector derivatives wrt edge vectors (same structure as your 2D code, now 3D)
    Dkappa1De = ( -kappa1 * tilde_t + np.cross(tf, tilde_d2) ) / ne
    Dkappa1Df = ( -kappa1 * tilde_t - np.cross(te, tilde_d2) ) / nf

    # Assemble gradient wrt node positions:
    # d/dn0 = -Dkappa1De
    # d/dn1 =  Dkappa1De - Dkappa1Df
    # d/dn2 =  Dkappa1Df
    gradKappa = np.zeros(9)
    gradKappa[0:3] = -Dkappa1De
    gradKappa[3:6] =  Dkappa1De - Dkappa1Df
    gradKappa[6:9] =  Dkappa1Df

    # Full Hessian is lengthy; return zeros (use Gauss–Newton in energy assembly).
    hessKappa = np.zeros((9, 9))
    return gradKappa, hessKappa

# ------------------------------
# Bending energy (linear elastic)
# ------------------------------
def get_energy_bending_linear_elastic(node0, node1, node2=None, l_eff=None, EI=None):
    """
    E_b = 0.5 * (EI / l_eff) * kappa^2
    """
    if node2 is None:
        return 0.0
    if l_eff is None or EI is None:
        raise ValueError("l_eff and EI must be provided.")
    kappa = get_strain_curvature_3D(node0, node1, node2)
    return 0.5 * EI / l_eff * (kappa**2.0)


def grad_and_hess_energy_bending_linear_elastic(node0, node1, node2=None, l_eff=None, EI=None):
    """
    Returns gradient (9,) and Hessian (9,9). Uses:
      ∇E = (EI/l_eff) * kappa * ∇kappa
      H  = (EI/l_eff) * [ kappa * H_kappa + (∇kappa)(∇kappa)^T ]
    We set H_kappa ≈ 0 (Gauss–Newton approximation). 
    """
    if node2 is None:
        return np.zeros(9), np.zeros((9, 9))
    if l_eff is None or EI is None:
        raise ValueError("l_eff and EI must be provided.")

    kappa = get_strain_curvature_3D(node0, node1, node2)
    G_kappa, H_kappa = grad_hess_strain_curvature_3D(node0, node1, node2)  # H_kappa = 0 here

    coeff = EI / l_eff
    G = coeff * kappa * G_kappa
    H = coeff * (np.outer(G_kappa, G_kappa))  # Gauss–Newton term

    return G, H


# ------------------------------
# Global bend force assembly (3D)
# ------------------------------
def get_bend_force(q, bend_springs, EI):
    """
    Assemble global bending forces.

    q: (3N,) global DOF vector [x0,y0,z0, x1,y1,z1, ...]
    bend_springs: list of tuples (i0, i1, i2, l_eff)
    EI: bending stiffness
    """
    q = np.asarray(q, dtype=float)
    ndofs = q.shape[0]
    F_bend = np.zeros_like(q)

    for (i0, i1, i2, l_eff) in bend_springs:
        n0 = q[3*i0:3*i0+3]
        n1 = q[3*i1:3*i1+3]
        n2 = q[3*i2:3*i2+3]

        G_bend, _ = grad_and_hess_energy_bending_linear_elastic(n0, n1, n2, l_eff, EI)

        # Negative gradient (force)
        F_bend[3*i0:3*i0+3] -= G_bend[0:3]
        F_bend[3*i1:3*i1+3] -= G_bend[3:6]
        F_bend[3*i2:3*i2+3] -= G_bend[6:9]

    return F_bend

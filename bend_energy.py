import numpy as np

# ------------------------------
# Discrete curvature (3D)
# ------------------------------
def get_strain_curvature3D(node0, node1, node2):
    """
    Discrete curvature scalar at node1 for a polyline (node0 -> node1 -> node2).
    Uses curvature-binormal: kb = 2 (te x tf) / (1 + te·tf), then kappa = ||kb||.
    """
    n0 = np.asarray(node0, dtype=float)
    n1 = np.asarray(node1, dtype=float)
    n2 = np.asarray(node2, dtype=float)

    ee = n1 - n0
    ef = n2 - n1
    le = np.linalg.norm(ee)
    lf = np.linalg.norm(ef)
    if le == 0.0 or lf == 0.0:
        raise ValueError("Zero-length edge encountered in curvature computation.")

    te = ee / le
    tf = ef / lf
    chi = 1.0 + np.dot(te, tf)
    # Near-straight or antiparallel protection
    if chi <= 1e-12:
        # limit case: kb ~ large; you may regularize; here we return 0 and let gradient be 0
        return 0.0

    kb = 2.0 * np.cross(te, tf) / chi
    kappa = np.linalg.norm(kb)
    return kappa


def grad_and_hess_strain_curvature3D(node0, node1, node2):
    """
    Gradient (9,) and Hessian (9,9) of scalar curvature kappa at node1 (3D).
    DOF order: [x0,y0,z0, x1,y1,z1, x2,y2,z2]

    We compute gradient exactly; Hessian is returned as zeros (use Gauss–Newton in energy).
    """
    n0 = np.asarray(node0, dtype=float)
    n1 = np.asarray(node1, dtype=float)
    n2 = np.asarray(node2, dtype=float)

    ee = n1 - n0
    ef = n2 - n1
    le = np.linalg.norm(ee)
    lf = np.linalg.norm(ef)
    if le == 0.0 or lf == 0.0:
        raise ValueError("Zero-length edge encountered in curvature gradient.")

    te = ee / le
    tf = ef / lf
    chi = 1.0 + np.dot(te, tf)
    if chi <= 1e-12:
        # Degenerate; return zeros to avoid NaNs.
        return np.zeros(9), np.zeros((9, 9))

    # Curvature binormal and scalar curvature
    N = np.cross(te, tf)                 # numerator/2
    kb = 2.0 * N / chi
    k = np.linalg.norm(kb)

    if k < 1e-15:
        # Straight configuration: curvature ~ 0; linearization becomes ill-conditioned
        return np.zeros(9), np.zeros((9, 9))

    u = kb / k                           # unit along kb

    # Derivatives of unit tangents wrt edge vectors:
    # d te = T_e de,  T_e = (I - te te^T) / ||ee||
    # d tf = T_f df,  T_f = (I - tf tf^T) / ||ef||
    I = np.eye(3)
    T_e = (I - np.outer(te, te)) / le
    T_f = (I - np.outer(tf, tf)) / lf

    # For any vector v:  (T_e de) × tf  => use adjoint to pull onto de
    # uᵀ[(T_e de) × tf] = (B × u)ᵀ (T_e de) where B=tf
    # Also dchi/de = (T_e de)·tf
    # So gradient wrt ee:  g_e such that δk = g_eᵀ δe
    #   dkb/de =  2/chi * ((T_e de) × tf)  -  2N/chi^2 * ((T_e de)·tf)
    #   δk = uᵀ dkb/de
    Bxu = np.cross(tf, u)                # (tf × u)
    term1_e = (2.0 / chi) * (T_e.T @ Bxu)
    term2_e = (2.0 * np.linalg.norm(N) / (chi**2)) * (T_e.T @ tf)
    g_e = term1_e - term2_e

    # Gradient wrt ef (analogous):
    uxA = np.cross(u, te)                # (u × te)
    term1_f = (2.0 / chi) * (T_f.T @ uxA)
    term2_f = (2.0 * np.linalg.norm(N) / (chi**2)) * (T_f.T @ te)
    g_f = term1_f - term2_f

    # Map edge-vector gradients to node DOFs:
    # ee = n1 - n0, ef = n2 - n1
    # ∂/∂n0 = -g_e
    # ∂/∂n1 = (g_e) - (g_f)
    # ∂/∂n2 =  g_f
    G = np.zeros(9)
    G[0:3] = -g_e
    G[3:6] =  g_e - g_f
    G[6:9] =  g_f

    # Full Hessian is lengthy; return zeros (use Gauss–Newton in energy assembly).
    H = np.zeros((9, 9))
    return G, H


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
    kappa = get_strain_curvature3D(node0, node1, node2)
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

    kappa = get_strain_curvature3D(node0, node1, node2)
    G_kappa, H_kappa = grad_and_hess_strain_curvature3D(node0, node1, node2)  # H_kappa = 0 here

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

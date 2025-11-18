import numpy as np

from utils import crossMat

def get_strain_curvature2D(node0, node1, node2):
  '''
  Compute the curvature (scalar) in 2D at node1 that is connected to node0 and
  node2.

  Inputs:
  node0: 2x1 vector - position of the first node
  node1: 2x1 vector - position of the middle node
  node2: 2x1 vector - position of the last node

  Outputs:
  kappa1: scalar - curvature at node1 (a scalar)

  Description:
  The function converts 2D nodes into 3D; computers the curvature binormal
  (a vector) and extracts the scalar curvature (z-component of curvature
  binormal)
  '''

  # Nodes in 3D
  node0 = np.array([node0[0], node0[1], 0.0])
  node1 = np.array([node1[0], node1[1], 0.0])
  node2 = np.array([node2[0], node2[1], 0.0])

  # Edge vectors
  ee = node1 - node0
  ef = node2 - node1

  # Norms of edge vectors
  norm_e = np.linalg.norm(ee)
  norm_f = np.linalg.norm(ef)

  # Unit tangents
  te = ee / norm_e
  tf = ef / norm_f

  # Curvature binormal
  kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

  # Curvature
  kappa1 = kb[2] # z-component for 2D bending about y-axis
  # kappa1 = kb[1] # y-component for 2D bending about y-axis

  return kappa1

def grad_and_hess_strain_curvature2D(node0, node1, node2):
  '''
  Compute the gradient and hessian of the curvature at node1 with
  respect to the dof vector (6 dofs: x,y coordinates of three nodes).

  Inputs:
  node0: 2x1 vector - position of the first node
  node1: 2x1 vector - position of the middle node
  node2: 2x1 vector - position of the last node

  Outputs:
  dF: 6x1 vector - gradient of curvature at node 1
  dJ: 6x6 vector - hessian of curvature at node 1.

  Description:
  The gradient and hessian at node1 is the average of the gradients and hessians
  of the two edges (one between node0 and node1 and the other one between node1
  and node2).
  '''

  # Nodes in 3D
  node0 = np.array([node0[0], node0[1], 0.0])
  node1 = np.array([node1[0], node1[1], 0.0])
  node2 = np.array([node2[0], node2[1], 0.0])

  # Unit vectors along z-axis
  m2e = np.array([0, 0, 1])
  m2f = np.array([0, 0, 1])

  # # Unit vectors along y-axis
  # m2e = np.array([0, 1, 0])
  # m2f = np.array([0, 1, 0])

  # Initialize gradient of curvature
  gradKappa = np.zeros(6)

  # Edge vectors
  ee = node1 - node0
  ef = node2 - node1

  # Norms of edge vectors
  norm_e = np.linalg.norm(ee)
  norm_f = np.linalg.norm(ef)

  # Unit tangents
  te = ee / norm_e
  tf = ef / norm_f

  # Curvature binormal
  kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

  chi = 1.0 + np.dot(te, tf)
  tilde_t = (te + tf) / chi
  tilde_d2 = (m2e + m2f) / chi

  # Curvature
  kappa1 = get_strain_curvature2D(node0, node1, node2)

  # Gradient of kappa1 with respect to edge vectors
  Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
  Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

  # Populate the gradient of kappa
  gradKappa[0:2] = -Dkappa1De[0:2]
  gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
  gradKappa[4:6] = Dkappa1Df[0:2]

  # Compute the Hessian (second derivative of kappa)
  DDkappa1 = np.zeros((6, 6))

  norm2_e = norm_e**2
  norm2_f = norm_f**2

  Id3 = np.eye(3)

  # Helper matrices for second derivatives
  tt_o_tt = np.outer(tilde_t, tilde_t)
  tmp = np.cross(tf, tilde_d2)
  tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
  kb_o_d2e = np.outer(kb, m2e)

  D2kappa1De2 = (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tf_c_d2t_o_tt.T) / norm2_e - \
    kappa1 / (chi * norm2_e) * (Id3 - np.outer(te, te)) + \
    (kb_o_d2e + kb_o_d2e.T) / (4 * norm2_e)

  tmp = np.cross(te, tilde_d2)
  te_c_d2t_o_tt = np.outer(tmp, tilde_t)
  tt_o_te_c_d2t = te_c_d2t_o_tt.T
  kb_o_d2f = np.outer(kb, m2f)

  D2kappa1Df2 = (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + te_c_d2t_o_tt.T) / norm2_f - \
                  kappa1 / (chi * norm2_f) * (Id3 - np.outer(tf, tf)) + \
                  (kb_o_d2f + kb_o_d2f.T) / (4 * norm2_f)
  D2kappa1DeDf = -kappa1 / (chi * norm_e * norm_f) * (Id3 + np.outer(te, tf)) \
                  + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + \
                  tt_o_te_c_d2t - crossMat(tilde_d2))
  D2kappa1DfDe = D2kappa1DeDf.T

  # Populate the Hessian of kappa
  DDkappa1[0:2, 0:2] = D2kappa1De2[0:2, 0:2]
  DDkappa1[0:2, 2:4] = -D2kappa1De2[0:2, 0:2] + D2kappa1DeDf[0:2, 0:2]
  DDkappa1[0:2, 4:6] = -D2kappa1DeDf[0:2, 0:2]
  DDkappa1[2:4, 0:2] = -D2kappa1De2[0:2, 0:2] + D2kappa1DfDe[0:2, 0:2]
  DDkappa1[2:4, 2:4] = D2kappa1De2[0:2, 0:2] - D2kappa1DeDf[0:2, 0:2] - \
                         D2kappa1DfDe[0:2, 0:2] + D2kappa1Df2[0:2, 0:2]
  DDkappa1[2:4, 4:6] = D2kappa1DeDf[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
  DDkappa1[4:6, 0:2] = -D2kappa1DfDe[0:2, 0:2]
  DDkappa1[4:6, 2:4] = D2kappa1DfDe[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
  DDkappa1[4:6, 4:6] = D2kappa1Df2[0:2, 0:2]


  return gradKappa, DDkappa1

def get_energy_bending_linear_elastic(node0, node1, node2 = None, l_eff = None, EI = None):
  if node2 is None:
    return 0
  strain_curvature = get_strain_curvature2D(node0, node1, node2)
  E_b = 0.5 * EI / l_eff * strain_curvature**2.0
  return E_b

def grad_and_hess_energy_bending_linear_elastic(node0, node1, node2 = None, l_eff = None, EI = None):

  if node2 is None:
    return np.zeros(4), np.zeros((4,4))

  strain_curvature = get_strain_curvature2D(node0, node1, node2)
  G_strain, H_strain = grad_and_hess_strain_curvature2D(node0, node1, node2)

  gradE_strain = EI * strain_curvature / l_eff
  hessE_strain = EI / l_eff

  G = gradE_strain * G_strain
  H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)

  return G, H

def get_bend_force(q, bend_springs, EI):
    '''
    Compute the bend force vector for the entire structure.
    
    Inputs:
    q: Nx1 vector - global dof vector (positions of all nodes)
    bend_springs: list of tuples - each tuple contains (node0_index, node1_index, node2_index, l_eff)
    EI: float - bending stiffness of the springs
    
    Outputs:
    F_bend: Nx1 vector - global bend force vector
    '''
    
    num_dofs = q.shape[0]
    F_bend = np.zeros(num_dofs)
    
    for spring in bend_springs:
        node0_idx, node1_idx, node2_idx, l_eff = spring
    
        node0 = q[2*node0_idx:2*node0_idx+2]
        node1 = q[2*node1_idx:2*node1_idx+2]
        node2 = q[2*node2_idx:2*node2_idx+2]
    
        G_bend, _ = grad_and_hess_energy_bending_linear_elastic(
        node0, node1, node2, l_eff, EI)
    
        F_bend[2*node0_idx:2*node0_idx+2] -= G_bend[0:2] # negative gradient
        F_bend[2*node1_idx:2*node1_idx+2] -= G_bend[2:4]
        F_bend[2*node2_idx:2*node2_idx+2] -= G_bend[4:6]

    return F_bend
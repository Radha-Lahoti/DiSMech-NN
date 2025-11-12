import numpy as np

def get_strain_stretch_edge2D(node0, node1, l_k):
  '''
    Compute the axial stretch of a edge connecting node0 and node1.
    The code does not change even if we move from 2D to 3D.

    Parameters:
        node0 (array-like): Coordinates of the first node (start of the edge),
                            typically a 2-element array [x0, y0] for the 2D case.
        node1 (array-like): Coordinates of the second node (end of the edge),
                            typically a 2-element array [x1, y1].
        l_k (float): Reference (undeformed) length of the edge.

    Returns:
        epsX (float): Axial stretch of the edge, defined as:
                      epsX = (current_length / reference_length) - 1,
                      where current_length is the Euclidean distance between node0 and node1.

    Description:
        The function calculates the axial stretch of a 2D edge by:
        1. Computing the edge vector as the difference between node1 and node0.
        2. Calculating the current length of the edge as the Euclidean norm of the edge vector.
        3. Using the ratio of the current length to the reference length to compute the stretch.
  '''
  edge = node1 - node0
  edgeLen = np.linalg.norm(edge)
  epsX = edgeLen / l_k - 1
  return epsX

def grad_and_hess_strain_stretch_edge2D(node0, node1, l_k):
  '''
  Compute the gradient and hessian of the axial stretch of a 2D edge with
  respect to the dof vector (4 dofs: x,y coordinates of the two nodes)

  Inputs:
  node0: 2x1 vector - position of the first node
  node1: 2x1 vector - position of the last node
  l_k: reference length (undeformed) of the edge

  Outputs:
  dF: 4x1  vector - gradient of axial stretch between node0 and node 1.
  dJ: 4x4 vector - hessian of axial stretch between node0 and node 1.
  '''

  edge = node1 - node0
  edgeLen = np.linalg.norm(edge)
  tangent = edge / edgeLen
  epsX = get_strain_stretch_edge2D(node0, node1, l_k)

  dF_unit = tangent / l_k # gradient of stretch with respect to the edge vector
  dF = np.zeros(6)
  dF[0:3] = - dF_unit
  dF[3:6] = dF_unit

  # M (see below) is the Hessian of square(stretch) with respect to the edge vector
  Id3 = np.eye(3)
  M = 2.0 / l_k * ((1 / l_k - 1 / edgeLen) * Id3 + 1 / edgeLen * ( np.outer( edge, edge ) ) / edgeLen ** 2)

  # M is the Hessian of stretch with respect to the edge vector
  if epsX == 0: # Edge case
    M2 = np.zeros_like(M)
  else:
    M2 = 1.0/(2.0*epsX) * (M - 2.0*np.outer(dF_unit,dF_unit))

  dJ = np.zeros((6,6))
  dJ[0:3,0:3] = M2
  dJ[3:6,3:6] = M2
  dJ[0:3,3:6] = - M2
  dJ[3:6,0:3] = - M2

  return dF,dJ


def get_strain_stretch2D(node0, node1, node2=None, l_0=1.0, l_1=1.0):
  '''
    Compute the axial stretch of an node (node1) based on the stretches on the two edges
    connected to that node. The first edge connects node0 and node1, whereas
    the second edge connects node1 and node2. The code does not change even if
    we move from 2D to 3D.

    Parameters:
        node0 (array-like): Coordinates of the first node (start of the first edge),
                            typically a 2-element array [x0, y0] for the 2D case.
        node1 (array-like): Coordinates of the second node (end of the first edge
                            and start of the second edge),
                            typically a 2-element array [x1, y1].
        node2 (array-like): Coordinates of the third node (end of the second edge),
                            typically a 2-element array [x2, y2].
        l_0 (float): Reference (undeformed) length of the first edge.
        l_1 (float): Reference (undeformed) length of the second edge.

    Returns:
        epsilon_1 (float): Axial stretch at node1, defined as the average of the
                            stretches on the two edges connected to node1. If
                            only two nodes are specified, the stretch is defined
                            as half of the stretch on the first edge.
  '''
  # Convert inputs to numpy arrays for vectorized operations
  node0 = np.array(node0, dtype=float)
  node1 = np.array(node1, dtype=float)

  if node2 is not None:
    # If all the nodes are specified (i.e., node1 is an internal node),
    # computation is as usual (average of the strains of the two edges).
    node2 = np.array(node2, dtype=float)

    stretch_firstEdge  = get_strain_stretch_edge2D(node0, node1, l_0)
    stretch_secondEdge = get_strain_stretch_edge2D(node1, node2, l_1)

    epsilon_1 = 0.5 * stretch_firstEdge + 0.5 * stretch_secondEdge
  else:
    # Simplified case if only two nodes are specified.
    epsilon_1 = 0.5 * get_strain_stretch_edge2D(node0, node1, l_0)

  return epsilon_1

def grad_and_hess_strain_stretch2D(node0, node1, node2=None, l_0=1.0, l_1=1.0):
  '''
  Compute the gradient and hessian of the axial stretch at node1 with
  respect to the dof vector (6 dofs: x,y coordinates of three nodes).

  Inputs:
  node0: 2x1 vector - position of the first node
  node1: 2x1 vector - position of the middle node
  node2: 2x1 vector - position of the last node
  l_0: reference length (undeformed) of the first edge
  l_1: reference length (undeformed) of the second edge

  Outputs:
  dF: 6x1  vector - gradient of axial stretch between node0 and node 1.
  dJ: 6x6 vector - hessian of axial stretch between node0 and node 1.

  Description:
  The gradient and hessian at node1 is the average of the gradients and hessians
  of the two edges (one between node0 and node1 and the other one between node1
  and node2).
  '''
  # Ensure inputs are numpy arrays
  node0 = np.array(node0, dtype=float)
  node1 = np.array(node1, dtype=float)

  if node2 is not None:
    node2 = np.array(node2, dtype=float)

    G_varepsilon = np.zeros(9)
    H_varepsilon = np.zeros((9, 9))

    G1, H1 = grad_and_hess_strain_stretch_edge2D(node0, node1, l_0)
    G2, H2 = grad_and_hess_strain_stretch_edge2D(node1, node2, l_1)
    G_varepsilon[0:6] += 0.5 * G1
    G_varepsilon[3:9] += 0.5 * G2
    H_varepsilon[0:6,0:6] += 0.5 * H1
    H_varepsilon[3:9,3:9] += 0.5 * H2
  else:
    G1, H1 = grad_and_hess_strain_stretch_edge2D(node0, node1, l_0)
    G_varepsilon = 0.5 * G1
    H_varepsilon = 0.5 * H1

  return G_varepsilon, H_varepsilon


def get_energy_stretch_linear_elastic(node0, node1, node2 = None, l_0=1.0, l_1= None, EA = None):
  strain_stretch = get_strain_stretch2D(node0, node1, node2, l_0, l_1)
  E_s = 0.5 * EA * strain_stretch**2.0 * l_1
  return E_s

def grad_and_hess_energy_stretch_linear_elastic(node0, node1, node2 = None, l_0=1.0, l_1= None, EA = None):

  strain_stretch = get_strain_stretch2D(node0, node1, node2, l_0, l_1)
  G_strain, H_strain = grad_and_hess_strain_stretch2D(node0, node1, node2, l_0, l_1)

  if l_1 is None:
      l_eff = l_0
  else:
      l_eff = (l_1 + l_0) * 0.5
  gradE_strain = EA * strain_stretch * l_eff
  hessE_strain = EA * l_eff

  G = gradE_strain * G_strain
  H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)

  return G, H

def get_stretch_force(q, stretch_springs, EA):
  '''
  Compute the stretch force vector for the entire structure.

  Inputs:
  q: Nx1 vector - global dof vector (positions of all nodes)
  stretch_springs: list of tuples - each tuple contains (node0_index, node1_index, node2_index, l_eff)
  EA: float - axial stiffness of the springs

  Outputs:
  F_stretch: Nx1 vector - global stretch force vector
  '''

  num_dofs = q.shape[0]
  F_stretch = np.zeros(num_dofs)

  for spring in stretch_springs:
    node0_idx, node1_idx, node2_idx, l_0, l_1 = spring

    node0 = q[3*node0_idx:3*node0_idx+3] # get coordinates of node0 3d
    node1 = q[3*node1_idx:3*node1_idx+3] # get coordinates of node1 3d
    if node2_idx is not None:
        node2 = q[3*node2_idx : 3*node2_idx + 3]
    else:
        node2 = None


    G_stretch, _ = grad_and_hess_energy_stretch_linear_elastic(node0, node1, node2, l_0, l_1, EA=EA)

    F_stretch[3*node0_idx:3*node0_idx+3] -= G_stretch[0:3]
    F_stretch[3*node1_idx:3*node1_idx+3] -= G_stretch[3:6]
    if node2_idx is not None:
        node2 = q[3*node2_idx : 3*node2_idx + 3]
        F_stretch[3*node2_idx:3*node2_idx+3] -= G_stretch[6:9]

  return F_stretch
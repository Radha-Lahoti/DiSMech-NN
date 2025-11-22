import numpy as np
import torchdiffeq
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from typing import List

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- build a simple rod: 3 DOF per node (x,y,z) ---
def create_rod_with_nodes(num_nodes: int):
    nodes = []
    for i in range(num_nodes):
        nodes.append(np.array([i * 0.1, 0.0, 0.0], dtype=np.float32))
    return nodes


from EnergyNN import EnergyNN
from strains_torch import get_strain_stretch2D_torch, get_strain_curvature_3D_torch

class neuralODE(nn.Module):
    def __init__(self, ndof: int, m_per_dof: float = 1.0, c_per_dof: float = 0.1, g: float = 9.81, freeDOF: List[int]=None, energy_nn: nn.Module = None, bend_springs: torch.Tensor = None, dtype=torch.float32):
        super().__init__()
        self.ndof = ndof
        self.freeDOF = freeDOF
        self.call_count = 0
        # Neural elastic energy
        assert energy_nn is not None, "Pass an EnergyNN instance"
        self.energy_model = energy_nn

        # Indices / masks
        if freeDOF is None:
            freeDOF = list(range(ndof))
        free_idx = torch.as_tensor(freeDOF, dtype=torch.long)
        fixed_mask = torch.ones(ndof, dtype=torch.bool)
        fixed_mask[free_idx] = False
        self.register_buffer("free_idx", free_idx)
        self.register_buffer("fixed_mask", fixed_mask)
        self.register_buffer("free_mask", ~fixed_mask)  # same shape

        # Diagonal mass and damping for now (easy to replace with full matrices later)
        M = torch.eye(ndof) * m_per_dof       # (ndof, ndof)
        C = torch.eye(ndof) * c_per_dof       # (ndof, ndof)

        # Gravity as external force: acts in z only (every 3rd DOF starting from index 2)
        f_ext = torch.zeros(ndof)
        f_ext[2::3] = -m_per_dof * g          # (ndof,)
        # f_ext[0::3] = -m_per_dof * g          # (ndof,) # gravity in x-direction for testing

        # Register as buffers so dtype/device track the module (and no grads)
        self.register_buffer("M", M)
        self.register_buffer("C", C)
        self.register_buffer("f_ext", f_ext)

        # Pre-slice free blocks once (for diagonal M this is overkill but scales)
        self.register_buffer("M_ff", M.index_select(0, free_idx).index_select(1, free_idx))
        self.register_buffer("C_ff", C.index_select(0, free_idx).index_select(1, free_idx))

        # stencils
        self.register_buffer("bend_springs", bend_springs)

    def elastic_force(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (..., ndof)
        returns f_elastic: (..., ndof)
        """
        bend_springs = self.bend_springs.to(q.device)  # (n_springs, 3) with (i, j, k)
        f_full = torch.zeros_like(q)
        l_eff = 0.1

        for spring in bend_springs:
            i, j, k = spring

            dof_indices = [
                3 * i, 3 * i + 1, 3 * i + 2,
                3 * j, 3 * j + 1, 3 * j + 2,
                3 * k, 3 * k + 1, 3 * k + 2,
            ]

            node0 = q[..., 3 * i:3 * i + 3]
            node1 = q[..., 3 * j:3 * j + 3]
            node2 = q[..., 3 * k:3 * k + 3]

            q_spring = torch.stack([node0, node1, node2], dim=-2)  # (..., 3, 3)
            q_spring = q_spring.clone().requires_grad_(True)

            node0_s = q_spring[..., 0, :]
            node1_s = q_spring[..., 1, :]
            node2_s = q_spring[..., 2, :]

            longitudinal_strain = get_strain_stretch2D_torch(
                node0_s, node1_s, node2_s, l_eff, l_eff
            )
            curvature = get_strain_curvature_3D_torch(
                node0_s, node1_s, node2_s, l_eff
            )

            strains = torch.stack([longitudinal_strain, curvature], dim=-1)  # (..., 2)

            E_spring = self.energy_model(strains)  # (..., 1) or scalar

            (dE_dq_spring,) = torch.autograd.grad(
                outputs=E_spring.sum(),
                inputs=q_spring,
                create_graph=True
            )   # (..., 3, 3)

            f_spring = -dE_dq_spring.reshape(*q.shape[:-1], 9)  # (..., 9)
            f_full[..., dof_indices] += f_spring

        return f_full

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (..., 2*ndof)
        Returns dx/dt with same shape.
        """
        ndof = self.ndof
        q = x[..., :ndof]        # (..., ndof)
        v = x[..., ndof:]        # (..., ndof)

        # Forces
        f_el = self.elastic_force(q)          # (..., ndof)
        f_damp = torch.matmul(v, self.C.T)    # (..., ndof)

        # External + elastic + damping
        rhs = -f_damp + f_el + self.f_ext     # (..., ndof)

        # Free-DOF solve
        idx = self.free_idx
        rhs_f = rhs.index_select(-1, idx)     # (..., n_free)
        v_f   = v.index_select(-1, idx)       # (..., n_free)

        M_ff = self.M_ff
        a_f = torch.linalg.solve(M_ff, rhs_f.unsqueeze(-1)).squeeze(-1)  # (..., n_free)

        # Scatter back to full
        a_full = torch.zeros_like(v)
        v_full = torch.zeros_like(v)
        a_full[..., idx] = a_f
        v_full[..., idx] = v_f

        dxdt = torch.cat([v_full, a_full], dim=-1)

        # Logging
        self.call_count += 1
        if self.call_count % 1 == 0:
            ke = 0.5 * torch.sum(v_full**2)
            print(f"[{self.call_count}] t={t.item():.3f}, KE={ke.item():.3e}")

        return dxdt
    
def trajectory_loss(student, x0, t_eval, x_ref_traj, weight_q=1.0, weight_v=1.0):
    """
    x0: (2*ndof,)
    t_eval: (T,)
    x_ref_traj: (T, 2*ndof)
    """
    x_pred = odeint(student, x0, t_eval)      # (T, 2*ndof)
    ndof = x0.numel() // 2

    q_pred = x_pred[..., :ndof]
    v_pred = x_pred[..., ndof:]
    q_ref  = x_ref_traj[..., :ndof]
    v_ref  = x_ref_traj[..., ndof:]

    loss_q = torch.mean((q_pred - q_ref)**2)
    loss_v = torch.mean((v_pred - v_ref)**2)

    return weight_q * loss_q + weight_v * loss_v

if __name__ == "__main__":
    # --- create rod ---
    num_nodes = 11
    nodes = create_rod_with_nodes(num_nodes)  # 11 nodes -> ndof = 33
    q0 = np.array(nodes, dtype=np.float32).reshape(-1)         # positions (ndof,)
    # v0 = np.zeros_like(q0, dtype=np.float32)                   # velocities (ndof,)
    # give some nonzero init velocity to the free nodes
    v0 = np.zeros_like(q0, dtype=np.float32)
    # omega = 0.1
    # for i in range(2, num_nodes):
    #     v0[3*i + 2] = omega*(q0[3*i]-0.1) # z-velocities

    x0 = torch.tensor(np.concatenate([q0, v0]), dtype=torch.float32)  # (2*ndof,)

    print("Initial positions q0:", q0)
    print("Initial velocities v0:", v0)
    print("Initial state x0:", x0)

    # create springs (3 consecutive nodes and effective length)
    stretch_springs = []
    bend_springs = []
    for i in range(0, num_nodes):
        if i == 0:
            # boundary nodes: leff = 0.05
            l0 = np.linalg.norm(np.array(nodes[i+1] - nodes[i]))
            stretch_springs.append((i, i + 1, None, l0, None))
        elif i == num_nodes - 1:
            # boundary nodes: leff = 0.05
            l0 = np.linalg.norm(np.array(nodes[i] - nodes[i-1]))
            stretch_springs.append((i - 1, i, None, l0, None))
        else:
            # internal nodes: leff = 0.1
            l0 = np.linalg.norm(np.array(nodes[i] - nodes[i-1]))
            l1 = np.linalg.norm(np.array(nodes[i+1] - nodes[i]))
            leff = 0.5 * (l0 + l1)
            stretch_springs.append((i - 1, i, i + 1, l0, l1))
            bend_springs.append((i - 1, i, i + 1))


    # Material properties
    E = 1e8  # Young's modulus
    r0 = 0.01  # radius
    A = np.pi * r0**2  # cross-sectional area
    I = 0.25 * np.pi * r0**4  # area moment of inertia
    rho = 1200  # density
    mass = rho*A*1  # mass per segment (length 1 m)
    m_per_node = mass/num_nodes

    EA = E * A  # axial stiffness
    EI = E * I  # bending stiffness

    # print("Stretch springs:", stretch_springs)
    # print("Bend springs:", bend_springs)
    print(f"EA={EA:.3e}, EI={EI:.3e}")
    print(f"Mass per node={m_per_node:.3e}")


    # read ref trajectory from file
    x_ref = torch.load('analytical_beam_trajectory.pt') # (T, 2*ndof)

        # --- set up and integrate ---
    ndof = q0.size
    freeDOF = list(range(6, ndof))  # fix first two nodes (first 6 DOF)
    energy_nn = EnergyNN(n_in=2, hidden=32, dtype=torch.float32)
    odefunc = neuralODE(ndof=ndof,
                    m_per_dof=1.0,
                    c_per_dof=0.1,
                    g=9.81,
                    freeDOF=freeDOF,
                    energy_nn=energy_nn,
                    bend_springs=torch.tensor(bend_springs, dtype=torch.int64), 
                    dtype=torch.float32).to(device)

    optimizer = optim.Adam(odefunc.energy_model.parameters(), lr=1e-3)

    dtype=torch.float32

    n_epochs = 20
    T = 1.0
    dt = 0.0001
    steps = int(T/dt) + 1
    t = torch.linspace(0., T, steps, dtype=dtype)

    # Now move all inputs / reference data
    x0    = x0.to(device=device, dtype=dtype)
    t     = t.to(device=device, dtype=dtype)
    x_ref = x_ref.to(device=device, dtype=dtype)

    print("model param device:", next(odefunc.parameters()).device)
    print("x0 device:", x0.device)
    print("t device:", t.device)
    print("x_ref device:", x_ref.device)

    # --- training loop ---
    # store loss to plot
    losses = []
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss = trajectory_loss(odefunc, x0, t, x_ref,
                            weight_q=1.0, weight_v=1.0)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss = {loss.item():.6e}")

    # simulate
    # initial state x0: concat(q0, v0)  shape (2*ndof,)
    dt_for_sim = 0.01
    steps = int(T/dt_for_sim) + 1
    t = torch.linspace(0., T, steps, dtype=torch.float32)
    traj = odeint(odefunc, x0, t, method='rk4')   # (steps, 2*ndof)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("Solution shape:", traj.shape)           # (T, 2*ndof)
    q_traj = traj[:, :ndof]                        # (T, ndof)
    v_traj = traj[:, ndof:]                        # (T, ndof)
    print("q_traj[0]:", q_traj[0, :6])
    print("v_traj[0]:", v_traj[0, :6])

    # Convert trajectory to numpy
    q_traj_np = q_traj.detach().cpu().numpy().reshape(len(t), num_nodes, 3)
    t_np = t.detach().cpu().numpy()

    # Plot Z-displacement (vertical motion) of each node
    plt.figure(figsize=(7, 5))
    for i in range(num_nodes):
        plt.plot(t_np, q_traj_np[:, i, 2], label=f'Node {i}' if i in [0, num_nodes-1] else "", lw=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Z position [m]")
    plt.title("Vertical motion of beam nodes")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot trajectory of beam tip (last node)
    plt.figure(figsize=(6, 5))
    plt.plot(q_traj_np[:, -1, 0], q_traj_np[:, -1, 2], 'r-', lw=2)
    plt.xlabel("X position [m]")
    plt.ylabel("Z position [m]")
    plt.title("Tip trajectory (XZ plane)")
    plt.grid(True)
    plt.show()






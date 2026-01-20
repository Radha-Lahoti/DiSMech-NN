import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from typing import List

# Local imports
from neuralODE_func import neuralODE
from EnergyNN import LinearEnergyLayer
from loss_functions import trajectory_loss


def get_device() -> torch.device:
    torch.cuda.empty_cache()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_rod_with_nodes(num_nodes: int, dx: float = 0.1) -> List[np.ndarray]:
    """Create a straight rod along +x with spacing dx."""
    return [np.array([i * dx, 0.0, 0.0], dtype=np.float32) for i in range(num_nodes)]


def build_springs_and_frames(nodes: List[np.ndarray]):
    """
    Builds:
      - edges: list[(j, j+1)]
      - l0_edges: list[float]
      - d1_init, d2_init: per-edge directors
      - stretch_springs, bend_twist_springs
    """
    num_nodes = len(nodes)

    edges = []
    l0_edges = []
    d1_init = []
    d2_init = []

    stretch_springs = []
    bend_twist_springs = []

    # per-edge data
    for j in range(num_nodes - 1):
        edges.append((j, j + 1))

        e = np.array(nodes[j + 1] - nodes[j], dtype=np.float32)
        l0 = float(np.linalg.norm(e))
        l0_edges.append(l0)

        # initial reference director d1
        d1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        d1_init.append(tuple(d1.tolist()))

        # d2 from cross(edge, d1) (note: if edge || d1 -> zero; here edge is along x so OK)
        d2 = np.cross(e, d1).astype(np.float32)
        d2_init.append(d2)

    # per-node springs
    for i in range(num_nodes):
        if i == 0:
            l0 = float(np.linalg.norm(np.array(nodes[i + 1] - nodes[i], dtype=np.float32)))
            stretch_springs.append((i, i + 1, None, l0, None))
        elif i == num_nodes - 1:
            l0 = float(np.linalg.norm(np.array(nodes[i] - nodes[i - 1], dtype=np.float32)))
            stretch_springs.append((i - 1, i, None, l0, None))
        else:
            l0 = float(np.linalg.norm(np.array(nodes[i] - nodes[i - 1], dtype=np.float32)))
            l1 = float(np.linalg.norm(np.array(nodes[i + 1] - nodes[i], dtype=np.float32)))
            _leff = 0.5 * (l0 + l1)  # currently unused, but kept for clarity

            stretch_springs.append((i - 1, i, i + 1, l0, l1))
            # (xi,xj,xk, theta_prev, theta_next)
            bend_twist_springs.append((i - 1, i, i + 1, i - 1, i))

    return edges, l0_edges, d1_init, d2_init, stretch_springs, bend_twist_springs


def build_initial_state(nodes: List[np.ndarray]) -> np.ndarray:
    """
    q0: stacked node positions (3*num_nodes) plus edge DOFs (num_nodes-1)
    v0: zeros like q0
    x0: concat(q0, v0)
    """
    num_nodes = len(nodes)

    q0_nodes = np.array(nodes, dtype=np.float32).reshape(-1)  # (3*num_nodes,)
    q0_edges = np.zeros(num_nodes - 1, dtype=np.float32)      # (num_nodes-1,)
    q0 = np.concatenate([q0_nodes, q0_edges], axis=0)

    v0 = np.zeros_like(q0, dtype=np.float32)
    x0 = np.concatenate([q0, v0], axis=0)  # (2*ndof,)
    return x0


def print_material_props():
    # Material properties
    E = 1e7
    r0 = 0.01
    A = np.pi * r0**2
    I = 0.25 * np.pi * r0**4
    J = 2 * I
    rho = 1200

    mass = rho * A * 1.0  # for length 1 m (your comment)
    return E, r0, A, I, J, rho, mass


def main():
    device = get_device()
    dtype = torch.float32

    # -----------------------------
    # Problem setup
    # -----------------------------
    num_nodes = 11
    nodes = create_rod_with_nodes(num_nodes)

    x0_np = build_initial_state(nodes)
    q0_np = x0_np[: (x0_np.size // 2)]
    ndof = q0_np.size

    edges, l0_edges, d1_init, d2_init, stretch_springs, bend_twist_springs = build_springs_and_frames(nodes)

    # Material properties
    E, r0, A, I, J, rho, mass = print_material_props()
    m_per_node = (mass / num_nodes)

    EA = E * A
    EI = E * I
    GJ = (E / (2 * (1 + 0.5))) * J  # assumes nu=0.5
    stiffnesses = torch.tensor([EA, EI, EI, GJ], dtype=dtype)

    print(f"EA={EA:.3e}, EI={EI:.3e}, GJ={GJ:.3e}")

    # Reference trajectory
    x_ref = torch.load("analytical_beam_trajectory.pt")  # (T, 2*ndof)

    # Fix first two nodes (first 6 DOF in node positions); remaining are free
    freeDOF = list(range(6, ndof))

    # Energy model
    # NOTE: you currently create n_strain=3 (so it ignores torsion if you had it)
    energy_nn = LinearEnergyLayer(n_strain=3, dtype=dtype)

    # If LinearEnergyLayer exposes stiffness() method:
    try:
        W = energy_nn.stiffness().detach().cpu()
        print("Initial stiffness (energy_nn):", W)
    except Exception:
        pass

    # ODE function
    odefunc = neuralODE(
        edges=torch.tensor(edges, dtype=torch.int64),
        l0_edges=l0_edges,
        n_nodes=num_nodes,
        ndof=ndof,
        m_per_dof=m_per_node,
        c_per_dof=0.1,
        g=9.81,
        freeDOF=freeDOF,
        energy_nn=energy_nn,
        bend_twist_springs=torch.tensor(bend_twist_springs, dtype=torch.int64),
        d1_init=torch.tensor(d1_init, dtype=dtype),
        d2_init=torch.tensor(d2_init, dtype=dtype),
        dtype=dtype,
    ).to(device)

    optimizer = optim.Adam(odefunc.energy_model.parameters(), lr=1e-3)

    # -----------------------------
    # Time grid + move tensors
    # -----------------------------
    n_epochs = 50
    T = 0.6
    dt = 1e-4
    steps = int(T / dt) + 1
    t = torch.linspace(0.0, T, steps, dtype=dtype, device=device)

    x0 = torch.tensor(x0_np, dtype=dtype, device=device)

    # Use only portion of x_ref matching t
    x_ref = x_ref[:steps].to(device=device, dtype=dtype)

    print("model param device:", next(odefunc.parameters()).device)
    print("x0 device:", x0.device)
    print("t device:", t.device)
    print("x_ref device:", x_ref.device)

    # -----------------------------
    # Training loop
    # -----------------------------
    losses = []
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss = trajectory_loss(odefunc, x0, t, x_ref, weight_q=1.0, weight_v=1.0)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        if epoch % 10 == 0:
            try:
                k = odefunc.energy_model.stiffness().detach().cpu().numpy()
                print("stiffness =", k)
            except Exception:
                pass
            print(f"Epoch {epoch:4d} | loss={loss.item():.3e}")

    # -----------------------------
    # Final rollout + plots
    # -----------------------------
    with torch.no_grad():
        traj = odeint(odefunc, x0, t, method="rk4")  # (steps, 2*ndof)

    # Split trajectory: q and v
    q_traj = traj[:, :ndof]  # (steps, ndof)
    v_traj = traj[:, ndof:]  # (steps, ndof)

    # Extract node positions only: first 3*num_nodes entries
    q_nodes = q_traj[:, : 3 * num_nodes]  # (steps, 3*num_nodes)
    q_traj_np = q_nodes.detach().cpu().numpy().reshape(len(t), num_nodes, 3)
    t_np = t.detach().cpu().numpy()

    # Z-displacement of each node
    plt.figure(figsize=(7, 5))
    for i in range(num_nodes):
        plt.plot(
            t_np,
            q_traj_np[:, i, 2],
            label=(f"Node {i}" if i in [0, num_nodes - 1] else ""),
            lw=1,
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Z position [m]")
    plt.title("Vertical motion of beam nodes")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Tip trajectory (XZ plane)
    plt.figure(figsize=(6, 5))
    plt.plot(q_traj_np[:, -1, 0], q_traj_np[:, -1, 2], lw=2)
    plt.xlabel("X position [m]")
    plt.ylabel("Z position [m]")
    plt.title("Tip trajectory (XZ plane)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

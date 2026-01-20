from torchdiffeq import odeint
import torch

def trajectory_loss(student, x0, t_eval, x_ref_traj,
                    weight_q=1.0, weight_v=1.0
                    ):
    """
    x0: (2*ndof,)
    t_eval: (T,)
    x_ref_traj: (T, 2*ndof)
    """

    # x_pred = odeint(
    #     student, x0, t_eval,
    #     method='dopri5',
    #     rtol=1e-3,
    #     atol=1e-3,
    #     options={'max_num_steps': 10000}
    # )  # (T, 2*ndof)

    x_pred = odeint(
        student, x0, t_eval,
        method='rk4',
        options={"step_size": float(t_eval[1]-t_eval[0])}
    )  # (T, 2*ndof)

    ndof = x0.numel() // 2

    q_pred = x_pred[..., :ndof]
    v_pred = x_pred[..., ndof:]
    q_ref  = x_ref_traj[..., :ndof]
    v_ref  = x_ref_traj[..., ndof:]

    loss_q = torch.mean((q_pred - q_ref)**2)
    loss_v = torch.mean((v_pred - v_ref)**2)
    return weight_q * loss_q + weight_v * loss_v



def batched_segment_loss(
    student,
    x_ref,           # (steps_total, 2*ndof)
    dt: float,
    seg_T: float,
    batch_size: int,
    weight_q: float = 1.0,
    weight_v: float = 1.0,
    method: str = 'dopri5',
    options: dict = {'max_num_steps': 10000}
):
    device = x_ref.device
    dtype  = x_ref.dtype

    total_steps, state_dim = x_ref.shape
    ndof = state_dim // 2
    seg_steps = int(seg_T / dt) + 1

    max_start = total_steps - seg_steps
    assert max_start >= 0, "trajectory too short for this seg_T"

    # sample random segment starts: shape (B,)
    start_idx = torch.randint(0, max_start + 1, (batch_size,), device=device)

    # build index matrix for each segment in the batch
    # shape (B, seg_steps): [i, :] = [start_i, start_i+1, ..., start_i+seg_steps-1]
    offsets = torch.arange(seg_steps, device=device)
    idx_mat = start_idx.unsqueeze(1) + offsets.unsqueeze(0)   # (B, seg_steps)

    # initial conditions: ref at each start time, shape (B, state_dim)
    x0_batch = x_ref[start_idx]   # (B, 2*ndof)

    # reference over segments: (B, seg_steps, state_dim)
    x_ref_batch = x_ref[idx_mat]  # (B, seg_steps, 2*ndof)
    # odeint returns (seg_steps, B, state_dim), so transpose to that layout
    x_ref_batch = x_ref_batch.transpose(0, 1)  # (seg_steps, B, 2*ndof)

    # time vector shared by all segments (relative time, ODE is autonomous)
    t_seg = torch.linspace(0., seg_T, seg_steps, device=device, dtype=dtype)

    # integrate with batched initial condition
    x_pred_batch = odeint(
        student,
        x0_batch,    # (B, 2*ndof)
        t_seg,       # (seg_steps,)
        method=method,
        rtol=1e-3,
        atol=1e-3,
        options=options
    )  # (seg_steps, B, 2*ndof)

    q_pred = x_pred_batch[..., :ndof]      # (seg_steps, B, ndof)
    v_pred = x_pred_batch[..., ndof:]      # (seg_steps, B, ndof)
    q_ref  = x_ref_batch[..., :ndof]       # (seg_steps, B, ndof)
    v_ref  = x_ref_batch[..., ndof:]       # (seg_steps, B, ndof)

    loss_q = torch.mean((q_pred - q_ref)**2)
    loss_v = torch.mean((v_pred - v_ref)**2)
    return weight_q * loss_q + weight_v * loss_v

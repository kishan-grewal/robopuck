import torch


def fk(q: torch.Tensor, l: float, base_z: float):
    # q could be (3,) or (N, 3)
    # unpack q with ... therefore
    q1, q2, q3 = [q[..., i] for i in range(3)]
    q23 = q2 + q3

    # EE_x before yaw
    rho = l * (torch.cos(q2) + torch.cos(q23))
    # EE_z
    z = base_z + l * (torch.sin(q2) + torch.sin(q23))
    # EE_x and EE_y
    px = rho * torch.cos(q1) # accounting for base
    py = rho * torch.sin(q1) # accounting for base

    # px py and z are (N,) or (,)
    return torch.stack([px, py, z], dim=-1)  # (N, 3) or (3,)

def jac(q: torch.Tensor, l: float, base_z: float):
    f = lambda q: fk(q, l, base_z)
    Jf = torch.func.jacrev(f)  # the wonky space
    J = torch.func.vmap(Jf)
    # f takes (3,) to (3,)
    # Jf takes (3,) to (3, 3)
    # J applies (3, 3) across N envs
    return J(q)

def dls(J, v_des, lam=0.05):
    # ic| eager_shapes: [('aten::bmm', [[16384, 3, 3], [16384, 3, 3]]),
    # ic| eager: {'count': 147, 'top': [('ampere_sgemm_128x128_nt', 287.3),
    # ic| compiled: {'count': 25, 'top': [('ampere_sgemm_128x128_nt', 287.0),
    # the J.mT @ J is takes  287.3 / 502.6 for eager and 287.3 / 337.1 for compiled

    # to replace:
    # A = J.mT @ J + lam**2 * torch.eye(3, device=J.device)

    # pick qdot to minimise two things at once
    # |J qdot - v_des|^2  + lam^2 |qdot|^2
    # (J^T J + lam^2 I) qdot = J^T v_des
    # A qdot = b
    
    # for each env, row we want the row times row transpose, 
    # this outer product creates mats we can sum together to get the mat^t @ mat
    # might not be intuitive, a simple paper example for a 2x2 helps
    # J is (N, 3, 3)
    view_1 = J[:, :, :, None]  # n, k, i, 1
    view_2 = J[:, :, None, :]  # n, k, 1, j
    JmT_J = (view_1 @ view_2).sum(dim=1)  # sum all the mats
    A = JmT_J + lam**2 * torch.eye(3, device=J.device)

    # .mT is matrix transpose of the 3, 3 in (N, 3, 3)
    b = J.mT @ v_des[..., None] 
    # @ takes the last two
    # J is (N, 3, 3) and V_des is (N, 3)
    # so we cant just broadcast it 
    
    # qdot is torch.linalg.solve(A, b)
    return torch.linalg.solve(A, b)[..., 0]

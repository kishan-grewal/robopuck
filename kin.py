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
    # pick qdot to minimise two things at once
    # |J qdot - v_des|^2  + lam^2 |qdot|^2
    # (J^T J + lam^2 I) qdot = J^T v_des
    # A qdot = b
    A = J.mT @ J + lam**2 * torch.eye(3, device=J.device)
    # .mT is matrix transpose of the 3, 3 in (N, 3, 3)
    b = J.mT @ v_des[..., None] 
    # @ takes the last two
    # J is (N, 3, 3) and V_des is (N, 3)
    # so we cant just broadcast it 
    
    # qdot is torch.linalg.solve(A, b)
    return torch.linalg.solve(A, b)[..., 0]

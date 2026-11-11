import math
import numpy as np
import torch
from dataclasses import dataclass, field
import rerun as rr

from kin import fk

# import lovely_tensors as lt
# lt.monkey_patch()  # lt override torch representation


@dataclass
class Sim:
    # sim
    n: int = 16384
    dt: float = 0.005
    eps: float = 0.001
    g: float = 9.81
    # puck
    m: float = 0.1
    mu: float = 0.3
    a: float = 0.02
    r_p: float = 0.05
    h: float = 0.03
    k: float = 2000
    # pillar
    r_P: float = 0.10
    # arm
    r_e: float = 0.02
    l: float = 0.4
    base_z: float = 0.4
    u_clip: float = 2.0
    dt_ctrl: float = 0.02  # 4 sim steps per control step
    c: float = 10

    device: str = "cuda"
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # (N, 2)
    x: torch.Tensor = field(init=False)
    # (N, 2)
    v: torch.Tensor = field(init=False)
    # (N, 3)
    q: torch.Tensor = field(init=False) # q1 yaw, q2 q3 pitch

    x0: torch.Tensor = field(init=False)
    route: torch.Tensor = field(init=False)
    q0: torch.Tensor = field(init=False)
    c_P: torch.Tensor = field(init=False)

    t: torch.Tensor = field(init=False)  # per env timestep
    episode: torch.Tensor = field(init=False)  # per env episode counter

    def __post_init__(self):
        # puck
        self.x = torch.zeros(self.n, 2, device=self.device)
        self.v = torch.zeros(self.n, 2, device=self.device)
        self.x0 = torch.tensor([0.2, 0.0], device=self.device)
        self.route = torch.tensor([[0.3, 0.2], [0.5, 0.2], [0.6, 0.0]], device=self.device)
        # pillar
        self.c_P = torch.tensor([0.4, 0.0], device=self.device)
        # arm
        self.q = torch.zeros(self.n, 3, device=self.device)
        # -0.3, 1.5, -2.9
        self.q0 = torch.tensor([-0.3, 1.5, -2.9], device=self.device)

        # state
        self.t = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.episode = torch.full((self.n,), -1, dtype=torch.long, device=self.device)

    def friction_force(self):
        # F_f = -mu mg . v / (|v| + eps)
        mag_v = torch.norm(self.v, dim=-1, keepdim=True)
        assert(mag_v.shape == (self.n, 1))
        return -self.mu * self.m * self.g * self.v / (mag_v + self.eps)

    def pillar_force(self):
        # F_P = k * min(0, r - R)
        r = torch.norm(self.x - self.c_P, dim=-1, keepdim=True)  # (N, 2) and (2,) broadcasts to (N, 2) then norm we want (N, 1)
        assert(r.shape == (self.n, 1))
        d_pP = torch.clamp(self.r_p + self.r_P - r, min=0.0)
        normal = (self.x - self.c_P) / (r + self.eps)  # (N, 2) and (N, 1) broadcasts to (N, 2)
        assert(normal.shape == (self.n, 2))
        return self.k * d_pP * normal

    def ee_force(self, p, pdot):
        # p, pdot: (N, 3) -> (N, 2)
        pxy = p[:, :2]; pz = p[:, 2:]
        pdotxy = pdot[:, :2]
        # x, p, r_p, r_e
        # Q1: are they touching and how much
        dist = torch.norm(self.x - pxy, dim=-1, keepdim=True)  # x and pxy are both (N, 2)
        assert(dist.shape == (self.n, 1))
        d_pe = torch.clamp(self.r_p + self.r_e - dist, min=0.0)
        active_z = (pz - self.r_e < self.h)
        active_xy = (d_pe > 0.0)
        active = (active_z & active_xy).float()  # (N, 1)
        # Q2: which way does the puck get pushed
        n_e = (self.x - pxy) / (dist + self.eps)
        # Q3: how hard does the puck get pushed 
        # spring force and damping force
        s = -((self.v - pdotxy) * n_e).sum(dim=-1, keepdim=True) # (N, 2) product to (N, 2) then sum the 2 dim
        # norm and dot seem to both need dim-1 keepdimTrue]
        return torch.clip(self.k * d_pe + self.c * s, min=0.0) * n_e * active  # (N, 2)

    def step(self, u: torch.Tensor):
        # p = mv -> F deltat = m deltav

        # arm calcs
        assert(u.shape == (self.n, 3))
        u = torch.clamp(u, min=-self.u_clip, max=self.u_clip)
        p_old = fk(self.q, self.l, self.base_z)
        self.q = self.q + self.dt_ctrl * u
        p_new = fk(self.q, self.l, self.base_z)
        pdot = (p_new - p_old) / self.dt_ctrl

        for i in range(4):
            p = torch.lerp(p_old, p_new, (i + 1) / 4)

            F_f = self.friction_force()
            F_p = self.pillar_force()
            F_c = self.ee_force(p, pdot)
            F_resultant = F_f + F_p + F_c

            self.v = self.v + self.dt * F_resultant / self.m  # F_resultant = ma
            self.x = self.x + self.dt * self.v

        # new code for adding multiple episodes now, and not all the envs in lockstep since a policy can fail unlike the expert
        self.t = self.t + 1  # (N,) broadcasts with () to (N,)
        done = (self.t == 600)  # (N,) -> (N,)
        goal = self.route[-1]  # (3, 2) -> (2,)
        d_goal = torch.norm(self.x - goal, dim=-1)  # (N, 2) - (2,) broadc to (N, 2), we want to remove the last dim to get (N,)
        success = (d_goal < 0.03)  # (N,) -> (N,)

        # reset ONLY the done envs
        # self.x = torch.where(done, self.route[0], self.x)
        # note: The tensors condition, input, other must be broadcastable <broadcasting-semantics>.
        # self.x = torch.where(done[:, None], .....)
        self.reset(done)

        # obs = x, v, q
        # and we return obs, done
        obs = torch.cat([self.x, self.v, self.q], dim=-1)
        return obs, done, success  # (N, 7) (N,) (N,)

    def reset(self, mask: torch.Tensor):
        fresh: torch.Tensor = torch.rand(self.n, 2, device=self.device)
        # random in [0, 1)
        fresh = (fresh * 2 - 1) * self.a + self.x0
        self.x = torch.where(mask[:, None], fresh, self.x)
        self.v = torch.where(mask[:, None], torch.zeros_like(self.v), self.v)     
        self.q = torch.where(mask[:, None], self.q0, self.q)

        # state stuff
        self.t = torch.where(mask, 0, self.t)
        self.episode = torch.where(mask, self.episode + 1, self.episode)

        # asserts
        assert(self.x.shape == (self.n, 2))
        assert(self.v.shape == (self.n, 2))
        assert(self.q.shape == (self.n, 3))

        assert(self.t.shape == (self.n,))
        assert(self.episode.shape == (self.n,))
        # even plain Python uses (x,) -> tuple but (x) -> number, this is why torch is this way

    def log(self, i, env=0):
        # sim knows its own state so nothing gets passed in
        # a (k,) vector logged as Scalars becomes k series on one entity
        rr.set_time("step", sequence=i)
        state = {
            "puck/x": self.x,
            "puck/v": self.v,
            "arm/q": self.q,
            "arm/ee": fk(self.q, self.l, self.base_z),
        }
        for name, t in state.items():
            rr.log(name, rr.Scalars(t[env].detach().cpu().numpy()))  # (N, k) -> (k,)

        # scene, 3D
        x = self.x[env].detach().cpu().numpy()
        p = state["arm/ee"][env].detach().cpu().numpy()
        q1, q2 = self.q[env, 0].item(), self.q[env, 1].item()
        elbow = [self.l * math.cos(q2) * math.cos(q1), self.l * math.cos(q2) * math.sin(q1), self.base_z + self.l * math.sin(q2)]
        rr.log("world/puck", rr.Cylinders3D(lengths=[self.h], radii=[self.r_p], centers=[[x[0], x[1], self.h / 2]]))
        rr.log("world/ee", rr.Points3D([p], radii=[self.r_e]))
        # arm links as cylinders: one from base to elbow, one from elbow to hand
        joints = np.array([[0, 0, self.base_z], elbow, p])
        a, b = joints[:-1], joints[1:]  # (2, 3) each
        d = b - a
        length = np.linalg.norm(d, axis=-1)  # (2,)
        # cylinders are drawn along z so rotate z onto each link direction
        axis = np.cross([0, 0, 1], d / length[:, None])  # (2, 3)
        axis[np.linalg.norm(axis, axis=-1) < 1e-9] = [1, 0, 0]  # link straight up or down
        angle = np.arccos(d[:, 2] / length)  # (2,)
        rr.log("world/arm", rr.Cylinders3D(
            lengths=length, radii=[0.01, 0.01], centers=(a + b) / 2,
            rotation_axis_angles=[rr.RotationAxisAngle(ax, radians=an) for ax, an in zip(axis, angle)],
        ))

    def log_scene(self):
        # things that never move, logged once
        h_P = 0.045  # pillar height for display only
        rr.log("world/pillar", rr.Cylinders3D(lengths=[h_P], radii=[self.r_P], centers=[[*self.c_P.tolist(), h_P / 2]]), static=True)
        rr.log("world/goal", rr.Cylinders3D(lengths=[0.002], radii=[0.03], centers=[[*self.route[-1].tolist(), 0.001]], colors=[[80, 200, 120]]), static=True)

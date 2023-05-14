import torch

from torch import tensor


def SE2_kinematics(x: tensor, action: tensor, tau: float) -> tensor:
    wt_2 = action[1] * tau / 2
    t_v_sinc_term = tau * action[0] * torch.sinc(wt_2 / torch.pi)
    ret_x = torch.empty(3)
    ret_x[0] = x[0] + t_v_sinc_term * torch.cos(x[2] + wt_2)
    ret_x[1] = x[1] + t_v_sinc_term * torch.sin(x[2] + wt_2)
    ret_x[2] = x[2] + 2 * wt_2
    return ret_x


def landmark_motion(mu: tensor, v: tensor, A: tensor, B: tensor) -> tensor:
    return mu @ A.T + v @ B.T

def landmark_motion_real(mu: tensor, v: tensor, A: tensor, B: tensor, W: tensor) -> tensor:
    # print(mu, A.T, torch.normal(mean=torch.zeros(mu.size()), std=torch.sqrt(W)), "\n\n\n")
    return mu @ A.T + v @ B.T + torch.normal(mean=torch.zeros(mu.size()), std=torch.sqrt(W))

def triangle_SDF(q: tensor, psi: float, r: float) -> tensor:
    x, y = q[:, 0], q[:, 1]
    p_x = r / (1 + torch.sin(psi))

    a_1, a_2, a_3 = tensor([-1, 1 / torch.tan(psi)]), tensor([-1, -1 / torch.tan(psi)]), tensor([1, 0],
                                                                                                dtype=torch.float)
    b_1, b_2, b_3 = 0, 0, -r
    q_1, q_2, q_3 = tensor([r, r * torch.tan(psi)]), tensor([r, -r * torch.tan(psi)]), tensor([0, 0])
    l_1_low, l_1_up, l_2_low, l_2_up = l_function(x, psi, r, p_x)

    SDF = torch.empty(q.size()[0])
    for i in range(q.size()[0]):
        if y[i] >= l_1_up[i]:
            SDF[i] = torch.linalg.norm(q[i, :] - q_1, dtype=torch.float)
        elif l_1_low[i] <= y[i] < l_1_up[i]:
            SDF[i] = (q[i, :] @ a_1 + b_1) / torch.linalg.norm(a_1)
        elif x[i] < 0 and l_2_up[i] <= y[i] < l_1_low[i]:
            SDF[i] = torch.linalg.norm(q[i, :] - q_3, dtype=torch.float)
        elif x[i] > p_x and l_2_up[i] <= y[i] < l_1_low[i]:
            SDF[i] = (q[i, :] @ a_3 + b_3) / torch.linalg.norm(a_3, dtype=torch.float)
        elif l_2_low[i] < y[i] < l_2_up[i]:
            SDF[i] = (q[i, :] @ a_2 + b_2) / torch.linalg.norm(a_2, dtype=torch.float)
        else:
            SDF[i] = torch.linalg.norm(q[i, :] - q_2, dtype=torch.float)

    # SDF = torch.linalg.norm(q - q_2, dim=1, dtype=torch.float)

    # P_1_inds = torch.nonzero(y >= l_1_up)
    # SDF[P_1_inds] = torch.linalg.norm(q[P_1_inds, :] - q_1, dim=-1, dtype=torch.float)

    # D_1_inds = torch.nonzero(torch.logical_and(l_1_low <= y, y < l_1_up))
    # SDF[D_1_inds] = (q[D_1_inds, :] @ a_1 + b_1) / torch.linalg.norm(a_1)
    #
    # P_3_inds = torch.nonzero(torch.logical_and(x < 0, torch.logical_and(l_2_up <= y, y < l_1_low)))
    # SDF[P_3_inds] = torch.linalg.norm(q[P_3_inds, :] - q_3, dim=-1, dtype=torch.float)
    #
    # D_3_inds = torch.nonzero(torch.logical_and(x > p_x, torch.logical_and(l_2_up <= y, y < l_1_low)))
    # SDF[D_3_inds] = (q[D_3_inds, :] @ a_3 + b_3) / torch.linalg.norm(a_3, dtype=torch.float)
    #
    # D_2_inds = torch.nonzero(torch.logical_and(l_2_low < y, y < l_2_up))
    # SDF[D_2_inds] = (q[D_2_inds, :] @ a_2 + b_2) / torch.linalg.norm(a_2, dtype=torch.float)

    return SDF


def l_function(x, psi, r, p_x):
    l_1_low, l_2_up = r * torch.tan(psi) * torch.ones(x.shape), -r * torch.tan(psi) * torch.ones(x.shape)

    inds_1 = torch.nonzero(x < 0)
    l_1_low[inds_1], l_2_up[inds_1] = - x[inds_1] / torch.tan(psi), x[inds_1] / torch.tan(psi)

    inds_2 = torch.nonzero(torch.logical_and(0 <= x, x < p_x))
    l_1_low[inds_2], l_2_up[inds_2] = 0, 0

    inds_3 = torch.nonzero(torch.logical_and(p_x <= x, x < r))
    l_1_low[inds_3] = torch.tan(torch.pi / 4 + psi / 2) * x[inds_3] - r / torch.cos(psi)
    l_2_up[inds_3] = - torch.tan(torch.pi / 4 + psi / 2) * x[inds_3] + r / torch.cos(psi)

    l_1_up, l_2_low = r * torch.tan(psi) * torch.ones(x.shape), -r * torch.tan(psi) * torch.ones(x.shape)

    inds_4 = torch.nonzero(x < r)
    l_1_up[inds_4] = - (x[inds_4] - r) / torch.tan(psi) + r * torch.tan(psi)
    l_2_low[inds_4] = (x[inds_4] - r) / torch.tan(psi) - r * torch.tan(psi)

    return l_1_low, l_1_up, l_2_low, l_2_up


def get_transformation(x: tensor) -> tensor:
    cos_term = torch.cos(x[2])
    sin_term = torch.sin(x[2])
    # transformation = torch.zeros((3, 3), requires_grad=False)
    # transformation[0, :] = tensor([cos_term, - sin_term, x[0]])
    # transformation[1, :] = tensor([sin_term, cos_term, x[1]])
    # transformation[2, 2] = 1
    return tensor([[cos_term, - sin_term, x[0]],
                   [sin_term, cos_term, x[1]],
                   [0, 0, 1]], requires_grad=True)


def phi(SDF: tensor, kappa: float) -> tensor:
    return 0.5 * (1 + torch.erf(SDF / (2**0.5 * kappa) - 2))

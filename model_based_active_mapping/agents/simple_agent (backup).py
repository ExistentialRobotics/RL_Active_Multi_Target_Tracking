import os, yaml
import torch

from torch import tensor
from torch.optim import Adam
# from model_based_active_mapping.models.simple_policy_net import SimplePolicyNet
from model_based_active_mapping.models.policy_net import PolicyNet
from model_based_active_mapping.utilities.utils import SE2_kinematics, get_transformation, triangle_SDF, phi


class SimpleAgent:

    def __init__(self, width, height, tau, lr, init_info, W, psi, radius, kappa, V, horizon):
        self._env_size = tensor([width, height])
        self._tau = tau

        # input_dim = 1 * 2 + 3
        # self._policy = SimplePolicyNet(input_dim=input_dim)

        input_dim = 1 * 4 + 3
        self._policy = PolicyNet(input_dim=input_dim)
        self._policy.train()

        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

        self._W = W
        self._psi = psi
        self._radius = radius
        self._kappa = kappa
        self._inv_V = V ** (-1)

        self._horizon = horizon

        self._info = torch.empty((horizon + 1, 2))
        self._info[0] = init_info * torch.ones(2)

    def run_simple_agent_training(self, debug=False):
        mu = torch.rand(2) * 10

        # x = torch.empty((self._horizon + 1, 3))
        # x[0, :2] = torch.rand(2) * 10
        # x[0, 2] = (torch.rand(1) * 2 - 1) * torch.pi

        # q = torch.empty((self._horizon, 2))
        # M = torch.empty((self._horizon, 2))
        # action = torch.empty((self._horizon, 2))
        # net_input = torch.empty((self._horizon, 7))
        # SDF = torch.empty(self._horizon)

        # for i in range(self._horizon):
        #     # net_input = torch.hstack((x, mu))
        #     net_input[i] = torch.hstack((x[i], self._info[i], mu))
        #     action[i] = self._policy.forward(net_input[i])
        #     x[i + 1] = SE2_kinematics(x[i], action[i], self._tau)
        #
        #     q[i] = torch.hstack((mu[0] * torch.cos(x[i + 1, 2]) - mu[1] * torch.sin(x[i + 1, 2]) + x[i + 1, 0],
        #                          mu[0] * torch.sin(x[i + 1, 2]) + mu[1] * torch.cos(x[i + 1, 2]) + x[i + 1, 1]))
        #
        #     SDF[i] = triangle_SDF(q[None, i, :], self._psi, self._radius)
        #     M[i] = (1 - phi(SDF[i], self._kappa)) * self._inv_V
        #     self._info[i + 1] = (self._info[i] ** (-1) + self._W) ** (-1) + M[i]

        # x_1 = torch.empty(3)
        # x_1[:2] = torch.rand(2) * 10
        # x_1[2] = (torch.rand(1) * 2 - 1) * torch.pi
        #
        # info_1 = 0.01 * torch.ones(2)
        #
        # net_input_1 = torch.hstack((x_1, info_1, mu))
        # action_1 = self._policy.forward(net_input_1)
        # x_2 = SE2_kinematics(x_1, action_1, self._tau)
        #
        # q_1 = torch.hstack((mu[0] * torch.cos(x_2[2]) - mu[1] * torch.sin(x_2[2]) + x_2[0],
        #                     mu[0] * torch.sin(x_2[2]) + mu[1] * torch.cos(x_2[2]) + x_2[1]))
        #
        # SDF_1 = triangle_SDF(q_1[None, :], self._psi, self._radius)
        # M_1 = (1 - phi(SDF_1, self._kappa)) * self._inv_V
        # info_2 = (info_1 ** (-1) + self._W) ** (-1) + M_1
        #
        # net_input_2 = torch.hstack((x_2, info_2, mu))
        # action_2 = self._policy.forward(net_input_2)
        # x_3 = SE2_kinematics(x_2, action_2, self._tau)
        #
        # q_2 = torch.hstack((mu[0] * torch.cos(x_3[2]) - mu[1] * torch.sin(x_3[2]) + x_3[0],
        #                     mu[0] * torch.sin(x_3[2]) + mu[1] * torch.cos(x_3[2]) + x_3[1]))
        #
        # SDF_2 = triangle_SDF(q_2[None, :], self._psi, self._radius)
        # M_2 = (1 - phi(SDF_2, self._kappa)) * self._inv_V
        # info_3 = (info_2 ** (-1) + self._W) ** (-1) + M_2
        #
        # loss = - torch.sum(torch.log(info_3))

        # loss = - torch.sum(torch.log(self._info[-1]))

        self._policy_optimizer.zero_grad()

        if debug:
            param_list = []
            grad_power = 0
            for i, p in enumerate(self._policy.parameters()):
                param_list.append(p.data.detach().clone())
                if p.grad is not None:
                    grad_power += (p.grad**2).sum()
                else:
                    grad_power += 0

            print("Gradient power before backward: {}".format(grad_power))

        loss.backward()
        self._policy_optimizer.step()

        if debug:
            grad_power = 0
            total_param_ssd = 0
            for i, p in enumerate(self._policy.parameters()):
                if p.grad is not None:
                    grad_power += (p.grad ** 2).sum()
                else:
                    grad_power += 0
                total_param_ssd += ((param_list[i] - p.data) ** 2).sum()

            print("Gradient power after backward: {}".format(grad_power))
            print("SSD of weights after applying the gradient: {}".format(total_param_ssd))


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)

    params_filename = os.path.join(os.path.abspath(os.path.join("", os.pardir)), "params/params.yaml")
    assert os.path.exists(params_filename)
    with open(os.path.join(params_filename)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    env_width = params['env_width']
    env_height = params['env_height']
    tau = params['tau']
    lr = params['lr']
    init_info = params['init_info']

    W = torch.zeros(2)
    W[0] = params['motion']['W']['_1']
    W[1] = params['motion']['W']['_2']

    radius = params['FoV']['radius']
    psi = tensor([params['FoV']['psi']])
    kappa = params['FoV']['kappa']

    V = torch.zeros(2)
    V[0] = params['FoV']['V']['_1']
    V[1] = params['FoV']['V']['_2']

    horizon = params['horizon']

    agent = SimpleAgent(width=env_width, height=env_height, tau=tau, lr=lr, init_info=init_info, W=W, psi=psi,
                        radius=radius, kappa=kappa, V=V, horizon=horizon)
    agent.run_simple_agent_training(debug=True)
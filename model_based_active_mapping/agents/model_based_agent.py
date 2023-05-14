import torch

from torch import tensor
from torch.optim import SGD, Adam
from models.policy_net import PolicyNet
from models.policy_net_att import PolicyNetAtt
from utilities.utils import landmark_motion, triangle_SDF, get_transformation, phi


class ModelBasedAgent:

    def __init__(self, num_landmarks, init_info, A, B, W, radius, psi, kappa, V, lr):
        self._init_info = init_info
        self._info = None

        self._num_landmarks = num_landmarks
        self._A = A
        self._B = B
        self._W = W
        self._psi = psi
        self._radius = radius
        self._kappa = kappa
        self._V = V
        self._inv_V = V ** (-1)

        # input_dim = num_landmarks * 4 + 3
        input_dim = num_landmarks * 4
        self._policy = PolicyNet(input_dim=input_dim)

        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

    def reset_agent_info(self):
        self._info = self._init_info * torch.ones((self._num_landmarks, 2))

    def reset_estimate_mu(self, mu_real):
        self._mu_update = mu_real + torch.normal(mean=torch.zeros(self._num_landmarks, 2), std=torch.sqrt(self._V))  # with the shape of (num_landmarks, 2)

    def eval_policy(self):
        self._policy.eval()

    def train_policy(self):
        self._policy.train()

    def plan(self, v, x):
        self._mu_predict = torch.clip(landmark_motion(self._mu_update, v, self._A, self._B),
                                      min=-tensor([self._num_landmarks, self._num_landmarks]),
                                      max=tensor([self._num_landmarks, self._num_landmarks]))
        self._info = (self._info**(-1) + self._W)**(-1)

        q_predict = torch.vstack(((self._mu_predict[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - self._mu_predict[:, 0]) * torch.sin(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.cos(x[2]))).T

        # net_input = torch.hstack((x, self._info.flatten(), next_mu.flatten()))

        net_input = torch.hstack((self._info.flatten(), q_predict.flatten()))
        # net_input = q.flatten()
        action = self._policy.forward(net_input)
        return action

    def update_info_mu(self, mu_real, x):
        q_real = torch.vstack(((mu_real[:, 0] - x[0]) * torch.cos(x[2]) + (mu_real[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - mu_real[:, 0]) * torch.sin(x[2]) + (mu_real[:, 1] - x[1]) * torch.cos(x[2]))).T
        sensor_value = torch.zeros(self._num_landmarks, 2)
        SDF_real = triangle_SDF(q_real, self._psi, self._radius)
        for i in range(self._num_landmarks):
            if SDF_real[i] <= 0:
                sensor_value[i] = mu_real[i] + torch.normal(mean=torch.zeros(1, 2), std=torch.sqrt(self._V))
                # print(sensor_value[i])
            else:
                sensor_value[i] = self._mu_predict[i].flatten()
                # print(sensor_value[i])

        info_mat = torch.diag(self._info.flatten())
        R_mat = torch.eye(self._num_landmarks * 2) * self._V[0]  # sensor uncertainty covariance matrix
        S_mat = torch.inverse(info_mat) + R_mat
        kalman_gain = torch.inverse(info_mat) @ torch.inverse(S_mat)

        self._mu_update = torch.reshape(self._mu_predict.flatten() +
                                        (kalman_gain @ (sensor_value.flatten() -
                                                        (self._mu_predict).flatten())).flatten(), (self._num_landmarks, 2))

        q_update = torch.vstack(
            ((self._mu_update[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.sin(x[2]),
             (x[0] - self._mu_update[:, 0]) * torch.sin(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.cos(x[2]))).T
        SDF_update = triangle_SDF(q_update, self._psi, self._radius)
        M = (1 - phi(SDF_update, self._kappa))[:, None] * self._inv_V.repeat(self._num_landmarks, 1)
        # Assuming A = I:
        self._info = self._info + M

    # def update_policy(self, debug=False):
    #     self._policy_optimizer.zero_grad()
    #
    #     if debug:
    #         param_list = []
    #         grad_power = 0
    #         for i, p in enumerate(self._policy.parameters()):
    #             param_list.append(p.data.detach().clone())
    #             if p.grad is not None:
    #                 grad_power += (p.grad**2).sum()
    #             else:
    #                 grad_power += 0
    #
    #         print("Gradient power before backward: {}".format(grad_power))
    #
    #     reward = - torch.sum(torch.log(self._info))
    #     reward.backward()
    #     self._policy_optimizer.step()
    #
    #     if debug:
    #         grad_power = 0
    #         total_param_ssd = 0
    #         for i, p in enumerate(self._policy.parameters()):
    #             if p.grad is not None:
    #                 grad_power += (p.grad ** 2).sum()
    #             else:
    #                 grad_power += 0
    #             total_param_ssd += ((param_list[i] - p.data) ** 2).sum()
    #
    #         print("Gradient power after backward: {}".format(grad_power))
    #         print("SSD of weights after applying the gradient: {}".format(total_param_ssd))
    #
    #     return -reward.item()

    def set_policy_grad_to_zero(self):
        self._policy_optimizer.zero_grad()

    def update_policy_grad(self, train=True):
        reward = - torch.sum(torch.log(self._info))
        if train == True:
            reward.backward()
        return -reward.item()

    # def update_policy_grad(self, mu, x):
    #     reward = ((x[:2] - mu)**2).sum()
    #     reward.backward()
    #     return reward.item()

    def policy_step(self, debug=False):
        if debug:
            param_list = []
            for i, p in enumerate(self._policy.parameters()):
                param_list.append(p.data.detach().clone())

        self._policy_optimizer.step()

        if debug:
            total_param_rssd = 0
            grad_power = 0
            for i, p in enumerate(self._policy.parameters()):
                if p.grad is not None:
                    grad_power += (p.grad ** 2).sum()
                else:
                    grad_power += 0
                total_param_rssd += ((param_list[i] - p.data) ** 2).sum().sqrt()

            print("Gradient power after backward: {}".format(grad_power))
            print("RSSD of weights after applying the gradient: {}".format(total_param_rssd))

    def get_policy_state_dict(self):
        return self._policy.state_dict()

    def load_policy_state_dict(self, load_model):
        self._policy.load_state_dict(torch.load(load_model))


class ModelBasedAgentAtt:

    def __init__(self, max_num_landmarks, init_info, A, B, W, radius, psi, kappa, V, lr):
        self._init_info = init_info
        self._info = None

        self._max_num_landmarks = max_num_landmarks
        self._A = A
        self._B = B
        self._W = W
        self._psi = psi
        self._radius = radius
        self._kappa = kappa
        self._V = V
        self._inv_V = V ** (-1)

        # input_dim = num_landmarks * 4 + 3
        input_dim = max_num_landmarks * 5 + 3
        self._policy = PolicyNetAtt(input_dim=input_dim)

        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

    def reset_agent_info(self):
        self._info = self._init_info * torch.ones((self._num_landmarks, 2))

    def reset_estimate_mu(self, mu_real):
        self._num_landmarks = mu_real.size()[0]
        self._mu_update = mu_real + torch.normal(mean=torch.zeros(self._num_landmarks, 2), std=torch.sqrt(self._V))  # with the shape of (num_landmarks, 2)
        self._padding = torch.zeros(2 * (self._max_num_landmarks - self._num_landmarks))
        self._mask = torch.tensor([True] * self._num_landmarks + [False] * (self._max_num_landmarks - self._num_landmarks))

    def eval_policy(self):
        self._policy.eval()

    def train_policy(self):
        self._policy.train()

    def plan(self, v, x):
        # self._mu_predict = torch.clip(landmark_motion(self._mu_update, v, self._A, self._B),
        #                               min=-tensor([self._num_landmarks, self._num_landmarks]),
        #                               max=tensor([self._num_landmarks, self._num_landmarks]))
        self._mu_predict = landmark_motion(self._mu_update, v, self._A, self._B)
        self._info = (self._info**(-1) + self._W)**(-1)

        q_predict = torch.vstack(((self._mu_predict[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - self._mu_predict[:, 0]) * torch.sin(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.cos(x[2]))).T

        # net_input = torch.hstack((x, self._info.flatten(), next_mu.flatten()))

        agent_pos_local = torch.zeros(3)
        net_input = torch.hstack((agent_pos_local, self._info.flatten(),
                                  self._padding, q_predict.flatten(), self._padding, self._mask))
        # net_input = q.flatten()
        action = self._policy.forward(net_input)
        return action

    def update_info_mu(self, mu_real, x):
        q_real = torch.vstack(((mu_real[:, 0] - x[0]) * torch.cos(x[2]) + (mu_real[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - mu_real[:, 0]) * torch.sin(x[2]) + (mu_real[:, 1] - x[1]) * torch.cos(x[2]))).T
        sensor_value = torch.zeros(self._num_landmarks, 2)
        SDF_real = triangle_SDF(q_real, self._psi, self._radius)
        for i in range(self._num_landmarks):
            if SDF_real[i] <= 0:
                sensor_value[i] = mu_real[i] + torch.normal(mean=torch.zeros(1, 2), std=torch.sqrt(self._V))
                # print(sensor_value[i])
            else:
                sensor_value[i] = self._mu_predict[i].flatten()
                # print(sensor_value[i])

        info_mat = torch.diag(self._info.flatten())
        R_mat = torch.eye(self._num_landmarks * 2) * self._V[0]  # sensor uncertainty covariance matrix
        S_mat = torch.inverse(info_mat) + R_mat
        kalman_gain = torch.inverse(info_mat) @ torch.inverse(S_mat)

        self._mu_update = torch.reshape(self._mu_predict.flatten() +
                                        (kalman_gain @ (sensor_value.flatten() -
                                                        (self._mu_predict).flatten())).flatten(), (self._num_landmarks, 2))

        q_update = torch.vstack(
            ((self._mu_update[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.sin(x[2]),
             (x[0] - self._mu_update[:, 0]) * torch.sin(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.cos(x[2]))).T
        SDF_update = triangle_SDF(q_update, self._psi, self._radius)
        M = (1 - phi(SDF_update, self._kappa))[:, None] * self._inv_V.repeat(self._num_landmarks, 1)
        # Assuming A = I:
        self._info = self._info + M

    def set_policy_grad_to_zero(self):
        self._policy_optimizer.zero_grad()

    def update_policy_grad(self, train=True):
        reward = - torch.sum(torch.log(self._info))
        if train == True:
            reward.backward()
        return -reward.item()

    def policy_step(self, debug=False):
        if debug:
            param_list = []
            for i, p in enumerate(self._policy.parameters()):
                param_list.append(p.data.detach().clone())

        self._policy_optimizer.step()

        if debug:
            total_param_rssd = 0
            grad_power = 0
            for i, p in enumerate(self._policy.parameters()):
                if p.grad is not None:
                    grad_power += (p.grad ** 2).sum()
                else:
                    grad_power += 0
                total_param_rssd += ((param_list[i] - p.data) ** 2).sum().sqrt()

            print("Gradient power after backward: {}".format(grad_power))
            print("RSSD of weights after applying the gradient: {}".format(total_param_rssd))

    def get_policy_state_dict(self):
        return self._policy.state_dict()

    def load_policy_state_dict(self, load_model):
        self._policy.load_state_dict(torch.load(load_model))

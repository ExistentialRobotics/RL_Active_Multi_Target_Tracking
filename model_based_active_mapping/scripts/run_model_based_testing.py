import os, sys, yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.manual_seed(0)
import argparse

from torch import tensor
from envs.simple_env import SimpleEnv, SimpleEnvAtt
from agents.model_based_agent import ModelBasedAgent, ModelBasedAgentAtt

parser = argparse.ArgumentParser(description='model-based mapping')
parser.add_argument('--network-type', type=int, default=1, help='by default, it should attention block,'
                                                                'otherwise, it would be MLP')
args = parser.parse_args()

def run_model_based_testing(params_filename):
    assert os.path.exists(params_filename)
    with open(os.path.join(params_filename)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    max_num_landmarks = params['max_num_landmarks']
    num_landmarks = params['num_landmarks']
    horizon = params['horizon']
    env_width = params['env_width']
    env_height = params['env_height']
    tau = params['tau']

    A = torch.zeros((2, 2))
    A[0, 0] = params['motion']['A']['_1']
    A[1, 1] = params['motion']['A']['_2']

    B = torch.zeros((2, 2))
    B[0, 0] = params['motion']['B']['_1']
    B[1, 1] = params['motion']['B']['_2']

    W = torch.zeros(2)
    W[0] = params['motion']['W']['_1']
    W[1] = params['motion']['W']['_2']

    landmark_motion_scale = params['motion']['landmark_motion_scale']

    init_info = params['init_info']

    radius = params['FoV']['radius']
    psi = tensor([params['FoV']['psi']])
    kappa = params['FoV']['kappa']

    V = torch.zeros(2)
    V[0] = params['FoV']['V']['_1']
    V[1] = params['FoV']['V']['_2']

    lr = params['lr']
    max_epoch = params['max_epoch']
    batch_size = params['batch_size']
    num_test_trials = params['num_test_trials']

    if args.network_type == 1:
        env = SimpleEnvAtt(max_num_landmarks=max_num_landmarks, horizon=horizon, tau=tau,
                           A=A, B=B, V=V, W=W, landmark_motion_scale=landmark_motion_scale, psi=psi, radius=radius)
        agent = ModelBasedAgentAtt(max_num_landmarks=max_num_landmarks, init_info=init_info, A=A, B=B, W=W,
                                   radius=radius, psi=psi, kappa=kappa, V=V, lr=lr)
    else:
        env = SimpleEnv(num_landmarks=num_landmarks, horizon=horizon, width=env_width, height=env_height, tau=tau,
                        A=A, B=B, V=V, W=W, landmark_motion_scale=landmark_motion_scale, psi=psi, radius=radius)
        agent = ModelBasedAgent(num_landmarks=num_landmarks, init_info=init_info, A=A, B=B, W=W,
                                radius=radius, psi=psi, kappa=kappa, V=V, lr=lr)

    agent.load_policy_state_dict('./checkpoints/best_model.pth')

    agent.eval_policy()
    for i in range(num_test_trials):
        mu_real, v, x, done = env.reset()
        num_landmarks = mu_real.size()[0]
        agent.reset_estimate_mu(mu_real)
        agent.reset_agent_info()
        env.render()
        while not done:
            action = agent.plan(v, x)
            mu_real, v, x, done = env.step(action)
            agent.update_info_mu(mu_real, x)
            env.render()

        reward = agent.update_policy_grad(False) / num_landmarks
        print("num_landmark:", num_landmarks, "reward:", reward)


if __name__ == '__main__':
    # torch.manual_seed(0)
    # torch.autograd.set_detect_anomaly(True)
    run_model_based_testing(params_filename=os.path.join(os.path.abspath(os.path.join("", os.pardir)),
                                                          "params/params_compare.yaml"))

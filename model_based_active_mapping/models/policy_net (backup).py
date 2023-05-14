import torch

from torch import nn


class PolicyNet(nn.Module):

    def __init__(self,
                 input_dim: int,
                 policy_dim: int = 2):

        super(PolicyNet, self).__init__()

        self.num_landmark = int((input_dim - 3) / 4)

        self.agent_pos_fc1_pi = nn.Linear(3, 32)
        self.agent_pos_fc2_pi = nn.Linear(32, 32)
        self.landmark_fc1_pi = nn.Linear(4, 64)
        self.landmark_fc2_pi = nn.Linear(64, 32)
        self.info_fc1_pi = nn.Linear(64, 64)
        self.action_fc1_pi = nn.Linear(64, policy_dim)
        # self.action_fc2_pi = nn.Linear(64, policy_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if len(observation.size()) == 1:
            observation = observation[None, :]

        # compute the policy
        # embeddings of agent's position
        agent_pos_embedding = self.relu(self.agent_pos_fc1_pi(observation[:, :3]))
        agent_pos_embedding = self.relu(self.agent_pos_fc2_pi(agent_pos_embedding))

        # embeddings of landmarks
        info_vector = observation[:, 3: 3 + 2 * self.num_landmark]
        estimated_landmark_pos = observation[:, 3 + 2 * self.num_landmark:]
        landmark_info = torch.cat((estimated_landmark_pos.reshape(observation.size()[0], self.num_landmark, 2),
                                   info_vector.reshape(observation.size()[0], self.num_landmark, 2)), 2)
        landmark_embedding = self.relu(self.landmark_fc1_pi(landmark_info))
        landmark_embedding = self.relu(self.landmark_fc2_pi(landmark_embedding))

        # attention
        landmark_embedding_tr = torch.transpose(landmark_embedding, 1, 2)
        attention = torch.matmul(agent_pos_embedding.unsqueeze(1), landmark_embedding_tr) / 4

        att = self.softmax(attention)
        landmark_embedding_att = self.relu((torch.matmul(att, torch.transpose(landmark_embedding_tr, 1, 2)).squeeze(1)))

        info_embedding = self.relu(self.info_fc1_pi(torch.cat((agent_pos_embedding, landmark_embedding_att), 1)))
        action = self.tanh(self.action_fc1_pi(info_embedding))
        # action = self.action_fc2_pi(action)

        if action.size()[0] == 1:
            action = action.flatten()

        scaled_action = torch.hstack((action[0] * 3, action[1] * 0.1))

        return scaled_action

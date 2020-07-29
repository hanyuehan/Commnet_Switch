import torch.nn as nn
import torch
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.n_agents = n_agents
        # actor
        self.action_layer = CommNetWork_Actor(state_dim, action_dim, n_agents)
        # critic
        self.value_layer = CommNetWork_Critic(state_dim, n_agents)

    def act(self, state, memory):
        state = torch.Tensor(state).to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.cpu().numpy()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        action_probs = action_probs.view(-1, self.n_agents, self.action_dim )
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state, self.n_agents)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class CommNetWork_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(CommNetWork_Actor, self).__init__()
        self.rnn_hidden_dim = 256
        self.n_agents = n_agents
        self.n_actions = action_dim
        self.cuda = True
        self.input_shape = state_dim
        self.encoding = nn.Linear(self.input_shape, self.rnn_hidden_dim)
        self.f_obs = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.f_comm = nn.GRUCell(self.rnn_hidden_dim,
                                 self.rnn_hidden_dim)  # nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.decoding0 = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.decoding = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def forward(self, obs):
        size = obs.view(-1, self.n_agents, self.input_shape).shape
        size0 = size[0]

        obs_encoding = torch.relu(self.encoding(obs.view(size0 * self.n_agents,
                                                         self.input_shape)))  # .contiguous()  # .reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)

        h_out = self.f_obs(obs_encoding)

        for k in range(2):
            if k == 0:
                h = h_out
                c = torch.zeros_like(h)
            else:
                h = h.reshape(-1, self.n_agents, self.rnn_hidden_dim)

                c = h.reshape(-1, 1, self.n_agents * self.rnn_hidden_dim)
                c = c.repeat(1, self.n_agents, 1)
                mask = (1 - torch.eye(self.n_agents))
                mask = mask.view(-1, 1).repeat(1, self.rnn_hidden_dim).view(self.n_agents, -1)
                if self.cuda:
                    mask = mask.cuda()
                c = c * mask.unsqueeze(0)
                c = c.reshape(-1, self.n_agents, self.n_agents, self.rnn_hidden_dim)
                c = c.mean(dim=-2)
                h = h.reshape(-1, self.rnn_hidden_dim)
                c = c.reshape(-1, self.rnn_hidden_dim)
            h = self.f_comm(c, h)

        weights = torch.relu(self.decoding0(h))
        weights = torch.tanh(self.decoding(weights))
        # print(weights)
        weights = torch.nn.functional.softmax(weights)
        return weights


class CommNetWork_Critic(nn.Module):
    def __init__(self, state_dim, n_agents):
        super(CommNetWork_Critic, self).__init__()
        self.rnn_hidden_dim = 256
        self.n_agents = n_agents
        self.cuda = True
        self.input_shape = state_dim
        self.encoding = nn.Linear(self.input_shape, self.rnn_hidden_dim)
        self.f_obs = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.f_comm = nn.GRUCell(self.rnn_hidden_dim,
                                 self.rnn_hidden_dim)  # nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.decoding = nn.Linear(self.rnn_hidden_dim, 1)

    def forward(self, obs, act):
        obs = obs.view(-1, self.n_agents, self.input_shape).cuda()
        size0 = obs.shape[0]
        obs_encoding = torch.relu(self.encoding(obs.view(size0 * self.n_agents, self.input_shape)))

        h_out = self.f_obs(obs_encoding)

        for k in range(2):
            if k == 0:
                h = h_out
                c = torch.zeros_like(h)
            else:
                h = h.reshape(-1, self.n_agents, self.rnn_hidden_dim)

                c = h.reshape(-1, 1, self.n_agents * self.rnn_hidden_dim)
                c = c.repeat(1, self.n_agents, 1)
                mask = (1 - torch.eye(self.n_agents))
                mask = mask.view(-1, 1).repeat(1, self.rnn_hidden_dim).view(self.n_agents, -1)
                if self.cuda:
                    mask = mask.cuda()
                c = c * mask.unsqueeze(0)
                c = c.reshape(-1, self.n_agents, self.n_agents, self.rnn_hidden_dim)
                c = c.mean(dim=-2)
                h = h.reshape(-1, self.rnn_hidden_dim)
                c = c.reshape(-1, self.rnn_hidden_dim)
            h = self.f_comm(c, h)

        weights = self.decoding(h).view(size0, self.n_agents, -1)
        return weights

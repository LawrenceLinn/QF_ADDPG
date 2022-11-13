import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tools.replay_buffer import ReplayBuffer
from torch.autograd import Variable


class Actor(nn.Module):
    def __init__(self, product_num, win_size):
        super(Actor, self).__init__()
        self.short_available = 1
        self.TimeEncoder = nn.GRU(16,16,batch_first=True,bidirectional=False,dropout=0.3)
        self.linear1 = nn.Linear(448, 64)
        self.posEnc = nn.Linear(16,64)
        self.linear2 = nn.Linear(64, 16)


    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.posEnc.weight.data.uniform_(*hidden_init(self.posEnc))
        self.linear2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):

        pos = state[2]
        adj = state[1]
        state = state[0]#b*120*16
        code1 = self.timeEnc(state)[0]
        out1 = code1[-1]
        code2 = F.relu(code1).view(code1.size(0),-1)
        # Flatten
        code3 = F.relu(self.linear1(code2))
        poscode = F.relu(self.posEnc(pos))
        code4 = F.relu(self.linear2(torch.add(code3,poscode)))
        out2 = torch.tanh(code4) if self.short_available else torch.softmax(code4,dim=1)

        return F.sigmoid(out1), out2

# Define Critic network
class Critic(nn.Module):
    def __init__(self, product_num, win_size):
        super(Critic, self).__init__()
        self.TimeEncoder = nn.GRU(16,16,batch_first=True,bidirectional=False,dropout=0.3)
        self.linear1 = nn.Linear(448, 64)
        self.posEnc = nn.Linear(16, 64)
        self.linear2 = nn.Linear(64, 64)
        self.actEnc = nn.Linear(16, 64)
        self.QDec = nn.Linear(64,1)

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.posEnc.weight.data.uniform_(*hidden_init(self.posEnc))
        self.actEnc.weight.data.uniform_(*hidden_init(self.actEnc))
        self.QDec.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        pos = state[2]
        adj = state[1]
        state = state[0]

        code1 = self.timeEnc(state)[0]

        code2 = F.relu(code1).view(code1.size(0), -1)
        # Flatten
        code3 = F.relu(self.linear1(code2))
        poscode = F.relu(self.posEnc(pos))
        code4 = torch.add(code3, poscode)
        actcode = F.relu(self.posEnc(action))
        code5 = torch.add(code4,actcode)
        out = self.QDec(code5)
        return out

def hidden_init(layer):
    # Initialize the parameter of hidden layer
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


# The DDPG agent
class DDPG(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.critic_target = Critic(0).to(self.device)

        self.actor_layer1 = optim.Adam(self.actor.TimeEncoder.parameters(),lr = 1e-3)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.01)

        self.buffer = ReplayBuffer(self.capacity)
        self.actor.reset_parameters()
        self.critic.reset_parameters()
        self.actor_target.reset_parameters()
        self.critic_target.reset_parameters()

        self.trend_loss = 0
        self.actor_loss = 0
        self.critic_loss = 0

    def act(self, s0):
        a0 = self.actor(s0).squeeze(0).detach().cpu().numpy()
        return a0

    def learn(self):
        if self.buffer.count < self.batch_size:
            return
        s0_batch, a0_batch, r1_batch, t1_batch, s1_batch = self.buffer.sample_batch(self.batch_size)

        def critic_learn():
            a1_batch = self.actor(s1_batch)
            Q_target = r1_batch + (t1_batch == False).unsqueeze(1) * self.gamma * self.critic_target(
                s1_batch, torch.tensor(a1_batch, requires_grad=True).to(self.device)).detach().to(self.device)

            Q_target = Variable(Q_target, requires_grad=True)

            Q_true = self.critic(s0_batch, a0_batch).to(self.device)

            self.critic_loss = F.mse_loss(Q_target, Q_true).to(self.device)
            self.critic_optim.zero_grad()
            self.critic_loss.backward()
            self.critic_optim.step()

        def actor_learn():
            a0_ = self.actor(s0_batch).to(self.device)
            self.actor_loss = -torch.mean(self.critic(s0_batch, a0_)).to(self.device)
            self.actor_optim.zero_grad()
            self.actor_loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)



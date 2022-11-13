'''
Author: Runsehng Lin
Inspired by Qiu
'''
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from tools.replay_buffer import ReplayBuffer
from tools.OUNoise import OrnsteinUhlenbeckActionNoise
from tensorboardX import SummaryWriter
from tqdm import tqdm
import config



# Define actor network--CNN
class Actor(nn.Module):
    def __init__(self,**actor_params):
        super(Actor, self).__init__()
        for key, value in actor_params.items():
            setattr(self, key, value)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3,1),
            # stride = (1,3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(self.win_size - 3,1),
            # stride = (1, win_size-2)
        )
        self.linear1 = nn.Linear((self.product_num + 1) * 1 * 32, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, self.product_num + 1)

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        conv1_out = self.conv1(state.unsqueeze(1))
        conv1_out = F.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = F.relu(conv2_out)
        # Flatten
        conv2_out = conv2_out.view(conv2_out.size(0), -1)
        fc1_out = self.linear1(conv2_out)
        fc1_out = F.relu(fc1_out)
        fc2_out = self.linear2(fc1_out)
        fc2_out = F.relu(fc2_out)
        fc3_out = self.linear3(fc2_out)
        fc3_out = F.softmax(fc3_out, dim=1)

        return fc3_out


# Define policy gradient actor network--LSTM
class Policy(nn.Module):
    def __init__(self,**policy_params):
        '''
        product_num
        win_size
        action_size
        :param critic_param:
        '''
        super(Policy, self).__init__()
        for key, value in policy_params.items():
            setattr(self, key, value)
        self.gru = nn.GRU(10,10,32,batch_first=True)

        self.linear1 = nn.Linear(self.win_size-1, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, self.action_size)

        # Define the  vars for recording log prob and reawrd
        self.saved_log_probs = []
        self.rewards = []

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        gru_out, _ = self.gru(state)#10,1,32
        batch_n, win_s, hidden_s = gru_out.shape
        gru_out = gru_out.view(hidden_s,-1)
        fc1_out = self.linear1(gru_out)
        fc1_out = F.relu(fc1_out)
        fc2_out = self.linear2(fc1_out)
        fc2_out = F.relu(fc2_out)
        fc3_out = self.linear3(fc2_out)
        fc3_out = F.softmax(fc3_out, dim=1)#1,2

        return fc3_out

# Define Critic network--CNN
class Critic(nn.Module):
    def __init__(self, **critic_params):
        super(Critic, self).__init__()
        for key, value in critic_params.items():
            setattr(self, key, value)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 1),
            # stride = (1,3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(self.win_size - 3, 1),
            # stride = (1, win_size-2)
        )
        self.linear1 = nn.Linear((self.product_num + 1) * 1 * 32, 64)
        self.linear2 = nn.Linear((self.product_num + 1), 64)
        self.linear3 = nn.Linear(64, 1)

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        # Observation channel
        conv1_out = self.conv1(state.unsqueeze(1))
        conv1_out = F.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = F.relu(conv2_out)
        # Flatten
        conv2_out = conv2_out.view(conv2_out.size(0), -1)
        fc1_out = self.linear1(conv2_out)
        # Action channel
        fc2_out = self.linear2(action)
        obs_plus_ac = torch.add(fc1_out, fc2_out)
        obs_plus_ac = F.relu(obs_plus_ac)
        fc3_out = self.linear3(obs_plus_ac)

        return fc3_out


def hidden_init(layer):
    # Initialize the parameter of hidden layer
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class QF_HRL(nn.Module):
    def __init__(self, **model_params):
        '''
        device
        agent_params
        noise
        buffer
        episode
        batch_size
        logdir
        args
        :param kwargs:
        '''
        super(QF_HRL, self).__init__()
        for key, value in model_params.items():
            setattr(self, key, value)
        self.actor = Actor(**self.agent_params).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = Critic(**self.agent_params).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.policy = Policy(**self.agent_params).to(self.device)

        self.buffer = ReplayBuffer(self.capacity)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-4)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.actor.reset_parameters()
        self.critic.reset_parameters()
        self.actor_target.reset_parameters()
        self.critic_target.reset_parameters()

        self.actor_loss = 0
        self.critic_loss = 0
        self.policy_loss = 0
        self.actor_noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.agent_params['product_num']+1))

    def act(self, state):
        action = self.actor(state.to(self.device)).squeeze(0).cpu().detach().numpy()
        return action

    def policy_act(self, state):
        # state = torch.tensor(state, dtype=torch.float) # 1,1,10,3
        # Get the probability distribution
        probs = self.policy(state.to(self.device)).cpu()
        m = Categorical(probs)
        # Sample action from the distribution
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action

    def online_learn(self):
        if self.buffer.count < 2000:
            return

        s0_batch, a0_batch, r1_batch, t1_batch, s1_batch = self.buffer.sample_batch(self.batch_size)

        def critic_learn():
            a1_batch = self.actor_target(s1_batch)
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

    def policy_learn(self):
        R = 0
        p_loss = []
        returns = []

        # Reversed Traversal and calculate cumulative rewards for t to T
        ''''''
        for r in self.policy.rewards[::-1]:
            R = r + 0.95 * R  # R: culumative rewards for t to T
            returns.insert(0, R)  # Evaluate the R and keep original order
        returns = torch.tensor(returns).to(self.device)
        # Normalized returns
        returns = (returns - returns.mean()) / (returns.mean() + 1e-8)

        # After one episode, update once
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            # Actual loss definition:
            p_loss.append(-log_prob * R)
        self.policy_optim.zero_grad()
        self.policy_loss = torch.cat(p_loss).sum()
        self.policy_loss.backward()
        self.policy_optim.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def test(self,env,episode,loader = None):
        s0 = env.reset()
        assets_name = ['Cash']+env.dataloader.product_name
        test_reward = 0
        for test_step in range(env.dataloader.max_step):
            a0 = self.act(s0)
            s1, r1, test_done, info= env.step(a0)
            # self.buffer.add(s0, a0, r1, test_done, s1)
            test_reward += r1
            s0 = s1
            if self.record:
                self.writer.add_scalar(f'Test return/{episode + 1}', env.portfolio.p0, test_step)
                assest_weight = dict(zip(assets_name, env.portfolio.w0))
                self.writer.add_scalars(f'Test position/{episode + 1}', assest_weight, test_step)
        env.render()

    def train(self,env,env_test,loader = None):
        '''
        :param env:
        :param env_test:
        :param loader:
        :return:
        '''
        record_gap = 10
        learn_gap = 1
        assets_name = ['Cash']+env.dataloader.product_name
        logdir = f'../results/{self.result_path}'
        config.reset_logdir(logdir)
        if self.record:
            self.writer = SummaryWriter(logdir)
        '''START'''
        total_step = 0
        absreward_list = [0]
        for episode in tqdm(range(self.episodes),desc=f'>>>Training with {self.device}, result saved in {self.result_path}<<<\n\t'):
            '''Training'''
            s0 = env.reset()
            train_reward = 0
            for train_step in tqdm(range(env.dataloader.max_step), desc=f'Training episode {episode}'):
                total_step += 1
                # Add noise
                action = self.act(s0)

                '''visualize network output'''
                if train_step % record_gap == 0 and self.record and episode % 1 == 0:
                    assest_weight = dict(zip(assets_name, action))
                    self.writer.add_scalars(f'Train action/{episode + 1}', assest_weight, train_step)

                a0 = action + self.actor_noise()
                p0 = self.policy_act(s0)
                s1, r1, policy_reward, train_done, info = env.step(a0,p0)
                self.policy.rewards.append(policy_reward)
                '''visualize action with noise'''

                if train_step % record_gap == 0 and self.record and episode % 1 == 0:
                    assest_weight = dict(zip(assets_name, env.portfolio.w0))
                    self.writer.add_scalars(f'Train position/{episode + 1}', assest_weight, train_step)

                absreward_list.append(abs(r1-env.portfolio.entropy_loss))
                if abs(r1-env.portfolio.entropy_loss)>=(np.mean(absreward_list)):
                    self.buffer.add(s0, a0, r1, train_done, s1)
                train_reward += r1
                s0 = s1

                '''ddpg learn and visualize loss'''
                if total_step>2000:
                    if total_step % learn_gap == 0:
                        self.online_learn()

                if train_step % record_gap == 0 and self.record:
                    self.writer.add_scalar('Loss/Actor', self.actor_loss, total_step)
                    self.writer.add_scalar('Loss/Critic', self.critic_loss, total_step)

                    '''visualize the return of training set'''
                    self.writer.add_scalar(f'Train return/{episode + 1}', env.portfolio.p0, train_step)

            self.policy_learn()
            if self.record:
                self.writer.add_scalar('Loss/Policy', self.policy_loss, episode)
            env.render()
            print(self.buffer.count)
            # self.test(env_test,episode)




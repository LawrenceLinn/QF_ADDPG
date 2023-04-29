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
from tqdm import tqdm
import config
from multiprocessing import Process,Queue,set_start_method,get_context
import itertools

def hidden_init(layer):
    # Initialize the parameter of hidden layer
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Unsqueeze(nn.Module):
    def __init__(self, item_index):
        super(Unsqueeze, self).__init__()
        self._name = 'unsqueeze'
        self.item_index = item_index
    def forward(self, inputs):
        return inputs.unsqueeze(self.item_index)

class HRLNet(nn.Module):
    def __init__(self,**network_params):
        super(HRLNet, self).__init__()
        for key, value in network_params.items():
            setattr(self, key, value)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(self.win_size - 3, 1),
        )

        self.state_encoder = nn.Sequential(Unsqueeze(1),
                                       self.conv1,
                                       nn.ReLU(),
                                       self.conv2,
                                       nn.ReLU(),
                                       nn.Flatten())

        self.state_decoder = nn.Sequential(nn.Linear(self.product_num * 32, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                           nn.ReLU())

        self.action_decoder = nn.Sequential(nn.Linear(self.product_num * 32, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.product_num + 1),
                                            nn.Softmax(dim = 1))

        self.weight_encoder = nn.Sequential(nn.Linear(self.product_num+1,64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.ReLU())

        self.Q_decoder = nn.Sequential(nn.Linear(64,64),
                                        nn.ReLU(),
                                        nn.Linear(64, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1),
                                        nn.Tanh())

        self.policy_encoder = nn.GRU(self.product_num,self.product_num,4,batch_first=True)

        self.policy_decoder = nn.Sequential(nn.Linear(self.win_size-1,64),
                                    nn.ReLU(),
                                    nn.Linear(64,64),
                                    nn.ReLU(),
                                    nn.Linear(64,self.action_size),
                                    nn.Softmax(dim = 1))

        self.saved_log_probs = []
        self.rewards = []


    def actor_forward(self,state):
        state_coding = self.state_encoder(state)
        action = self.action_decoder(state_coding)
        return action

    def critic_forward(self,state,action):
        state_coding = self.state_encoder(state)
        state_coding2 = self.state_decoder(state_coding)
        weight_coding = self.weight_encoder(action)
        Q = self.Q_decoder(torch.add(state_coding2,weight_coding))
        return Q

    def policy_forward(self,state):
        state_coding = self.policy_encoder(state)[0]
        policy_coding = self.policy_decoder(state_coding.view(state_coding.size(-1),-1))
        return policy_coding

class HRL_comp(nn.Module):
    def __init__(self,**model_params):
        super(HRL_comp, self).__init__()
        for key, value in model_params.items():
            setattr(self, key, value)

        self.Network = HRLNet(**self.network_params)
        self.target_Network = copy.deepcopy(self.Network)
        self.actor_optim = optim.Adam(itertools.chain(self.Network.state_encoder.parameters(),
                                                      self.Network.action_decoder.parameters()),lr = 1e-4)

        self.critic_optim = optim.Adam(itertools.chain(self.Network.state_encoder.parameters(),
                                                       self.Network.weight_encoder.parameters(),
                                                       self.Network.Q_decoder.parameters()),lr = 1e-3)

        self.policy_optim = optim.Adam(itertools.chain(self.Network.policy_encoder.parameters(),
                                                        self.Network.policy_decoder.parameters()),lr = 1e-4)
        self.buffer = ReplayBuffer(self.capacity)

        self.actor_loss = 0
        self.critic_loss = 0
        self.policy_loss = 0
        self.actor_noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.network_params['product_num']+1))
        self.y_i = [0]


    def act(self, state):
        action = self.Network.actor_forward(state.to(self.device)).squeeze(0).cpu().detach().numpy()
        return action

    def policy_act(self, state):
        probs = self.Network.policy_forward(state.to(self.device)).cpu()
        m = Categorical(probs)
        # Sample action from the distribution
        action = m.sample()
        self.Network.saved_log_probs.append(m.log_prob(action))
        return action

    def soft_update(self,net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def online_learn(self):
        if self.buffer.count < self.batch_size:
            return

        s0_batch, a0_batch, r1_batch, t1_batch, s1_batch = self.buffer.sample_batch(self.batch_size)

        target_q = self.target_Network.critic_forward(s1_batch, self.target_Network.actor_forward(s1_batch)).detach()

        self.y_i = []

        for k in range(self.batch_size):
            if t1_batch[k]:
                self.y_i.append((r1_batch[k]).numpy())
            else:
                self.y_i.append((r1_batch[k] + self.gamma * target_q[k]).numpy())
        def critic_learn():
            actual_q = self.Network.critic_forward(s0_batch, a0_batch)
            target_Q = torch.tensor(np.array(self.y_i).reshape(self.batch_size,1).astype('float32'), dtype=torch.float32)
            target_Q = Variable(target_Q, requires_grad=True)
            self.critic_loss = F.mse_loss(actual_q, target_Q)
            self.critic_optim.zero_grad()
            self.critic_loss.backward()
            self.critic_optim.step()

        def actor_learn():
            a0_ = self.Network.actor_forward(s0_batch).to(self.device)
            self.actor_loss = -torch.mean(self.Network.critic_forward(s0_batch, a0_)).to(self.device)
            self.actor_optim.zero_grad()
            self.actor_loss.backward()
            self.actor_optim.step()


        critic_learn()
        actor_learn()
        self.soft_update(self.target_Network, self.Network, self.tau)

    def policy_learn(self):
        R = 0
        p_loss = []
        returns = []
        for r in self.Network.rewards[::-1]:
            R = r + 0.95 * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.mean() + 1e-8)
        for log_prob, R in zip(self.Network.saved_log_probs, returns):
            p_loss.append(-log_prob * R)
        self.policy_optim.zero_grad()
        self.policy_loss = torch.cat(p_loss).sum()
        self.policy_loss.backward()
        self.policy_optim.step()
        del self.Network.rewards[:]
        del self.Network.saved_log_probs[:]

    def test(self,env,episode):
        s0 = env.reset()
        # assets_name = ['Cash']+env.dataloader.product_name
        test_reward = 0
        for test_step in range(env.dataloader.max_step):
            a0 = self.act(s0)
            p0 = self.Network.policy_forward(s0.to(self.device)).cpu()
            p0 = torch.argmax(p0,dim = 1)
            s1, r1, pr1, test_done, info= env.step(a0,p0)
            test_reward += r1
            s0 = s1
        env.render()
        self.writer.add_scalar(f'Test return', env.portfolio.p0, episode)


    def train(self,env,env_test,loader = None):
        '''
        :param env:
        :param env_test:
        :param loader:
        :return:
        '''

        record_gap = 30
        learn_gap = 20
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
            max_Q = [0]
            train_reward = 0
            for train_step in tqdm(range(env.dataloader.max_step), desc=f'Training episode {episode}'):
                # Add noise
                action = self.act(s0)
                a0 = action + self.actor_noise()
                self.p0 = self.policy_act(s0)
                s1, r1, policy_reward, train_done, info = env.step(a0,self.p0)
                self.Network.rewards.append(policy_reward)
                self.buffer.add(s0, a0, r1, train_done, s1)
                train_reward += r1
                s0 = s1

                '''ddpg learn and visualize loss'''
                if episode>5:
                    total_step += 1
                    self.online_learn()

                self.writer.add_scalar('Loss/Actor', self.actor_loss, total_step)
                self.writer.add_scalar('Loss/Critic', self.critic_loss, total_step)


            self.policy_learn()
            print(f'Actor loss:{self.actor_loss:.5f}\n'
                  f'Critic loss:{self.critic_loss:.5f}\n'
                  f'Policy loss:{self.policy_loss:.5f}\n'
                  f'maxQ: {np.mean(max_Q)}')
            if self.record:
                self.writer.add_scalar('Loss/Policy', self.policy_loss, episode)
            env.render()
            print(self.buffer.count)
            self.test(env_test,episode)




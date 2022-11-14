"""
Source: https://github.com/vermouth1992/deep-learning-playground/blob/master/tensorflow/ddpg/replay_buffer.py
"""
from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        # self.prob = deque()

    def add(self, *transition):
        # experiment = (s, a, r, t, s2)

        if self.count < self.buffer_size:
            self.buffer.append(transition)
            # self.prob.append(abs(transition[2]))
            self.count += 1
        else:
            self.buffer.popleft()
            # self.prob.popleft()
            # self.prob.append(abs(transition[2]))
            self.buffer.append(transition)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            return

        # print(torch.softmax(self.prob))
        # print(np.random.choice(a=self.buffer, size=batch_size, replace=True, p=self.prob))
        batch = random.sample(self.buffer, batch_size)

        s0, a0, r1, t1, s1 = zip(*batch)
        obs_0 = []
        obs_1 = []
        for i in range(batch_size):
            obs_0.append(s0[i])
            obs_1.append(s1[i])

        obs_0 = torch.cat(obs_0,0)
        obs_1 = torch.cat(obs_1,0)

        s0_batch = obs_0
        a0_batch = torch.tensor(np.array(a0), dtype=torch.float)


        r1_batch = torch.tensor(np.array(r1), dtype=torch.float)
        t1_batch = torch.tensor(np.array(t1), dtype=torch.float)

        # s1_batch = (obs_1, adj_1, delta_pos_1)
        s1_batch = obs_1
        return s0_batch, a0_batch, r1_batch, t1_batch, s1_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
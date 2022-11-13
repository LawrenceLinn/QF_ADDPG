"""
Author: Runsheng LIN
Date: Sep 28 2022
"""
import random

import gym.spaces
import numpy as np
import pandas as pd
import torch

class stockPortfolioEnv(gym.Env):

    """
    df: 原始数据表格
    win_size: 输入窗口大小
    mode: 模式 option: train, validate, test
    trans_rate: 手续费 一般为0.001
    log: 记录器
    curr_step: 当前步数
    gap: 距离开始日期的间隔数
    corr: 产品回报之间的相关系数矩阵
    rf: 无风险收益率 一般为年化3%
    product_num: 产品数量
    value: 当前组合价值
    position：当前组合持仓
    train_val_test: 数据集比例
    short_available : 是否可以做空
    """

    def __init__(self, **kwargs):
        super(stockPortfolioEnv, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.log = {'step': [], 'weight': [], 'reward': [], 'rp': []}
        self.curr_step = 0
        self.value = np.array([1] + self.df.shape[1] * [0])
        self.position = np.array([1] + self.df.shape[1] * [0])
        self.date = self.df.index
        self.rf = 0.03/365/24
        self.riskfree = np.ones([self.win_size,1])*self.rf
        self.normalize_factor = 1000

        self.variance = pd.read_csv('./variance.csv',header=[0],index_col=[0])
        self.mean = pd.read_csv('./mean.csv',header=[0],index_col=[0])

        if self.mode == 'train':
            self.gap = self.win_size
            self.max_step = int(self.df.shape[0]*self.train_val_test[0])-self.win_size
        elif self.mode == 'val':
            self.gap = int(self.df.shape[0]*self.train_val_test[0])
            self.max_step = int(self.df.shape[0]*self.train_val_test[1])-self.win_size
        elif self.mode == 'test':
            self.gap = int(self.df.shape[0]*self.train_val_test[1])
            self.max_step = self.df.shape[0]-self.win_size
        else:
            exit()


    def step(self,action):
        """
        :param action: array,
        :return: reward and the state of next step
        state, rew, done
        """

        def action2position(self,action):
            action = np.clip(action, -self.short_available, 1)
            weight = action
            X = 1-sum(weight)
            weight = (weight+X/weight.shape[0]) if self.short_available else weight/(weight.sum() + 1e-8)
            weight[0] += np.clip(1 - weight.sum(), -self.short_available, 1)
            return weight

        # def action2position_short(action):#sum 不等于1
        #     action = np.clip(action,-1,1)


        def update_position(self):
            tem_pos = action2position(self,action)
            # 调仓
            delta_pos = tem_pos-self.position

            #如果调仓幅度小于震荡幅度，维持原有持仓
            reallocation = abs(delta_pos)>self.osc_range
            delta_pos *= reallocation
            delta_pos -= delta_pos.mean()
            tem_pos = self.position+delta_pos

            # 调仓产生的手续费百分率
            tc_r = abs(delta_pos[1:]).sum() * self.trans_cost
            # 当日的收益率~1
            ret = np.append(1 + self.rf, self.df.iloc[self.curr_step+self.win_size,:] + 1)
            #结算前组合价值
            tem_value = self.value.sum()*tem_pos*(1-tc_r)

            #结算后组合价值
            new_value = tem_value*ret
            #reward
            rew = np.log(new_value.sum()/self.value.sum())
            #更新
            self.value = new_value
            self.position = self.value/self.value.sum()
            return rew,delta_pos

        rew,delta_pos = update_position(self)

        #步数更新
        self.curr_step += 1
        self.curr_date = self.df.index[self.curr_step+self.win_size-1]
        #下一步状态
        obs = self.df.iloc[self.curr_step:self.curr_step + self.win_size, :]
        obs.insert(0,'Cash',self.riskfree)
        adj = torch.tensor(obs.cov().values,dtype=torch.float32)
        delta_pos = torch.tensor(delta_pos,dtype=torch.float32)
        p1 = torch.tensor(self.position,dtype=torch.float32)
        obs = torch.tensor(obs.values,dtype=torch.float32).unsqueeze(0)*self.normalize_factor
        state = (obs, adj, p1)

        done = self.curr_step >= self.max_step

        return state, rew, done

    def reset(self,init_position = None):
        """
        重置环境
        :return: 第一步的state
        """
        #重置
        self.curr_step = self.gap-self.win_size
        self.curr_date = self.df.index[self.curr_step+self.win_size-1]

        #验证和测试的时候使用上一轮的持仓为初始持仓
        # if self.mode == 'train':
        self.value = np.array([1] + self.df.shape[1] * [0])
        self.position = np.array([1] + self.df.shape[1] * [0])
        # else:
        #     self.value = self.position = init_position

        #第一步状态
        obs = self.df.iloc[self.curr_step:self.curr_step + self.win_size, :]
        obs.insert(0,'Cash',self.riskfree)
        adj = torch.tensor(obs.cov().values,dtype=torch.float32)
        delta_pos = torch.tensor(np.array([0]*(self.df.shape[1]+1)),dtype=torch.float32)
        p0 = torch.tensor(np.array([0]*(self.df.shape[1]+1)),dtype=torch.float32)
        obs = torch.tensor(obs.values,dtype=torch.float32).unsqueeze(0)*self.normalize_factor
        state = (obs, adj, p0)
        return state

    def noise(self):
        mu = self.mean.loc[self.curr_date,:]
        var = self.variance.loc[self.curr_date,:]
        return random.gauss(mu,var)

    def report(self):
        print(self.curr_step)
        print(self.df.index[self.curr_step+self.win_size])
        print(self.value.sum())

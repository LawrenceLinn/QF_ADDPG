"""
@Author: Runsheng Lin
Inspired by Qiu.
"""
import numpy as np
import pandas as pd
import matplotlib as plt
import os
import datetime

eps = 1e-8
date_format = '%Y-%m-%d'
start_date = '2014-03-21'
end_date = '2020-10-14'
start_datetime = datetime.datetime.strptime(start_date, date_format)
end_datetime = datetime.datetime.strptime(end_date, date_format)

def date_to_index(date_string):
    # Transfer the date to index 0, 1, 2 ,3...
    return (datetime.datetime.strptime(date_string, date_format) - start_datetime).days

def index_to_date(index):
    # Transfer index back to date
    return (start_datetime + datetime.timedelta(index)).strftime(date_format)


class DataLoader(object):
    """
    data_type
    train_ratio
    win_size
    start_idx
    feature_num
    market_feature
    mode
    args
    :returns
    obs, done, next_obs
    """
    def __init__(self,**loader_params):
        for key, value in loader_params.items():
            setattr(self, key, value)
        self.product_name = []
        self.load_obs()
        self.date = self.df.index
        self.max_step = self.data.shape[0]-self.win_size-1
        self.normalize_factor = 100
        # self.start_date = datetime.datetime(self.df.index[0])
        # self.end_date = datetime.datetime(self.df.index[-1])
        assert self.gap <= self.win_size

    def date_to_index(self,date_string):
        # Transfer the date to index 0, 1, 2 ,3...
        return (datetime.datetime.strptime(date_string, date_format) - start_datetime).days

    def index_to_date(self,index):
        # Transfer index back to date
        return (start_datetime + datetime.timedelta(index)).strftime(date_format)


    def load_obs(self):
        data_root = f'../Data/{self.data_type}'
        filenames = os.listdir(data_root)
        dt_list = []
        for filepath in filenames:
            self.product_name.append(filepath[:-3])
            '''2048,6'''
            self.df = pd.read_csv(f'{data_root}/{filepath}',index_col=[0,1,2],header=[0]).dropna(axis=0)[self.market_feature].iloc[::-1]
            dt_list.append(np.array(self.df))
        '''2048,6,9'''
        dt_arr = np.array(dt_list).transpose(1,0,2)
        if self.mode == 'Train':
            self.data = dt_arr[:int(self.train_ratio * dt_arr.shape[0])]
            print(f'The shape of data is {self.data.shape}--{self.mode} mode')
        if self.mode == 'Test':
            self.data = dt_arr[int(self.train_ratio * dt_arr.shape[0]):]
            print(f'The shape of data is {self.data.shape}--{self.mode} mode')

    def step(self):
        self.cur_step+=self.gap
        obs = self.data[self.cur_step:self.cur_step + self.win_size].copy()
        next_obs = self.data[self.cur_step + self.win_size:self.cur_step + self.win_size + self.gap].copy()
        done = self.cur_step >= self.max_step
        return obs, done, next_obs

    def reset(self):
        """
        :return: obs (win_size,prod_num,feat_num) and next_obs
        """
        self.cur_step = 0
        obs = self.data[self.cur_step:self.cur_step + self.win_size].copy()
        next_obs = self.data[self.cur_step + self.win_size:self.cur_step + self.win_size+self.gap].copy()
        return obs, next_obs


if __name__ == '__main__':
    param = {
        'data_type':'frex',
        'win_size':10,
        'steps':1000,
        'train_ratio':0.8,
        'mode':'Train'
    }
    dl = DataLoader(**param)


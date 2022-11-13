import numpy as np
import pandas as pd
import gym.spaces
from tools.data import DataLoader
from Environment.portfolio import Portfolio
import torch


def sharpe(returns, freq=30, rfr=0):
    # The function that is used to caculate sharpe ratio
    return (np.sqrt(freq) * np.mean(returns - rfr + 1e-8)) / np.std(returns - rfr + 1e-8)

def max_drawdown(return_list):
    # The function that is used to calculate the max drawndom
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])
    return (return_list[j] - return_list[i]) / (return_list[j])

class marketEnv(gym.Env):
    def __init__(self,**env_param):
        """
        product_list,
        market_feature,
        feature_num,
        steps,
        window_length,
        mode,
        start_index=0,
        start_date
        portfolio_param:

        loader_param:
            gap
            data_type
            train_ratio
            step
            win_size
            start_idx
            feature_num
            market_feature
            mode
            args

        :param env_param:
        """
        super(marketEnv, self).__init__()
        for key, value in env_param.items():
            setattr(self, key, value)
        self.dataloader = DataLoader(**self.loader_params)
        self.portfolio = Portfolio(**self.portfolio_params)
        self.infos = []
        self.normalize_factor = 10000

    def step(self,action):

        def action2position(self,action):
            action = np.clip(action, -self.short_available, 1)
            # action = np.clip(action, 0, 1)
            weight = action
            X = 1-sum(weight)
            weight = (weight+X/weight.shape[0]) if self.short_available else weight/(weight.sum() + 1e-8)
            # weight = weight/(weight.sum() + 1e-8)
            weight[0] += np.clip(1 - weight.sum(), -self.short_available, 1)
            # weight[0] += np.clip(1 - weight.sum(), 0, 1)
            return weight

        weight = action2position(self,action)
        reward, info, done1 = self.portfolio.step(weight,self.y1)
        self.infos.append(info)

        obs, done2, next_obs = self.dataloader.step()
        state = np.log(obs[1:,:,3]/obs[:-1,:,3])  # win_size,prod_num
        cash = np.zeros((state.shape[0], 1))
        state = np.concatenate([cash,state],axis = 1)
        self.y1 = np.concatenate([np.ones([1,1]),next_obs[:,:,3]/obs[-1,:,3]],axis=1).squeeze()#next day return


        return torch.tensor(state,dtype=torch.float32).unsqueeze(0)*self.normalize_factor, reward, done1 or done2, info

    def reset(self):
        self.portfolio.reset()
        obs, next_obs = self.dataloader.reset()
        state = np.log(obs[1:,:,3]/obs[:-1,:,3])#win_size,prod_num
        a = obs[0,:,3]
        b = obs[-1,:,3]
        c = a/b
        self.y1 = np.concatenate([np.ones([1,1]),next_obs[:,:,3]/obs[-1,:,3]],axis=1).squeeze()
        cash = np.zeros((state.shape[0],1))
        state = np.concatenate([cash,state],axis = 1)

        return torch.tensor(state,dtype=torch.float32).unsqueeze(0)*self.normalize_factor

    def render(self):
        df_info = pd.DataFrame(self.infos)
        # df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        # df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.portfolio_value)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        print("\nMax drawdown", mdd)
        print("Sharpe ratio",sharpe_ratio)
        print("Final portfolio value", df_info["portfolio_value"].iloc[-1])

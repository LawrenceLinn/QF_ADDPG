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

    def step(self,action,policy):

        def action2position(action):
            action = np.clip(action, 0, 1)
            weight = action
            weight = weight/(weight.sum() + 1e-8)
            return weight


        policy_reward = np.ones([self.portfolio.product_num])
        weight = action2position(action)
        delta_weight = weight-self.portfolio.w0

        Open =  self.next_obs[0,:,0]
        High = self.next_obs[0,:,1]
        Low = self.next_obs[0,:,2]
        Close = self.next_obs[0, :, 3]
        QPLp1 =  self.next_obs[0,:,4]
        QPLn1 = self.next_obs[0, :, 5]

        for i in range(self.next_obs.shape[1]):
            if Low[i] < QPLn1[i]:
                if policy[i] == 0 and delta_weight[i + 1] < 0:
                    delta_weight[i + 1] *= 0.5
                    self.y1[i + 1] = QPLn1[i] / self.obs[-1, i, 3]
                    policy_reward[i] = Close[i] / QPLn1[i]
                elif policy[i] == 1 and delta_weight[i + 1] > 0:
                    delta_weight[i + 1] *= 0.5
                    self.y1[i + 1] = QPLn1[i] / self.obs[-1, i, 3]
                    policy_reward[i] = Close[i] / QPLn1[i]
                else:
                    pass
            else:
                if High[i] > QPLp1[i]:
                    if policy[i] == 0 and delta_weight[i + 1] < 0:
                        delta_weight[i + 1] *= 0.5
                        self.y1[i + 1] = QPLp1[i] / self.obs[-1, i, 3]
                        policy_reward[i] = Close[i] / QPLp1[i]
                    elif policy[i] == 1 and delta_weight[i + 1] > 0:
                        delta_weight[i + 1] *= 0.5
                        self.y1[i + 1] = QPLp1[i] / self.obs[-1, i, 3]
                        policy_reward[i] = Close[i] / QPLp1[i]
                else:
                    delta_weight[i] *= 0.5

        policy_weight = delta_weight - (weight - self.portfolio.w0)

        delta_weight[0] = -sum(delta_weight[1:])

        if self.portfolio.w0[0]+delta_weight[0]>0 and self.portfolio.w0[0]+delta_weight[0]<1:
            weight = self.portfolio.w0 + delta_weight
        elif self.portfolio.w0[0]+delta_weight[0]<0:
            weight = self.portfolio.w0 + delta_weight*abs(self.portfolio.w0[0]/(delta_weight[0]+1e-8))
        else:
            weight = self.portfolio.w0 + delta_weight * abs((1-self.portfolio.w0[0])/(delta_weight[0]+1e-8))

        weight = np.clip(weight,0,1)

        reward, info, done1 = self.portfolio.step(weight,self.y1)#TODO:放大
        self.infos.append(info)

        self.obs, done2, self.next_obs = self.dataloader.step()
        state = np.log(self.obs[1:,:,3]/self.obs[:-1,:,3])*self.normalize_factor

        self.y1 = np.concatenate([np.ones([1]),self.obs[-1,:,3]/self.obs[-2,:,3]*policy_reward],axis=0).squeeze()

        policy_reward = (policy_reward-1)@policy_weight[1:]*1000

        return torch.tensor(state,dtype=torch.float32).unsqueeze(0), reward, policy_reward, done1 or done2, info

    def reset(self):
        self.portfolio.reset()
        self.obs, self.next_obs = self.dataloader.reset()
        state = np.log(self.obs[1:,:,3]/self.obs[:-1,:,3])*self.normalize_factor
        self.y1 = np.concatenate([np.ones([1,1]),self.next_obs[:,:,3]/self.obs[-1,:,3]],axis=1).squeeze()
        return torch.tensor(state,dtype=torch.float32).unsqueeze(0)

    def render(self):
        print(f'{self.mode}======')
        df_info = pd.DataFrame(self.infos)
        mdd = max_drawdown(df_info.portfolio_value)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        print("Max drawdown", mdd)
        print("Sharpe ratio",sharpe_ratio)
        print("Final portfolio value", df_info["portfolio_value"].iloc[-1])

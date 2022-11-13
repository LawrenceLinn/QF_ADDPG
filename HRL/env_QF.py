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




        '''
        actor输出weight
        PG输出policy 
        价格走位 
        1.收盘价在QPL+之上:P0看涨
            1.直线突破+
            2.下探-后反弹
        2.收盘价在QPL+-之间:P1看震荡
            1.震荡行情 触碰QPL做相反操作 对delta_w操作（反向操作砍掉50% or 整体仓位*50%）：eta*delta*(C/QPL-1)
        3.收盘价在QPL-之下:P2看跌
            1.直线下探- 判断delta 反向操作砍掉 
            2，触碰+后反弹
        '''


        policy_reward = np.ones([9])
        weight = action2position(self,action)
        delta_weight = weight-self.portfolio.w0

        Open =  self.next_obs[0,:,0]
        High = self.next_obs[0,:,1]
        Low = self.next_obs[0,:,2]
        Close = self.next_obs[0, :, 3]
        QPLp1 =  self.next_obs[0,:,4]
        QPLn1 = self.next_obs[0, :, 5]

        # for i in range(self.next_obs.shape[1]):
        #     '''收盘价在QPL+之上'''
        #     if Close[i]>QPLp1[i] and policy==1:#看涨
        #         if QPLn1[i]<Low[i]:#直线突破
        #             if delta_weight[i+1]<0:
        #                 delta_weight[i+1] = 0
        #                 self.y1[i+1] = QPLp1[i]/self.obs[-1,i,3]
        #                 policy_reward [i] = Close[i]/QPLp1[i]
        #         elif QPLn1[i]>Low[i]:#下探QPL-
        #             if delta_weight[i+1]<0:
        #                 delta_weight[i+1] = 0
        #                 self.y1[i + 1] = self.next_obs[:, i, 5] / self.obs[-1, i, 3]
        #                 policy_reward[i] = self.next_obs[:, i, 3] / self.next_obs[:, i, 5]
        #     if self.next_obs[:,i,3]<self.next_obs[:,i,4] and self.next_obs[:,i,3]>self.next_obs[:,i,5] and policy==2:#看震荡
        #         if self.next_obs[:,i,5]>self.next_obs[:,i,2] and :#下探QPL-



        '''
        价格走势
        1.最低价突破QPLn1
            1.Policy看涨 反向调仓-50%
            2.Policy看跌 反向调仓 -50%
        2.L没突破QPLn1
            2.1.H突破QPLp1
                1.Policy看涨
                2.Policy看跌
            2.1 H没突破QPLp1
                delta = 0
        '''
        for i in range(self.next_obs.shape[1]):
            if Low[i]<QPLn1[i]:
                if policy[i]==0 and delta_weight[i+1]<0:
                    delta_weight[i+1] = 0
                    self.y1[i + 1] = QPLn1[i] / self.obs[-1, i, 3]
                    policy_reward [i] = Close[i]/QPLn1[i]
                elif policy[i]==1 and delta_weight[i+1]>0:
                    delta_weight[i+1] = 0
                    self.y1[i + 1] = QPLn1[i] / self.obs[-1, i, 3]
                    policy_reward[i] = Close[i] / QPLn1[i]
                else:
                    pass
            else:
                if High[i]>QPLp1[i]:
                    if policy[i]==0 and delta_weight[i+1]<0:
                        delta_weight[i + 1] = 0
                        self.y1[i + 1] = QPLp1[i] / self.obs[-1, i, 3]
                        policy_reward[i] = Close[i] / QPLp1[i]
                    elif policy[i] == 1 and delta_weight[i + 1] > 0:
                        delta_weight[i + 1] = 0
                        self.y1[i + 1] = QPLp1[i] / self.obs[-1, i, 3]
                        policy_reward[i] = Close[i] / QPLp1[i]
                else:
                    delta_weight[i] = 0


        delta_weight[0] -= sum(delta_weight)


        if self.portfolio.w0[0]+delta_weight[0]>0 and self.portfolio.w0[0]+delta_weight[0]<1:
            weight = self.portfolio.w0 + delta_weight
        elif self.portfolio.w0[0]+delta_weight[0]<0:
            weight = self.portfolio.w0 + delta_weight*abs(weight[0]/delta_weight[0])
        else:
            weight = self.portfolio.w0 + delta_weight * abs((1-weight[0])/ delta_weight[0])


        # weight = self.portfolio.w0 + delta_weight
        # weight[0] += 1-sum(weight)

        reward, info, done1 = self.portfolio.step(weight,self.y1)
        self.infos.append(info)

        self.obs, done2, self.next_obs = self.dataloader.step()
        state = np.log(self.obs[1:,:,3]/self.obs[:-1,:,3])  # win_size,prod_num
        cash = np.zeros((state.shape[0], 1))
        state = np.concatenate([cash,state],axis = 1)
        self.y1 = np.concatenate([np.ones([1,1]),self.next_obs[:,:,3]/self.obs[-1,:,3]*policy_reward],axis=1).squeeze()#next day return

        policy_reward = policy_reward@weight[1:]

        return torch.tensor(state,dtype=torch.float32).unsqueeze(0)*self.normalize_factor, reward, policy_reward, done1 or done2, info

    def reset(self):
        self.portfolio.reset()
        self.obs, self.next_obs = self.dataloader.reset()
        state = np.log(self.obs[1:,:,3]/self.obs[:-1,:,3])#win_size,prod_num
        self.y1 = np.concatenate([np.ones([1,1]),self.next_obs[:,:,3]/self.obs[-1,:,3]],axis=1).squeeze()
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

"""
@Author: Runsheng Lin
Inspired by Qiu.
"""
import numpy as np

class Portfolio(object):

    def __init__(self,**port_param):
        """
           steps
           cost
           mode
           eta
        """
        super(Portfolio, self).__init__()
        for key, value in port_param.items():
            setattr(self, key, value)
        self.infos = []
        self.share = np.array([0.0] * self.product_num)
        self.cash = self.init_portfolio

    def step(self,cash,share,strike_price,Close):
        '''

        :param share:
        :param price:
        :return:
        '''
        p0 = self.p0
        delta_share = share - self.share

        sell_stock = delta_share<0
        sell = delta_share[sell_stock]*strike_price[sell_stock]

        cash = self.cash - sell.sum()*(1-self.cost)

        buy_stock = delta_share > 0
        buy = delta_share[buy_stock] * strike_price[buy_stock]

        if buy.sum()*(1+self.cost)>cash:
            delta_share[buy_stock] *= cash/buy.sum()*(1+self.cost)
            buy = delta_share[buy_stock] * strike_price[buy_stock]

        cash = cash - buy.sum()

        share = self.share+delta_share

        share_value = share*Close

        p1 = cash + share_value.sum()

        share_weight = share_value/p1

        self.weight = np.concatenate([[cash/p1],share_weight],axis=0)

        rho1 = p1/self.p0

        r1 = np.log(rho1)

        self.share = share
        self.cash = cash
        self.p0 = p1

        self.unit_rt = p1/self.init_portfolio

        self.entropy_loss = self.eta * (1 - (self.weight ** 2).sum())

        reward = r1 * 100 - self.entropy_loss

        done = p1<=0

        info = {
            "portfolio_value": self.unit_rt,
            "rate_of_return": rho1-1,
            "log_return": r1,
        }
        self.infos.append(info)
        return reward, info, done


    def reset(self):
        self.infos = []
        self.share = np.array([0.0] * self.product_num)
        self.p0 = self.init_portfolio
        self.unit_rt = 1
        self.cash = self.p0
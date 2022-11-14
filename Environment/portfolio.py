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


    def step(self,w1,y1,reset = 0):
        """
        :param w1:
        :param y1:
        :param reset:
        :return: reward, info, done
        """
        w0 = self.w0
        p0 = self.p0
        y0 = self.y0
        # dw1 = ((y0 * w0) / (y0@w0 + 1e-8))

        # '''如果为回测模式 并且选择止盈 则'''
        # if self.mode == "Test" and reset == 1:
        #     mu1 = self.cost * (np.abs(w1[1:])).sum()
        # else:
        #     mu1 = self.cost * (np.abs(dw1[1:] - w1[1:])).sum()

        mu1 = self.cost * (np.abs(w0[1:] - w1[1:])).sum()

        p1 = p0 * (1 - mu1) * np.dot(y1, w1)

        rho1 = p1 / p0 - 1
        r1 = np.log((p1 + 1e-8) / (p0 + 1e-8))
        self.entropy_loss = self.eta*(1-(w1**2).sum())

        reward = r1*100-self.entropy_loss

        self.w0 = w1
        self.p0 = p1
        self.y0 = y1

        # Run out of money, done
        done = p1 == 0

        info = {
            "portfolio_value": p1,
            "rate_of_return": rho1,
            "log_return": r1,
        }
        self.infos.append(info)
        return reward, info, done


    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * self.product_num)
        self.p0 = 1.0
        self.y0 = np.ones((self.product_num+1,), dtype=float)
        self.y0[0] = 1
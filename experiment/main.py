import torch
from models.QF_DDPG import QF_DDPG
from Environment.env import marketEnv


win_size = 20
gap = 1
model_params = {
    'agent_params':{
        'product_num':9,
        'win_size':win_size,
        'action_size':2#policy
    },
    'device':torch.device('cpu'),
    'capacity':100000,
    'result_path':'1',
    'episodes':100,
    'record':1,
    'batch_size':64,
    'gamma':0.99,
    'tau':0.01,
}


env_params = {
    'loader_params':{
        'gap':gap,
        'data_type':'frex',
        'train_ratio':0.8,
        'win_size':win_size,
        'start_idx': 1,
        'feature_num': 4,
        'market_feature': ['Open','High','Low','Close'],
        'mode':'Train',
        # 'args':args,
    },
    'portfolio_params':{
        'cost':0,
        'mode':'Train',
        'eta':0.01,
    },
    'mode':'Train',
    'short_available':0
}

testenv_params = {
    'loader_params':{
        'gap':gap,
        'data_type':'frex',
        'train_ratio':0.8,
        'win_size':win_size,
        'start_idx': 1,
        'feature_num': 4,
        'market_feature': ['Open','High','Low','Close'],
        'mode':'Train',
        # 'args':args,
    },
    'portfolio_params':{
        'cost':1e-3,
        'mode':'Test',
        'eta':0.01,
    },
    'mode':'Test',
    'short_available':0
}

#TODO:
'''
Policy输出反弹or突破or hold

'''


env = marketEnv(**env_params)
env_test = marketEnv(**testenv_params)

QF = QF_DDPG(**model_params)

QF.train(env,env_test)




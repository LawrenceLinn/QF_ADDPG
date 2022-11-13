import torch
from HRL.QF_HRL import QF_HRL
from HRL.env_QF import marketEnv

win_size = 20
gap = 1
normalize_factor = 10000
model_params = {
    'agent_params':{
        'product_num':9,
        'win_size':win_size,
        'action_size':2#policy
    },
    'device':torch.device('cpu'),
    'capacity':100000,
    'result_path':'0',
    'episodes':100,
    'record':1,
    'batch_size':64,
    'gamma':0.99,
    'tau':0.01,
}

loader_params = {
    'gap':gap,
    'data_type':'frex',
    'train_ratio':0.8,
    'win_size':win_size,
    'start_idx': 1,
    'feature_num': 4,
    'market_feature': ['Open','High','Low','Close','QPL1','QPL-1'],
    'mode':'Train',
    # 'args':args,
}

env_params = {
    'loader_params':loader_params,
    'portfolio_params':{
        'cost':0,
        'mode':'Train',
        'eta':0.01,
    },
    'normalize_factor':normalize_factor,
    'mode':'Train',
    'short_available':0
}

testenv_params = {
    'loader_params':loader_params,
    'portfolio_params':{
        'cost':1e-3,
        'mode':'Test',
        'eta':0.01,
    },
    'normalize_factor':normalize_factor,
    'mode':'Test',
    'short_available':0
}

#TODO:
'''
Policy输出反弹or突破or hold

'''


env = marketEnv(**env_params)
env_test = marketEnv(**testenv_params)

model = QF_HRL(**model_params)

model.train(env,env_test)



import torch
from HRL_comp.QF_HRL_comp import HRL_comp
from HRL_comp.env_QF_comp import marketEnv
import config

'''
实验变量

eta
交易手续费

'''


win_size = 4
gap = 1
normalize_factor = 100
product_num = 9
datatype = 'frex'

model_params = {
    'network_params':{
        'product_num':product_num,
        'win_size':win_size,
        'action_size':2#policy
    },
    'device':torch.device('cpu'),
    'capacity':10000,
    'result_path':'test',
    'episodes':50,
    'record':0,
    'batch_size':64,
    'gamma':0.99,
    'tau':0.001,
}

loader_params = {
    'gap':gap,
    'data_type':datatype,
    'train_ratio':0.8,
    'win_size':win_size,
    'start_idx': 1,
    'feature_num': 4,
    'market_feature': ['Open','High','Low','Close','QPL1','QPL-1'],
    'mode':'Train',
    # 'args':args,
}

testloader_params = {
    'gap':gap,
    'data_type':datatype,
    'train_ratio':0.8,
    'win_size':win_size,
    'start_idx': 1,
    'feature_num': 4,
    'market_feature': ['Open','High','Low','Close','QPL1','QPL-1'],
    'mode':'Test',
    # 'args':args,
}

eta = 0.01

env_params = {
    'loader_params':loader_params,
    'portfolio_params':{
        'cost':0,
        'mode':'Train',
        'eta':eta,
        'product_num':product_num,
    },
    'normalize_factor':normalize_factor,
    'mode':'Train'
}

testenv_params = {
    'loader_params':testloader_params,
    'portfolio_params':{
        'cost':1e-3,
        'mode':'Test',
        'eta':eta,
        'product_num':product_num,
    },
    'normalize_factor':normalize_factor,
    'mode':'Test'
}

#TODO:
'''
Policy输出反弹or突破or hold

'''

torch.autograd.set_detect_anomaly(True)
config.setup_seed()

env = marketEnv(**env_params)
env_test = marketEnv(**testenv_params)
model = HRL_comp(**model_params)
model.train(env,env_test)



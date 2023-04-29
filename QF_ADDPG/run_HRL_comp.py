import torch
from QF_HRL_comp import HRL_comp
from env_QF_comp import marketEnv
import config

win_size = 4
gap = 1
normalize_factor = 1000
product_num = 5
datatype = 'forex'

model_params = {
    'network_params':{
        'product_num':product_num,
        'win_size':win_size,
        'action_size':2
    },
    'device':torch.device('cpu'),
    'capacity':10000,
    'result_path':'comp1',
    'episodes':100,
    'record':1,
    'batch_size':64,
    'gamma':0.99,
    'tau':0.001,
}

loader_params = {
    'gap':gap,
    'data_type':datatype,
    'train_ratio':0.8,
    'win_size':win_size,
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

}

eta = 0.05

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


env = marketEnv(**env_params)
env_test = marketEnv(**testenv_params)
path_checkpoint = './model/checkpoint_comp_eta005.pth'

model = HRL_comp(**model_params)

torch.save(model.state_dict(),path_checkpoint)
model.train(env,env_test)



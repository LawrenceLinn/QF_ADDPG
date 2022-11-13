import random
import torch
import numpy as np
import os
import shutil,time



'''Random seed'''
def setup_seed(seed = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

'''folder for TensorboardX summary writer'''
def load_device(args):
    device = torch.device(args.device)
    return device

def reset_logdir(logdir):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    else:
        shutil.rmtree(logdir)
        os.mkdir(logdir)

"""
Loading andd storing utilities
"""
import pickle
import random
import string
import os
import torch

def load_net_and_opt_checkpoint(resume_checkpoint):

    checkpoint = torch.load(resume_checkpoint)

    return checkpoint#['model_state_dict'],checkpoint['optimizer_state_dict'],checkpoint['epoch']

"""
def save_checkpoint(net, optimizer, epoch, checkpoint_dir, seed, prepend=None):

    checkpoint_name = "epoch_{}_seed_{}".format(epoch,seed)

    #useful for adding on the initial checkpoint name
    if prepend:
        checkpoint_name = "{}_{}".format(prepend,checkpoint_name)

    path = os.path.join(checkpoint_dir,checkpoint_name)

    #save endpoints of curve model
    if hasattr(net,'midpoint_dict'):
        #assumes that only a curve model has this
        checkpoint = {"epoch":epoch,"model_state_dict":net.net.state_dict(),"midpoint":net.midpoint_dict,"optimizer_state_dict":optimizer.state_dict(),"endpoint_1":net.dict_1,"endpoint_2":net.dict_2}

    else:
        checkpoint = {"epoch":epoch,"model_state_dict":net.state_dict(),"optimizer_state_dict":optimizer.state_dict()}
    
    torch.save(checkpoint,path)
"""

def save_bma(checkpoint_dir, checkpoint_name, args, predictions):

    path = os.path.join(checkpoint_dir,checkpoint_name)

    checkpoint = {"args":args,"predictions":predictions}

    torch.save(checkpoint,path)

def save_checkpoint(checkpoint_dir, checkpoint_name, **kwargs):

    path = os.path.join(checkpoint_dir,checkpoint_name)
    torch.save(kwargs,path)

def load_checkpoint(resume_checkpoint):

    checkpoint = torch.load(resume_checkpoint)

    return checkpoint#['model_state_dict'],checkpoint['optimizer_state_dict'],checkpoint['epoch']


def save_dict(to_save,save_dir,save_name):
    with open(os.path.join(save_dir,save_name), 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(load_dir,load_name):
    with open(os.path.join(load_dir,load_name), 'rb') as handle:
        return pickle.load(handle)
            

def get_random_save_name(save_dir):

    taken = set(os.listdir(save_dir))
    
    N = 10

    name =  ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))+'.pkl'

    while name in taken:
        name =  ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))+'.pkl'

    return name

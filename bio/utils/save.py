"""
Loading andd storing utilities
"""

import os
import torch
import pickle
import random
import string

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

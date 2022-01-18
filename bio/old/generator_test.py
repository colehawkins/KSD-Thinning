import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import torch
import ksdp




samples,grads = utils.data.get_samples_and_grads('goodwin')



sample_generator = ((i,samples[i],grads[i]) for i in range(samples.shape[0]))

for _ in range(4):
    step,sample,grad = next(sample_generator)
    print(step,sample,grad)




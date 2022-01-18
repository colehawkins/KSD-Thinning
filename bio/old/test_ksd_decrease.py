import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import torch
import ksdp

samples,grads = utils.data.get_samples_and_grads('goodwin')

samples = torch.tensor(samples)
grads = torch.tensor(grads)

if torch.cuda.is_available():
    samples = samples.cuda()
    grads = grads.cuda()

gap = 250
indices = range(int(0.2*samples.shape[0]),samples.shape[0],gap)
samples = samples[indices]
grads = grads[indices]

plt.figure()

ksds = ksdp.ksd.get_sequential_KSDs(samples,grads,kernel_type='rbf',h_method='dim')

plt.plot([x.item() for x in ksds],label='Thinned')
plt.yscale('log')
plt.xscale('log')
plt.show()


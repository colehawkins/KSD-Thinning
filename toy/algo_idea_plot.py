import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.distributions as D

torch.manual_seed(2)

DELTA = 0.025
S = 600
XRANGE = (-3.0,3.0)
YRANGE = (-3.0,3.0)

x = np.arange(*XRANGE, DELTA)
y = np.arange(*YRANGE, DELTA)
X, Y = np.meshgrid(x, y)


mix = D.Categorical(torch.ones(2))  
means = torch.Tensor([[1.0,1.0],[-1.0,-1.0]])
comp = D.Independent(D.Normal(means,scale=0.7),1)
gmm = D.MixtureSameFamily(mix,comp)

density_fn = gmm.log_prob

concat = np.stack([np.ravel(X),np.ravel(Y)],axis=1)

Z = density_fn(torch.Tensor(concat))
Z = np.reshape(Z.numpy(),X.shape)



samples = gmm.sample([20]).numpy()

samples = np.concatenate([samples[:6,:],samples[7:,:]])

new_sample = torch.tensor([-1.3,-1.0]).view([1,2])

idx_1 = 11
idx_2 = 3

to_remove = np.stack([samples[idx_1],samples[idx_2]])

fig, axes = plt.subplots(1,1)
"""
ax = axes[0]
ax.contourf(X, Y, Z, levels=30, cmap='Blues')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.invert_yaxis()

rows = []

for sample in samples:
    rows.append({'x':sample[0],'y':sample[1],'type':'curr'})
    
df = pd.DataFrame(rows)
sns.scatterplot(ax=ax,x='x',y='y',data=df,hue='type',palette=['green'],s=S,legend=False)
"""

ax = axes
ax.contourf(X, Y, Z, levels=30, cmap='Blues')

rows = []

for sample in samples:
    rows.append({'x':sample[0],'y':sample[1],'type':'curr'})
    
rows.append({'x':new_sample[0,0],'y':new_sample[0,1],'type':'new'})
rows.append({'x':to_remove[0,0],'y':to_remove[0,1],'type':'to_remove'})
rows.append({'x':to_remove[1,0],'y':to_remove[1,1],'type':'to_remove'})
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.invert_yaxis()


df = pd.DataFrame(rows)
sns.scatterplot(ax=ax,x='x',y='y',data=df,hue='type',palette=['green','yellow','red'],s=S,legend=False)#label='Current Samples'
"""
final_samples = np.concatenate([samples[:3],samples[4:11],samples[12:],new_sample])

rows = []

for sample in final_samples:
    print(sample.shape)
    rows.append({'x':sample[0],'y':sample[1],'type':'curr'})
    

ax = axes[2]
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.invert_yaxis()
ax.contourf(X, Y, Z, levels=30, cmap='Blues')
df = pd.DataFrame(rows)
sns.scatterplot(ax=ax,x='x',y='y',data=df,hue='type',palette=['green'],s=S,legend=False)#label='Current Samples'
fig.set_size_inches(20,7)
plt.tight_layout()
"""
fig.set_size_inches(10,10)
plt.savefig('../plots/algo_idea.png')
plt.show()
plt.close()


"""
Process and plot results from various thinning methods
"""
import itertools
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import utils.plotting
import ray.tune
from scipy.ndimage.filters import gaussian_filter1d

STYLE_ORDER = ['None', 'KSDT-SQRT', 'KSDT-LINEAR']
metrics = ['Agreement','Total Variation','Accuracy','ECE']
samplers = ['MALA','SPMCMC-MALA']#Agreement','Total Variation','Accuracy','ECE']
x_vals = ['# Samples']#, 'Time (s)']
eval_num_filter = [100,200,500,1000,2000]


parser = argparse.ArgumentParser()
parser.add_argument('--results-path',
                    type=str,
                    default='.ray_data/sample_Nov-05-2021_cifar10paper')
parser.add_argument('--sample-generation',
                    type=str,
                    nargs='+',
                    default=['mala','spmcmc'])
parser.add_argument('--metrics',
                    type=str,
                    nargs='+',
                    default=['Agreement','Total Variation'])
parser.add_argument('--kernel-type',
                    type=str,
                    default='imq')
parser.add_argument('--smoke-test',
                    action='store_true',
                    default=False)
parser.add_argument('--show',
                    action='store_true',
                    default=False)
args = parser.parse_args()


metrics = [x for x in metrics if x in args.metrics]

analysis = ray.tune.Analysis(args.results_path)


configs = analysis.get_all_configs()
results = analysis.trial_dataframes

new_results = []
#appendd configuarion info to each datafrane row
for config in configs:
    df = results[config]
    for key,val in configs[config].items():
        if type(val) in [tuple,list]:
            df[key]=len(df)*[val]
        else:
            df[key]=val

    df = df[df['num_evals'].isin(eval_num_filter)]
    df = df[df['kernel_type']==args.kernel_type]

    new_results.append(df)

df = pd.concat(new_results)



processed_rows = []

#tag with name and date
plot_tag = args.results_path.split('_')[-1][:-5]+'_'+''.join(args.kernel_type)+'_'+''.join([x.replace(' ','') for x in args.metrics])

for _, tmp_row in df.iterrows():
    row = tmp_row.to_dict()
    sampler_name = utils.plotting.format_sampler_name(row)
    pruning_name = utils.plotting.format_pruning_name(row)
    kernel_type = utils.plotting.format_kernel_name(row)
    
    new_row = {
        'Sampler':
        sampler_name,
        'Pruning':
        pruning_name,
        'Kernel Type':
        kernel_type,
        'KSD':
        row['ksd'],
        'Normalized KSD':
        row['ksd'] * np.sqrt(row['num_samples']),
        'Agreement':
        row['bma_agreement'],
        'Total Variation':
        row['bma_tv'],
        'ECE':
        row['bma_ece'],
        'Accuracy':
        row['bma_accuracy'],
        '# Samples':
        row['num_samples'],
        '# Evals':
        row['num_evals']
    }
    if False:#new_row['Sampler']=='MALA' and new_row['Pruning']=='None':
        pass
    else:
        processed_rows.append(new_row)

processed_df = pd.DataFrame(processed_rows)


FONT_MULTIPLE = 1.75

rows = list(itertools.product(x_vals,samplers))
cols = metrics
fig, ax = plt.subplots(len(rows),len(cols))
font_scale=3*FONT_MULTIPLE
sns.set(style='whitegrid',font_scale=font_scale)
LINEWIDTH = 4
fig.set_size_inches(30,15)

raveled_axes = ax.ravel() if len(metrics)*len(x_vals)>1 else ax
for row, (x_val,sampler) in enumerate(rows):
    for col, metric in enumerate(cols):
        to_plot = processed_df[processed_df['Sampler']==sampler]
        to_plot = to_plot.groupby(['Pruning','# Evals']).mean()

        curr_ax = raveled_axes[row*len(cols)+col]
        g = sns.lineplot(ax=curr_ax,
                         x=x_val,
                         y=metric,
                         hue='Pruning',
                         #size='# Samples',
                         data=to_plot,
                         linewidth=LINEWIDTH,
                         hue_order=STYLE_ORDER,
                         #style='Pruning',
                         marker='o',
                         markers=True,
                         markersize=35
                         )
        if 'KSD' in metric:
            g.set_yscale('log')

        g.set_title(sampler,fontsize=10*font_scale)
        g.set_xscale('log',base=2)
        y_label = metric
        if 'KSD' in metric:
            y_label += ' ({})'.format(kernel_type)

        g.set_ylabel(y_label,fontsize=10*font_scale)
        if x_val == '# Evals':
            g.set_xlim(min_evals, None)
        elif x_val == 'Time (s)':
            g.set_xlim(min_time, None)
        if row!=(len(rows)-1):
            g.set_xlabel('')
        else:
            g.set_xlabel(x_val,fontsize=10*font_scale)
        curr_ax.tick_params(axis='y',labelsize=25*FONT_MULTIPLE)
        curr_ax.tick_params(axis='x',labelsize=25*FONT_MULTIPLE)

for x in ax.ravel():
    x.legend([],[], frameon=False)
handles, labels = raveled_axes[-1].get_legend_handles_labels()
for handle in handles:
    handle.set_linewidth(7.0)
#    handle.set_markersize(20.0)

#fig.legend(handles, labels, loc='upper center',ncol=len(labels),bbox_to_anchor=[0.5,1.01])

lgd = fig.legend(handles, labels, loc='upper center',ncol=len(labels),bbox_to_anchor=[0.5,1.1])
plt.tight_layout()
plt.savefig('../plots/{}_full.png'.format(plot_tag),bbox_extra_artists=(lgd,), bbox_inches='tight')
#plt.tight_layout()
if args.show:
    plt.show()
#plt.savefig('../plots/{}_full.png'.format(plot_tag))


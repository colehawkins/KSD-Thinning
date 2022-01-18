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

FRAC=0.1
STYLE_ORDER = ['None', 'KSDT-SQRT', 'KSDT-LINEAR']
metrics = ['KSD','Normalized KSD']
x_vals = ['# Evals']
#,'Total Variation', 'Normalized Total Variation']#'Agreement','Total Variation','Accuracy','ECE']
#kernel_types = ['IMQ', 'RBF']

parser = argparse.ArgumentParser()
parser.add_argument('--results-path',
                    type=str,
                    default='.ray_data/sample_Nov-05-2021_cifar10paper')
parser.add_argument('--kernel-types',
                    type=str,
                    nargs='+',
                    default=['imq','rbf'])
parser.add_argument('--smoke-test',
                    action='store_true',
                    default=False)
parser.add_argument('--show',
                    action='store_true',
                    default=False)
args = parser.parse_args()

analysis = ray.tune.Analysis(args.results_path)

configs = analysis.get_all_configs()
results = analysis.trial_dataframes

if 'modes' in args.results_path:
    metrics = ['# Samples']
    hue='# Modes'
elif 'alpha' in args.results_path:
    hue=r'$\alpha$'

keys_to_smooth = ['ksd']#'ksd','bma_agreement','bma_tv','bma_ece','bma_accuracy']
for trial in results:
    for to_smooth in keys_to_smooth:
        results[trial][to_smooth] = gaussian_filter1d(results[trial][to_smooth].to_numpy(),sigma=5)



#appendd configuarion info to each datafrane row
for config in configs:
    df = results[config]
    for key,val in configs[config].items():
        if type(val) in [tuple,list]:
            df[key]=len(df)*[val]
        else:
            df[key]=val


#result_dfs = [val for key,val in ray.tune.Analysis(args.results_path).trial_dataframes.items()]

df = pd.concat(results.values())

df=df[df['kernel_type'].apply(lambda x: x in args.kernel_types)]
#df=df[df['config/sample_generation'].apply(lambda x: x in args.sample_generation)]


processed_rows = []

#tag with name and date
plot_tag = args.results_path.split('/')[-1].split('_')[0]+'_'+''.join(args.kernel_types)
print(plot_tag)

for _, tmp_row in df.iterrows():
    row = tmp_row.to_dict()
    sampler_name = utils.plotting.format_sampler_name(row)
    pruning_name = utils.plotting.format_pruning_name(row)
    kernel_type = utils.plotting.format_kernel_name(row)
    
    new_row = {
        'Sampler':
        sampler_name,
        r'$\alpha$':
        row['alpha'],
        '# Modes':
        row['modes'],
        'Pruning':
        pruning_name,
        'Kernel Type':
        kernel_type,
        'KSD':
        row['ksd'],
        'Normalized KSD':
        row['ksd'] * np.sqrt(row['num_samples']),
        '# Samples':
        row['num_samples'],
        '# Evals':
        row['num_evals']
    }
    if np.isnan(new_row['KSD']):
        pass
    else:
        processed_rows.append(new_row)

processed_df = pd.DataFrame(processed_rows)
processed_df.sort_values(['Sampler', 'Pruning'], ascending=[True, False])

print(len(processed_df))

if args.smoke_test:
    processed_df=processed_df.sample(frac=0.05)

FONT_MULTIPLE=1.75
sns.set(style='whitegrid',font_scale=3*FONT_MULTIPLE)
LINEWIDTH = 4
fig, ax = plt.subplots(len(args.kernel_types)*len(x_vals),len(metrics))
if len(metrics)==1:
    fig.set_size_inches(16.7,13.4)
else:
    fig.set_size_inches(16.7*len(metrics),13.4)

args.kernel_types = [x.upper() for x in args.kernel_types]


rows = list(itertools.product(args.kernel_types,x_vals))
cols = metrics

raveled_axes = ax.ravel() if len(rows)*len(cols)>1 else ax
for row, (kernel_type,x_val) in enumerate(rows):
    for col, metric in enumerate(cols):
        print(row,col)
        to_plot = processed_df[processed_df['Kernel Type'] == kernel_type]
        if FRAC<1.0:
            subsampled_to_plot = utils.plotting.logarithmic_uniform_thinning(
                df=to_plot, frac=FRAC, key=x_val)
        else:
            subsampled_to_plot = to_plot
        print(len(subsampled_to_plot))

        curr_ax = raveled_axes[row*len(cols)+col] if len(rows)*len(cols)>1 else ax
        g = sns.lineplot(ax=curr_ax,
                         x=x_val,
                         y=metric,
                         hue=hue,
                         hue_order=[hue],
                         data=subsampled_to_plot,
                         linewidth=LINEWIDTH,
                         )
        g.set_xscale('log')
        if 'KSD' in metric:
            g.set_yscale('log')
        g.set_ylabel(metric + ' ({})'.format(kernel_type) if 'KSD' in metric else metric)
if len(rows)*len(cols)>1:
    raveled_axes = ax.ravel()
else:
    raveled_axes = [ax]
for x in raveled_axes:
    x.legend([],[], frameon=False)
handles, labels = raveled_axes[-1].get_legend_handles_labels()
for handle in handles:
    handle.set_linewidth(7.0)
    handle.set_markersize(20)

ph = [plt.plot([],marker="",ls="")[0]]
title=hue
handles = ph+handles
labels = [title]+labels
lgd = fig.legend(handles, labels, loc='upper center',ncol=len(labels),bbox_to_anchor=(0.5, 1.1))
plt.tight_layout()
if args.show:
    plt.show()
plt.savefig('../plots/{}_full.png'.format(plot_tag),bbox_extra_artists=(lgd,),bbox_inches='tight')

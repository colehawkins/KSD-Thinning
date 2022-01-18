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
metrics = ['KSD', 'Normalized KSD','Total Variation', 'Normalized Total Variation']#'Agreement','Total Variation','Accuracy','ECE']
x_vals = ['# Evals']#, 'Time (s)']
#kernel_types = ['IMQ', 'RBF']

parser = argparse.ArgumentParser()
parser.add_argument('--results-path',
                    type=str,
                    default='.ray_data/sample_Nov-05-2021_cifar10paper')
parser.add_argument('--sample-generation',
                    type=str,
                    nargs='+',
                    default=['mala','spmcmc'])
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

keys_to_smooth = ['ksd','bma_agreement','bma_tv','bma_ece','bma_accuracy']
for trial in results:
    for to_smooth in keys_to_smooth:
        results[trial][to_smooth] = gaussian_filter1d(results[trial][to_smooth].to_numpy(),sigma=4)



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

HUE_ORDER = ['MALA','SPMCMC-MALA']
min_evals = 50
min_time = 1

def hue_filter(x):
    for z in args.sample_generation:
        if z in x.lower():
            return True
    
    return False

HUE_ORDER = [x for x in HUE_ORDER if hue_filter(x)]

processed_rows = []

#tag with name and date
plot_tag = args.results_path.split('/')[-1].split('_')[0]+'_'+''.join(args.kernel_types)+'_'+''.join(args.sample_generation)

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
        'Normalized Total Variation':
        row['bma_tv'] * np.sqrt(row['num_samples']),
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
    if np.isnan(new_row['KSD']):
        pass
    else:
        processed_rows.append(new_row)

processed_df = pd.DataFrame(processed_rows)
processed_df.sort_values(['Sampler', 'Pruning'], ascending=[True, False])

if args.smoke_test:
    processed_df=processed_df.sample(frac=0.05)


sns.set(style='whitegrid',font_scale=1)
LINEWIDTH = 1
fig, ax = plt.subplots(len(args.kernel_types)*len(x_vals),len(metrics))


args.kernel_types = [x.upper() for x in args.kernel_types]


rows = list(itertools.product(args.kernel_types,x_vals))
cols = metrics

raveled_axes = ax.ravel()
for row, (kernel_type,x_val) in enumerate(rows):
    for col, metric in enumerate(cols):
        print(row,col)
        to_plot = processed_df[processed_df['Kernel Type'] == kernel_type]
        if FRAC<1.0:
            subsampled_to_plot = utils.plotting.logarithmic_uniform_thinning(
                df=to_plot, frac=FRAC, key=x_val)
        else:
            subsampled_to_plot = to_plot

        curr_ax = raveled_axes[row*len(cols)+col]
        g = sns.lineplot(ax=curr_ax,
                         x=x_val,
                         y=metric,
                         style='Pruning',
                         hue='Sampler',
                         data=subsampled_to_plot,
                         linewidth=LINEWIDTH,
                         style_order=STYLE_ORDER,
                         hue_order=HUE_ORDER
                         )
        g.set_xscale('log')
        g.set_yscale('log')
        g.set_ylabel(metric + ' ({})'.format(kernel_type))
        if x_val == '# Evals':
            g.set_xlim(min_evals, None)
        elif x_val == 'Time (s)':
            g.set_xlim(min_time, None)

"""
for x in ax.ravel():
    x.legend([],[], frameon=False)
handles, labels = raveled_axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',ncol=len(labels))
"""
fig.set_size_inches(40, 15)
if args.show:
    plt.show()
plt.savefig('../plotting/plots/{}_full.png'.format(plot_tag))
print(args.show)

"""
metrics = ['KSD', 'Normalized KSD']
x_vals = ['# Evals']

rows = list(itertools.product(args.kernel_types,x_vals,HUE_ORDER,metrics))

fig, ax = plt.subplots(len(rows))

LINEWIDTH = 1
sns.set(style='whitegrid',font_scale=1.0)

raveled_axes = ax.ravel()
for row, (kernel_type,x_val,sampler,metric) in enumerate(rows):
    to_plot = processed_df[processed_df['Kernel Type'] == kernel_type]
    to_plot = to_plot[to_plot['Sampler'] == sampler]

    subsampled_to_plot = utils.plotting.logarithmic_subsample(
        df=to_plot, frac=FRAC, key=x_val)

    curr_ax = raveled_axes[row]
    g = sns.lineplot(ax=curr_ax,
                     x=x_val,
                     y=metric,
                     style='Pruning',
                     data=subsampled_to_plot,
                     linewidth=LINEWIDTH,
                     style_order=STYLE_ORDER,
                     )
    g.set_xscale('log' if 'KSD' in metric else None)
    g.set_yscale('log' if 'KSD' in metric else None)
    g.set_ylabel(metric + ' ({})'.format(kernel_type))
    g.set_title(sampler)
    if row!=ax.shape[0]-1:
        g.set_xlabel(None)
    if x_val == '# Evals':
        g.set_xlim(min_evals, None)
    elif x_val == 'Time (s)':
        g.set_xlim(min_time, None)

for x in ax.ravel():
    x.legend([],[], frameon=False)
handles, labels = ax[-1,-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',ncol=len(labels)+len(handles))

fig.set_size_inches(30, 15)
plt.tight_layout()
plt.savefig('../plotting/plots/{}_per_sampler.png'.format(plot_tag))
if args.show:
    plt.show()
"""

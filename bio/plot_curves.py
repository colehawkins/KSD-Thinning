"""
Process and plot results from various thinning methods
"""
import itertools
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import utils

FRAC = 0.01
STYLE_ORDER = ['None', 'KSDT-SQRT', 'KSDT-LINEAR']
metrics = ['KSD', 'Norm. KSD']
X_VAL = '# Evals'#, 'Time (s)']
#kernel_types = ['IMQ', 'RBF']

parser = argparse.ArgumentParser()
parser.add_argument('--results-path',
                    type=str,
                    default='results/main/cardiac_Oct-28-2021.pkl')
parser.add_argument('--sample-generation',
                    type=str,
                    nargs='+',
                    default=['mala','rwm','tempered'])
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

df = pd.read_pickle(args.results_path)
df['results'] = df['save_name'].apply(lambda x: utils.save.load_dict(
    load_dir='results/linked_dicts', load_name=x))

df=df[df['config/kernel_type'].apply(lambda x: x in args.kernel_types)]

def hue_filter(x):
    for z in args.sample_generation:
        if z in x.lower():
            return True
    
    return False

if 'cardiac' in args.results_path:
    HUE_ORDER=['RWM','SPMCMC-RWM']
    min_evals = 500
    min_time = 1

    yticks = [[1e3,5e3,1e4],[1e4,5e4,1e5]]
elif 'goodwin' in args.results_path:
    df=df[df['config/sample_generation'].apply(lambda x: x in args.sample_generation)]
    HUE_ORDER = ['MALA','RWM','SPMCMC-MALA','SPMCMC-RWM']
    min_evals = 11000
    min_time = 10
    HUE_ORDER = [x for x in HUE_ORDER if hue_filter(x)]
    yticks = [[1e0,3e5,1e5],[1e1,1e4,1e7]]

processed_rows = []

#tag with name and date
plot_tag = args.results_path.split('/')[-1].split('_')[0]+'_'+''.join(args.kernel_types)+'_'+''.join(args.sample_generation)

for _, tmp_row in df.iterrows():
    row = tmp_row.to_dict()
    sampler_name = utils.plotting.format_sampler_name(row)
    pruning_name = utils.plotting.format_pruning_name(row)
    kernel_type = utils.plotting.format_kernel_name(row)
    row_results = row['results']
    
    for i in range(len(row['results']['ksd'])):

        new_row = {
            'Sampler':
            sampler_name,
            'Pruning':
            pruning_name,
            'Kernel Type':
            kernel_type,
            'KSD':
            row_results['ksd'][i],
            'Time (s)':
            row_results['time'][i],
            'Norm. KSD':
            row_results['ksd'][i] * np.sqrt(row_results['num_samples'][i]),
            '# Samples':
            row_results['num_samples'][i],
            '# Evals':
            row_results['num_evals'][i]
        }
        if np.isnan(new_row['KSD']):
            pass
        else:
            processed_rows.append(new_row)

processed_df = pd.DataFrame(processed_rows)
processed_df.sort_values(['Sampler', 'Pruning'], ascending=[True, False])

if args.smoke_test:
    processed_df=processed_df.sample(frac=0.1)



sampler_types = sorted(processed_df['Sampler'].unique(),key=lambda x:'SPMCMC' in x)

args.kernel_types = [x.upper() for x in args.kernel_types]
rows = list(itertools.product(args.kernel_types,sampler_types))
print(rows)
cols = metrics

FONT_MULTIPLE = 1.75

fig, ax = plt.subplots(len(rows),len(cols))
sns.set(style='whitegrid',font_scale=3*FONT_MULTIPLE)
LINEWIDTH=4
fig.set_size_inches(30, 15)
import matplotlib
raveled_axes = ax.ravel()
for row, (kernel_type, sampler_type) in enumerate(rows):
    for col, metric in enumerate(cols):
        #sns.set_context('poster')
        to_plot = processed_df[processed_df['Kernel Type'] == kernel_type]
        to_plot = to_plot[to_plot['Sampler']==sampler_type]
        subsampled_to_plot = utils.plotting.logarithmic_subsample(
            df=to_plot, frac=FRAC, key=X_VAL)

        curr_ax = raveled_axes[row*len(cols)+col]
        g = sns.lineplot(ax=curr_ax,
                         x=X_VAL,
                         y=metric,
                         hue='Pruning',
                         data=subsampled_to_plot,
                         linewidth=LINEWIDTH,
                         hue_order=STYLE_ORDER,
                         )
        """
        if kernel_type=='IMQ' and metric=='Normalized KSD':
            g.set_yticks([1e4,1e5])
        """
        g.set_xscale('log')
        g.set_yscale('log')
        """
        g.set_yticklabels(g.get_yticks(), size = 15)
        g.set_xticklabels(g.get_yticks(), size = 15)
        """

        """
        yticks = ax.get_yticks()
        print(yticks)
        ytick_labels = [yticks[0]]+(len(yticks)-2)*['']+[yticks[-1]]
        
        curr_ax.set_yticklabels(ytick_labels)
        """
        g.set_title(sampler_type,fontsize=30*FONT_MULTIPLE)
        g.set_xlim(min_evals, None)

        if row!=(len(rows)-1):
            g.set_xlabel('')
        else:
            g.set_xlabel('# Evals',fontsize=30*FONT_MULTIPLE)
        """ 
        curr_ax.tick_params(axis='x',labelsize=30)
        curr_ax.tick_params(axis='y',labelsize=30)
        """
        if metric!='Norm. KSD':
            g.set_ylabel(metric + ' ({})'.format(kernel_type),fontsize=30*FONT_MULTIPLE)
        else:
            g.set_ylabel(metric,fontsize=30*FONT_MULTIPLE)
        #g.set_yticks(yticks[col])
        curr_ax.set_yticks(yticks[col])
        curr_ax.tick_params(axis='y',labelsize=25*FONT_MULTIPLE)
        curr_ax.tick_params(axis='x',labelsize=25*FONT_MULTIPLE)
        #curr_ax.set_yticklabels(yticks[col])
       
for x in ax.ravel():
    x.legend([],[], frameon=False)
handles, labels = raveled_axes[-1].get_legend_handles_labels()
for handle in handles:
    handle.set_linewidth(7.0)
#fig.legend(handles, labels, loc='upper center',ncol=len(labels))

lgd = fig.legend(handles, labels, loc='upper center',ncol=len(labels),bbox_to_anchor=[0.5,1.1])
plt.tight_layout()
plt.savefig('../plots/{}_full.png'.format(plot_tag),bbox_extra_artists=(lgd,), bbox_inches='tight')
if args.show:
    plt.show()

"""
metrics = ['KSD', 'Normalized KSD']

rows = list(itertools.product(args.kernel_types,HUE_ORDER))
cols = metrics

sns.set(style='whitegrid',font_scale=3)
fig, ax = plt.subplots(len(rows),len(cols))
LINEWIDTH=4
fig.set_size_inches(31.4, 13.4)

raveled_axes = ax.ravel()


for row, (kernel_type,x_val,sampler) in enumerate(rows):
    for col, metric in enumerate(cols):
        to_plot = processed_df[processed_df['Kernel Type'] == kernel_type]
        to_plot = to_plot[to_plot['Sampler'] == sampler]

        subsampled_to_plot = utils.plotting.logarithmic_subsample(
            df=to_plot, frac=FRAC, key=x_val)

        curr_ax = ax[row,col]
        g = sns.lineplot(ax=curr_ax,
                         x=x_val,
                         y=metric,
                         style='Pruning',
                         data=subsampled_to_plot,
                         linewidth=LINEWIDTH,
                         style_order=STYLE_ORDER,
                         )
        g.set_xscale('log')
        g.set_yscale('log')
        g.set_ylabel(metric.replace('Normalized','Norm.') + ' ({})'.format(kernel_type))
        g.set_title(sampler)
        g.legend(loc='lower left')
        handle, label = curr_ax.get_legend_handles_labels()
        for h in handle:
            h.set_linewidth(7.0)


        if row!=ax.shape[0]-1:
            g.set_xlabel(None)
        g.set_xlim(min_evals, None)

for x in ax.ravel():
    x.legend([],[], frameon=False)
handles, labels = ax[-1,-1].get_legend_handles_labels()
for handle in handles:
    handle.set_linewidth(7.0)

fig.legend(handles, labels, loc='upper center')#,ncol=len(labels))
plt.tight_layout()
fig.set_size_inches(30, 25)
plt.savefig('../plots/{}_per_sampler.png'.format(plot_tag))
if args.show:
    plt.show()
"""

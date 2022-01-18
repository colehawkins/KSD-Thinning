"""
Utility functions for plotting
"""

import pandas as pd
import numpy as np

def logarithmic_subsample(df, frac, key):

    if len(df)==0:
        return df
    #use log of relevant key for sorting
    bucket_key = df.loc[:,key].apply(lambda x: np.log2(x + 1e-10))

    #get bounds
    min_val, max_val = int(np.floor(bucket_key.min())), int(
        np.ceil(bucket_key.max()))

    buckets = []
    for bucket_left_edge in range(min_val, max_val):
        in_bucket = bucket_key.apply(lambda z:  bucket_left_edge < z < (bucket_left_edge + 1))
        bucket_subset = df[in_bucket]

        #don't filter too small
        if bucket_subset.shape[0]<100:
            buckets.append(bucket_subset)
        else:
            buckets.append(bucket_subset.sample(frac=frac))

    return pd.concat(buckets)


def format_kernel_name(row):
    '''Produce pruning name based on pruning characteristics'''
    kernel_type = row['config/kernel_type']

    if kernel_type == 'imq':
        return 'IMQ'
    elif kernel_type == 'rbf':
        return 'RBF'
    else:
        raise NotImplementedError(
            "Kernel type {} not implemented".format(kernel_type))


def format_pruning_name(row):
    '''Produce pruning name based on pruning characteristics'''

    if row['config/prune'][0]:
        row_name = 'KSDT-' + row['config/prune'][1].upper()
    else:
        row_name = "None"

    return row_name


def format_sampler_name(row):
    '''Produce sampler name based on sampler characteristics'''
    if row['config/sample_generation'] == 'mala':
        row_name = 'MALA'
    elif row['config/sample_generation'] == 'rwm':
        row_name = 'RWM'
    elif row['config/sample_generation'] == 'tempered':
        row_name = 'RWM'

    if row['config/addition_rule'] == 'spmcmc':
        row_name = 'SPMCMC-' + row_name

    return row_name

"""
Utility functions for plotting
"""

import pandas as pd
import numpy as np
import random

def logarithmic_uniform_thinning(df, frac, key):
    """
    Bucket dataframe into logarithmic chunks by key
    then perform uniform thinning on each chunk.
    Thin by unique keys to preserve vals for std err
    """

    if len(df)==0:
        return df
    #use log of relevant key for sorting
    bucket_key = df.loc[:,key].apply(lambda x: np.log2(x + 1e-10))

    #get bounds for key
    min_val, max_val = int(np.floor(bucket_key.min())), int(
        np.ceil(bucket_key.max()))

    buckets = []
    for bucket_left_edge in range(min_val, max_val):
        in_bucket = bucket_key.apply(lambda z:  bucket_left_edge < z < (bucket_left_edge + 1))
        bucket_subset = df[in_bucket]

        #don't filter if too few samples
        if bucket_subset.shape[0]<100:
            buckets.append(bucket_subset)
        else:
            keys_to_thin = [x for x in bucket_subset[key].unique()]
            indices = range(0,len(keys_to_thin),max(1,int(1.0/frac)))
            thinned_keys = [keys_to_thin[i] for i in indices]
            rows_kept =  bucket_subset[bucket_subset[key].isin(thinned_keys)]#.apply(lambda z: z[key] in thinned_keys)]
            buckets.append(rows_kept)
            #buckets.append(bucket_subset.sample(frac=frac))

    return pd.concat(buckets)


def format_kernel_name(row):
    '''Produce pruning name based on pruning characteristics'''
    kernel_type = row['kernel_type']

    if kernel_type == 'imq':
        return 'IMQ'
    elif kernel_type == 'rbf':
        return 'RBF'
    else:
        raise NotImplementedError(
            "Kernel type {} not implemented".format(kernel_type))


def format_pruning_name(row):
    '''Produce pruning name based on pruning characteristics'''

    if row['prune'][0]:
        row_name = 'KSDT-' + row['prune'][1].upper()
    else:
        row_name = "None"

    return row_name


def format_sampler_name(row):
    '''Produce sampler name based on sampler characteristics'''
    row_name = 'MALA'

    if row['addition_rule'] == 'spmcmc':
        row_name = 'SPMCMC-' + row_name

    return row_name

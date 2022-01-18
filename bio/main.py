"""
Run sampling with or without KSDP pruning using sample chains
from the paper Optimal Thinning of MCMC Output (Riabiz et al.)

Assumes samples have been downloaded using setup scripts
"""

import math
import argparse
import time
import ksdp
import ray
import torch
import utils
import numpy as np
EPSILON = 0


def main(args):
    '''Run sampling (and pruning) on data used in various past Stein Methods papers'''
    save_name = utils.save.get_random_save_name(
        args.save_dir) if args.save else None

    samples, gradients = utils.data.get_samples_and_grads(
        problem=args.problem,
        sample_generation=args.sample_generation,
        smoke_test=args.smoke_test)

    if getattr(args, 'smoke_test', False):
        print("Running smoke test with sample subset size {}".format(
            samples.shape[0]))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pruning_container = ksdp.PruningContainer(kernel_type=args.kernel_type,
                                              h_method='dim' if args.kernel_type=='rbf' else None,
                                              full_mat=args.full_mat)

    init_sample = torch.tensor(samples[0]).to(device)
    init_gradient = torch.tensor(gradients[0]).to(device)
    pruning_container.add_point(point=init_sample, gradient=init_gradient)

    samples = samples[1:]
    gradients = gradients[1:]

    sample_generator = (
        (torch.tensor(samples[i:i + args.samples_per_iter]).to(device),
         torch.tensor(gradients[i:i + args.samples_per_iter]).to(device))
        for i in range(0, samples.shape[0], args.samples_per_iter))

    results = {'num_evals': [], 'ksd': [], 'num_samples': [], 'time': []}

    t = time.time()
    for step, (batch_samples, batch_gradients) in enumerate(sample_generator):

        #deduplicate (no MCMC state change)
        _, idx = batch_samples.unique_consecutive(dim=0,return_inverse=True)
        idx = idx.unique()
        batch_samples = batch_samples[idx]
        batch_gradients = batch_gradients[idx]

        next_sample, next_gradient = utils.bayesian.select_samples(
            pruning_container=pruning_container,
            new_samples=batch_samples,
            new_gradients=batch_gradients,
            addition_rule=args.addition_rule)

        pruning_container.add_point(point=next_sample, gradient=next_gradient)


        if args.prune[0]:

            min_samples = ksdp.utils.get_min_samples(growth=args.prune[1],
                                                         step=step)
                
            pruning_container.prune_to_cutoff(cutoff=EPSILON,
                                              min_samples=max(min_samples, 10))

        if step % args.eval_every == 0 or args.smoke_test:
            num_samples = pruning_container.points.shape[0]

            ksd = pruning_container.get_ksd_squared().sqrt().item()
            print("Step {} ksd {:.5f} samples {}".format(step, ksd,num_samples))

            results['num_evals'].append(step * args.samples_per_iter)
            results['ksd'].append(ksd)
            results['num_samples'].append(num_samples)
            results['time'].append(time.time() - t)

            ray.tune.report(num_evals=results['num_evals'][-1],
                        ksd=results['ksd'][-1],
                        num_samples=results['num_samples'][-1],
                        save_name=save_name)

            if args.save:
                utils.save.save_dict(to_save=results,
                                     save_dir=args.save_dir,
                                     save_name=save_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run grid experriments for toy problem')
    parser.add_argument('--problem',
                        type=str,
                        choices=['goodwin', 'cardiac'],
                        default='goodwin')
    parser.add_argument('--sample-generation',
                        type=str,
                        default='mala',
                        choices=['rwm', 'mala', 'tempered'])
    parser.add_argument('--samples-per-iter', type=int, default=10)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--addition-rule', type=str, default='spmcmc')
    parser.add_argument('--kernel-type',
                        type=str,
                        default='imq',
                        choices=['rbf', 'imq'])
    #parser.add_argument('--h-method', type=str, default=None)
    parser.add_argument('--full-mat', action='store_true', default=False)
    parser.add_argument('--prune', nargs=2, default=[False, None])
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--smoke-test', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

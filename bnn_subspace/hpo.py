"""
Use ray to run HPO
"""
#%%
import os
import json
import sys
import argparse
import datetime
import importlib
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule


def run_from_dict_wrapper(fun):
    def wrapper(params):
        args = argparse.Namespace()

        for key, val in params.items():
            setattr(args, key, val)

        return fun(args)

    return wrapper


parser = argparse.ArgumentParser(
    description='Run grid experriments for toy problem')
parser.add_argument('--run-target',
                    type=str,
                    help="File that contains target function as run_from_dict")
parser.add_argument('--problem-setting',
                    nargs='+',
                    type=str,
                    help="Additional problem setting information")
parser.add_argument('--grace-period',
                    type=int,
                    help="All runs go at least this far before being killed")
parser.add_argument('--max-iter',
                    type=int,
                    help="Maximum number of iterations")
parser.add_argument('--num-samples',
                    type=int,
                    help="Number of sampled configurations")
parser.add_argument('--smoke-test', action="store_true", default=False)
parser.add_argument('--resume', action="store_true", default=False)
args = parser.parse_args()

date = datetime.datetime.today().strftime("%b-%d-%Y")

save_name = "{}_{}_{}".format(args.run_target.split('.')[0], date, ''.join(args.problem_setting))

print(save_name)
save_path = os.path.join('results', args.run_target.split('.py')[0])

try:
    os.makedirs(save_path)
except Exception as e:
    print("Caught {}".format(e))

resource_config = {'gpu': 1, 'cpu': 4}

#remove filetype ending before import
run_target = importlib.import_module(args.run_target.split('.py')[0])

#wrapper for easy runs from ray
eval_fn = run_from_dict_wrapper(run_target.main)

if args.run_target == 'pretrain.py':
    config = {
        "problem_setting":args.problem_setting[0],
        "seed": 12345,
        "num_epochs": args.max_iter,
        "batch_size": 128,
        "save_every": None,
        "save_dir": None,
        "lr": tune.grid_search([1e-1, 5e-2, 1e-2]),
        "wd": tune.grid_search([1e-3, 1e-4, 1e-5]),
    }
    scheduler = None
    metric = 'accuracy'
    mode = 'max'

elif args.run_target == 'train_curve.py':
    if args.problem_setting[0]=='cifar10':
        endpoint_1 = '/data/frn_checkpoints/pretrain/epoch_299_seed_1'
        endpoint_2 =  '/data/frn_checkpoints/pretain/epoch_299_seed_2'
        num_epochs = 300

    elif args.problem_setting[0]=='imdb':
        endpoint_1 = '/data/cnn_lstm_checkpoints/pretrain/epoch_29_seed_1'
        endpoint_2 =  '/data/cnn_lstm_checkpoints/pretrain/epoch_29_seed_2'
        num_epochs = 30

    config = {
        "seed": 12345,
        "problem_setting":args.problem_setting[0],
        "endpoint_1": endpoint_1,
        "endpoint_2": endpoint_2,
        "num_epochs": num_epochs,
        "batch_size": 128,
        "save_every": None,
        "save_dir": None,
        "lr": tune.grid_search([5e-2, 1e-2, 5e-3]),
        "wd": tune.grid_search([1e-4, 1e-5]),
    }
    scheduler = None
    metric = 'accuracy'
    mode = 'max'

elif args.run_target == 'sample.py':
    
    if args.problem_setting[0] == 'cifar10':
        chkpt_path = '/data/frn_checkpoints/curve/epoch_299_seed_1'
        #reference_path = '/data/frn_checkpoints/sampling/MALA_seed_123_step_20000_Nov-01-2021'
        reference_path = tune.grid_search(['/data/frn_checkpoints/sampling/MALA_seed_12345_step_20000_Nov-11-2021','/data/frn_checkpoints/sampling/MALA_seed_123456_step_20000_Nov-11-2021'])
    elif args.problem_setting[0] == 'imdb':
        chkpt_path ='/data/cnn_lstm_checkpoints/curve/epoch_29_seed_1'
        reference_path = '/data/cnn_lstm_checkpoints/sampling/MALA_seed_123_step_20000_Nov-04-2021'

    if args.problem_setting[1] == 'hpo':
        
        config = {
            "seed": 1,
            "problem_setting": args.problem_setting[0],
            "num_steps": 1000,
            "eval_every": 50,
            "save_every": 1e10,
            "save_dir": None,
            "reference_eval": None,
            "batch_size": 256,
            "prune": (False, None),
            "full_mat": False,
            "kernel_type": 'imq',
            "samples_per_iter": 1,
            "curve_checkpoint_path": chkpt_path,
            "samples_per_iter": 1,
            "addition_rule":'std',
            "lr": tune.grid_search([1e-2, 3e-2, 1e-1, 3e-1]),
            "temperature": tune.grid_search([1e0, 1e1, 1e2, 1e3, 1e4]),
        }

    elif args.problem_setting[1] == 'paper':
        config = {
            "seed":tune.grid_search([1,2,3,4,5]),
            "problem_setting":args.problem_setting[0],
            "num_steps":2000,
            "eval_every":5,
            "save_every":1e10,
            "save_dir":None,
            "reference_eval":None,
            "batch_size":256,
            "prune":tune.grid_search([(False, None), (True, 'sqrt'),(True, 'linear')]),
            "samples_per_iter": tune.sample_from(lambda spec: 1 if spec.config.addition_rule=='std' else 5),
            "addition_rule":tune.grid_search(['std','thin','spmcmc']),
            "full_mat":False,
            "kernel_type":tune.grid_search(['imq','rbf']),
            "curve_checkpoint_path":chkpt_path,
            "reference_predictions":reference_path,
            "lr":3e-1,
            "temperature":1.0,
            "linked_dict_save_path":os.path.abspath('results/linked_dicts'),
        }

    scheduler = None
    mode = 'max'
    metric = 'bma_accuracy'

config['smoke_test'] = args.smoke_test

analysis = tune.run(
    eval_fn,
    name=save_name,
    metric=metric,
    mode=mode,
    config=config,
    resources_per_trial=resource_config,
    local_dir=".ray_data/{}".format(save_name),
    max_failures=5,
    scheduler=scheduler,
    num_samples=args.num_samples,
    resume=args.resume
)

print("Best hyperparameters: ", analysis.best_config)

df = analysis.results_df

df.to_pickle(os.path.join(save_path, save_name))

"""
Use ray to run HPO
"""
#%%
import os
import argparse
import datetime
import importlib
from ray import tune


def run_from_dict_wrapper(fun):
    def wrapper(params):
        args = argparse.Namespace()

        for key, val in params.items():
            setattr(args, key, val)

        return fun(args)

    return wrapper


parser = argparse.ArgumentParser(
    description='Run grid experiments for toy problem')
parser.add_argument('--run-target',
                    type=str,
                    help="File that contains target function as run_from_dict")
parser.add_argument('--samples-per-config',
                    type=int,
                    default=5,
                    help="Number of sampled configurations")
parser.add_argument('--num-samples',
                    type=int,
                    default=1,
                    help="Number of sampled configurations")
parser.add_argument('--problem',
                    type=str,
                    help="Problem setting",
                    choices=['alpha','modes'])
parser.add_argument('--smoke-test', action="store_true", default=False)
parser.add_argument('--resume', action="store_true", default=False)
args = parser.parse_args()


NUM_STEPS = 1000 if args.smoke_test else 100000
if args.smoke_test:
    setattr(args,'samples_per_config',2)

date = datetime.datetime.today().strftime("%b-%d-%Y")

save_name = "{}_{}".format(args.problem, date)

save_path = os.path.join('results', args.run_target.split('.py')[0])

resource_config = {'gpu': 1}

#remove filetype ending before import
run_target = importlib.import_module(args.run_target.split('.py')[0])

eval_fn = run_from_dict_wrapper(run_target.main)

if args.problem == 'alpha':

    config = {
        'modes':2,
        'num_steps':NUM_STEPS,
        'dim':2,
        'samples_per_iter':5,
        'eval_every':100,
        'addition_rule':'spmcmc',
        'kernel_type':tune.grid_search(['imq','rbf']),
        'prune':(True,'exponent',0.5),
        'alpha':tune.grid_search([1.0,1.2,1.4,1.6,1.8,2.0]),
    }
    scheduler=None

elif args.problem == 'modes':

    config = {
        'modes':tune.grid_search([2,4,6,8,10]),
        'num_steps':NUM_STEPS,
        'dim':2,
        'samples_per_iter':5,
        'eval_every':100,
        'addition_rule':'spmcmc',
        'kernel_type':tune.grid_search(['imq','rbf']),
        'prune':(True,'linear',0.0),
        'alpha':1.0,
    }
    scheduler=None


analysis = tune.run(
    eval_fn,
    name=save_name,
    metric='ksd',
    config=config,
    resources_per_trial=resource_config,
    local_dir=".ray_data/{}".format(save_name),
    max_failures=2,
    num_samples=args.samples_per_config if scheduler is None else args.num_samples,
    resume=args.resume
)


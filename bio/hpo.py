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
parser.add_argument('--num-samples',
                    type=int,
                    default=1,
                    help="Number of sampled configurations")
parser.add_argument('--problem',
                    type=str,
                    help="Problem setting",
                    choices=['goodwin', 'cardiac'])
parser.add_argument('--smoke-test', action="store_true", default=False)
args = parser.parse_args()

date = datetime.datetime.today().strftime("%b-%d-%Y")

save_name = "{}_{}".format(args.problem, date)

print(save_name)
save_path = os.path.join('results', args.run_target.split('.py')[0])

try:
    os.makedirs(save_path)
except FileExistsError:
    print("Results directory already exists")

resource_config = {'gpu': 1, 'cpu': 4}

#remove filetype ending before import
run_target = importlib.import_module(args.run_target.split('.py')[0])

eval_fn = run_from_dict_wrapper(run_target.main)

if args.problem == 'goodwin':

    config = {
        "problem": args.problem,
        "sample_generation": tune.grid_search(['rwm', 'mala']),
        "samples_per_iter": tune.sample_from(lambda spec: 1 if spec.config.addition_rule=='std' else 10),
        "eval_every": 10,
        "addition_rule": tune.grid_search(['thin','spmcmc']),
        "kernel_type": tune.grid_search(['imq','rbf']),
        #"h_method": None,
        "full_mat": False,
        "prune":   tune.grid_search([(True,'linear'),(True,'sqrt'), (False,None)]),
        "burn_in": None,
        "save": True,
        "save_dir": os.path.abspath('results/linked_dicts'),
        "smoke_test": args.smoke_test,
    }
elif args.problem == 'cardiac':

    config = {
        "problem": args.problem,
        "sample_generation": 'tempered',
        "samples_per_iter": tune.sample_from(lambda spec: 1 if spec.config.addition_rule=='std' else 100),
        "eval_every": 10,
        "addition_rule": tune.grid_search(['thin','spmcmc']),
        "kernel_type": tune.grid_search(['imq','rbf']),
        #"h_method": None,
        "full_mat": False,
        "prune":   tune.grid_search([(True,'linear'),(True,'sqrt'), (False,None)]),
        "burn_in": None,
        "save": True,
        "save_dir": os.path.abspath('results/linked_dicts'),
        "smoke_test": args.smoke_test,
    }

analysis = tune.run(
    eval_fn,
    name=save_name,
    metric='ksd',
    config=config,
    resources_per_trial=resource_config,
    local_dir=".ray_data/{}".format(save_name),
    max_failures=2,
    num_samples=1,
)

df = analysis.dataframe()

df.to_pickle(os.path.join(save_path, save_name + '.pkl'))

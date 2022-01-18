"""
Test algorithm features on toy problems
"""

import argparse
import ksdp
import ray.tune
import torch
import utils
import seaborn as sns
import matplotlib.pyplot as plt
EPSILON=0.0


def get_samples_and_grads(distribution, num_samples=1):
    """Utility wrapper function to get samples and gradients"""
    samples = torch.nn.Parameter(distribution.sample([num_samples])).cuda()
    energy_val = distribution.log_prob(samples).sum()
    energy_val.backward()

    return samples.data.clone().detach(), samples.grad.data.clone().detach()


def main(args):
    '''Test algo features on toy problems'''
    print(args)
    assert torch.cuda.is_available()

    #set up gmm
    component_means = torch.cat([torch.zeros(1,args.dim)+i for i in range(args.modes)],dim=0).cuda()
    component_scales = 0.5*torch.cat([torch.ones(1,args.dim) for _ in range(args.modes)],dim=0).cuda()
    component_distribution = torch.distributions.Independent(torch.distributions.Normal(component_means,component_scales),1)
    mix=torch.distributions.Categorical(torch.ones(args.modes,).cuda())
    distribution = torch.distributions.MixtureSameFamily(mix,component_distribution)
    """
    samples = distribution.sample([100000]).cpu().numpy()
    sns.kdeplot(samples[:,0],samples[:,1],levels=20,shade=True)
    plt.show()
    """
    pruning_container = ksdp.PruningContainer(
        kernel_type=args.kernel_type,
        h_method='dim' if args.kernel_type == 'rbf' else None,
        full_mat=False)

    init_sample, init_grad = get_samples_and_grads(distribution=distribution,
                                                   num_samples=1)

    pruning_container.add_point(point=init_sample.squeeze(), gradient=init_grad.squeeze())


    for step in range(args.num_steps):

        batch_samples, batch_gradients = get_samples_and_grads(
            distribution=distribution, num_samples=args.samples_per_iter)

        next_sample, next_gradient = utils.bayesian.select_samples(
            pruning_container=pruning_container,
            new_samples=batch_samples,
            new_gradients=batch_gradients,
            addition_rule=args.addition_rule)

        pruning_container.add_point(point=next_sample.squeeze(), gradient=next_gradient.squeeze())

        if args.prune[0]:
            min_samples = ksdp.utils.get_min_samples(growth=args.prune[1],
                                                     step=step,
                                                     coeff=args.prune[2],
                                                     exponent=args.alpha
                                                     )

            pruning_container.prune_to_cutoff(cutoff=EPSILON,
                                              min_samples=max(min_samples, 10))

        if step>0 and step % args.eval_every == 0:
            """
            import seaborn as sns
            import matplotlib.pyplot as plt
            #print(pruning_container.points)
            sns.kdeplot(x=pruning_container.points[:,0].cpu(),y=pruning_container.points[:,1].cpu(),levels=30)
            samples = distribution.sample([1000])
            sns.kdeplot(x=samples[:,0].cpu(),y=samples[:,1].cpu(),levels=30)
            plt.show()
            """
            num_samples = pruning_container.points.shape[0]

            ksd = pruning_container.get_ksd_squared().sqrt().item()
            print("Step {} ksd {:.5f} samples {}".format(
                step, ksd, num_samples))

            results={'num_evals':step*args.samples_per_iter,
                    'ksd':ksd,
                    'num_samples':num_samples}

            ray.tune.report(**results)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run grid experriments for toy problem')
    parser.add_argument('--modes', type=int, default=1)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--samples-per-iter', type=int, default=10)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--addition-rule', type=str, default='spmcmc',choices=['std','thin','spmcmc'])
    parser.add_argument('--kernel-type',
                        type=str,
                        default='imq',
                        choices=['rbf', 'imq'])
    #parser.add_argument('--h-method', type=str, default=None)
    parser.add_argument('--prune', nargs='+', default=[True, 'exponent',0.5])
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--num-steps', type=int, default=10000)
    arguments = parser.parse_args()
    main(arguments)

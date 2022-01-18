"""
Run sampling for curve model subspace
"""
import argparse
import datetime
import torch
from torch import optim
import ray.tune
import seaborn as sns
import matplotlib.pyplot as plt
import ksdp
import utils

EPSILON = 0.0
SAMPLER_TYPE = 'MALA'


def main(args):
    """Run sampling on NN curve subspace"""
    print(args)
    
    if ray.tune.is_session_enabled() and hasattr(args,'linked_dict_save_path'): 
        results_save_name = utils.save.get_random_save_name(args.linked_dict_save_path)
    else:
        results_save_name = None

    if getattr(args, 'smoke_test', False):
        print("Short run smoke test")
        setattr(args, 'num_steps', 3)

    torch.manual_seed(args.seed)
    
    #build dataloaders
    train_loader, test_loader, _, _, extras = utils.data.get_train_test_loaders(
        batch_size=args.batch_size,dataset=args.problem_setting)

    net = utils.models.get_curve_model(
        problem_setting=args.problem_setting,
        curve_checkpoint_path=args.curve_checkpoint_path,
        TEXT=extras[0] if args.problem_setting=='imdb' else None)

    net.cuda()

    #initialize from curve midpoint
    curr_sample = torch.nn.Parameter(
        torch.tensor([0.0, 1.0]).reshape(1, 2).cuda())
    
    #optimizer only used to zero out gradients
    optimizer = optim.SGD([curr_sample] + list(net.net.parameters()), lr=0.0)

    _, data = next(enumerate(train_loader))

    #add first point
    log_posterior, log_posterior_grad = utils.bayesian.get_vals_and_grads(
        net=net,
        curr_sample=curr_sample,
        optimizer=optimizer,
        data=data,
        temperature=args.temperature)


    stacked_labels = utils.bnn_metrics.get_all_labels(test_loader).cuda()

    epochs = int(args.num_steps // len(train_loader) + 1)

    pruning_container = ksdp.PruningContainer(
        kernel_type=args.kernel_type,
        h_method='dim' if args.kernel_type == 'rbf' else None,
        full_mat=args.full_mat)
    
    #container to store bayesian model averaging predictions
    bma_container = utils.bnn_metrics.BMAContainer(add_only=not args.prune[0])

    #add current sample
    curr_test_acc, preds = utils.loss.get_sample_preds_and_acc(
        curr_sample=curr_sample.detach().clone(), net=net, loader=test_loader)

    pruning_container.add_point(
        point=curr_sample.detach().clone().squeeze(),
        gradient=log_posterior_grad.detach().clone().squeeze())

    bma_container.add_pred(preds, sample=curr_sample.detach().clone())

    reference_predictions = None if getattr(args,'reference_predictions',None) is None else torch.load(
        args.reference_predictions)['predictions']

    step = 0
    results = {}
    for _ in range(epochs):
        for _, data in enumerate(train_loader):
            if step >= args.num_steps:
                return

            batch_samples = []
            batch_gradients = []

            for _ in range(int(args.samples_per_iter)):
                step += 1
                #unadjusted langevin (ULA) step
                ula_step = 0.5 * args.lr * log_posterior_grad + torch.normal(
                    mean=0.0, std=args.lr, size=curr_sample.shape).to(
                        curr_sample.device)

                curr_sample.data.add_(ula_step)
                
                #compute current sample energy value and gradient
                log_posterior, log_posterior_grad = utils.bayesian.get_vals_and_grads(
                    net=net,
                    curr_sample=curr_sample,
                    optimizer=optimizer,
                    data=data,
                    temperature=args.temperature)

                batch_samples.append(curr_sample.detach().clone().squeeze())
                batch_gradients.append(
                    log_posterior_grad.detach().clone().squeeze())


            batch_samples = torch.stack(batch_samples)
            batch_gradients = torch.stack(batch_gradients)

            next_sample, next_gradient = utils.bayesian.select_samples(
                pruning_container=pruning_container,
                new_samples=batch_samples,
                new_gradients=batch_gradients,
                addition_rule=args.addition_rule)

            #add current sample
            curr_test_acc, preds = utils.loss.get_sample_preds_and_acc(
                curr_sample=next_sample.unsqueeze(0), net=net, loader=test_loader)

            pruning_container.add_point(
                point=next_sample.detach().clone().squeeze(),
                gradient=next_gradient.detach().clone().squeeze())

            bma_container.add_pred(preds, sample=next_sample.detach().clone())

            if args.prune[0]:
                min_samples = ksdp.utils.get_min_samples(growth=args.prune[1],
                                                         step=int(step/args.samples_per_iter))

                pruned_samples = pruning_container.prune_to_cutoff(
                    cutoff=EPSILON, min_samples=max(min_samples, 5))

                for sample in pruned_samples:
                    bma_container.drop_sample(sample)

            save_this_iter = (step % args.save_every == 0 or step == (
                args.num_steps - 1)) and args.save_dir is not None

            eval_this_iter = getattr(
                args, 'smoke_test',
                False) or step % args.eval_every == 0 or save_this_iter

            if eval_this_iter:
                #compute relevant metrics
                metrics = utils.bnn_metrics.get_metrics(
                    bma_pred=bma_container.bma_pred,
                    stacked_labels=stacked_labels,
                    pruning_container=pruning_container)
                metrics['num_evals']=step

                print("Step {}".format(step))
                print("\tACC: {:.3f}".format(curr_test_acc).rjust(15))
                print("\tLOG-P: {:.3f}".format(log_posterior).rjust(15))
                print("\tBMA ACC: {:.3f}".format(
                    metrics['bma_accuracy']).rjust(15))
                print("\tBMA ECE: {:.3f}".format(metrics['bma_ece']).rjust(15))
                print("\tKSD: {:.3f}".format(metrics['ksd']).rjust(15))
                print("\tNum samples: {}".format(
                    metrics['num_samples']).rjust(15))

                if getattr(args,'reference_predictions',None) is not None:
                    reference_metrics = utils.bnn_metrics.get_reference_metrics(
                        bma_pred=bma_container.bma_pred,
                        reference_predictions=reference_predictions)

                    metrics = {**metrics, **reference_metrics}
                    print("\tBMA TV: {:.3f}".format(
                        metrics['bma_tv']).rjust(15))
                    print("\tBMA Agree: {:.3f}".format(
                        metrics['bma_agreement']).rjust(15))
                """
                to_plot = pruning_container.points.cpu().numpy()
                plt.figure()
                sns.scatterplot(x=to_plot[:, 0], y=to_plot[:, 1])
                plt.savefig('tmp_{}.png'.format(args.seed))
                plt.close()
                """
                print("Saving numerical results")
                #if running ray session, save the results dict

                if ray.tune.is_session_enabled():
                    ray.tune.report(**metrics, save_name=results_save_name)

                    if hasattr(args,'linked_dict_dave_path'):
                        for key, val in metrics.items():
                            if key in results:
                                results[key].append(val)
                            else:
                                results[key] = [val]
                        utils.save.save_dict(to_save=results,
                                             save_dir=args.linked_dict_save_path,
                                             save_name=results_save_name)

                #save on interval and at last step
                if save_this_iter:
                    date = datetime.datetime.today().strftime("%b-%d-%Y")
                    checkpoint_save_name = "{}_seed_{}_step_{}_{}".format(
                        SAMPLER_TYPE, args.seed, step, date)
                    print("Saving predictions")
                    utils.save.save_checkpoint(
                        checkpoint_dir=args.save_dir,
                        checkpoint_name=checkpoint_save_name,
                        args=args,
                        predictions=bma_container.bma_pred)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--problem-setting', type=str, default='cifar10')
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=1e10)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-1)
    parser.add_argument('--kernel-type',
                        type=str,
                        default='imq',
                        choices=['rbf', 'imq'])
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--save-dir',
                        type=str,
                        default='/data/frn_checkpoints/sampling')
    parser.add_argument('--curve-checkpoint-path',
                        type=str,
                        default='/data/frn_checkpoints/curve/epoch_299_seed_1')
    parser.add_argument('--full-mat', action='store_true', default=False)
    parser.add_argument('--prune', nargs=2, default=[False, None])
    parser.add_argument('--reference-predictions', type=str, default=None)
    parser.add_argument('--addition-rule', type=str, default='std')
    parser.add_argument('--samples-per-iter', type=int, default=1)
    parser.add_argument('--smoke-test', action='store_true', default=False)

    args = parser.parse_args()

    main(args)

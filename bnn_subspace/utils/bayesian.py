import torch
import utils


def get_vals_and_grads(net,curr_sample,optimizer,data,temperature):
    """Wrapper to get energy value and gradient"""
    net.eval()
    net.update_weights(vec=curr_sample)
    optimizer.zero_grad()
    log_likelihood = utils.loss.log_likelihood_fn(
        net=net, batch=data,
        mean_reduce=True) * (1.0 / temperature)
    log_likelihood.backward()
    log_likelihood_grad = net.get_proj_grad()

    optimizer.zero_grad()
    log_prior = utils.loss.gaussian_log_prior(curr_sample)
    log_prior.backward()
    log_prior_grad = curr_sample.grad

    log_posterior = log_likelihood + log_prior

    log_posterior_grad = log_likelihood_grad + log_prior_grad

    return log_posterior, log_posterior_grad

def select_samples(pruning_container,new_samples,new_gradients,addition_rule):

    if addition_rule=='std':
        index = 0 
    elif addition_rule=='thin':
        index=-1
    elif addition_rule=='spmcmc':

        index = pruning_container.best_index(candidate_points=new_samples, candidate_gradients=new_gradients)

    return new_samples[index],new_gradients[index]

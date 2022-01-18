"""
Handles log-posterior, log-likelihood, log-prior and accuracy computations
"""
import math
import torch
import torch.nn.functional as F
import utils

PRIOR_VARIANCE = 10.0

def get_sample_preds_and_acc(curr_sample, net, loader):
    """Utility function to collect predictions"""
    net.update_weights(vec=curr_sample)
    test_acc, preds = utils.loss.evaluate_fn(loader, net)

    return test_acc, preds

def log_likelihood_fn(net, batch, mean_reduce=False):
    """Computes the log-likelihood."""
    x, y = batch

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    logits = net(x)
    num_classes = logits.shape[-1]
    labels = F.one_hot(y.to(torch.int64), num_classes=num_classes)
    softmax_xent = torch.sum(labels * F.log_softmax(logits, dim=1))
    if mean_reduce:
        return (1.0/x.shape[0])*softmax_xent
    else:
        return softmax_xent

def gaussian_log_prior(sample):
    """Computes the Gaussian prior log-density."""
    n_params = sample.numel()
    exp_term = (-sample**2 / (2 * PRIOR_VARIANCE)).sum()
    norm_constant = -0.5 * n_params * math.log((2 * math.pi * PRIOR_VARIANCE))
    return exp_term + norm_constant
"""
def log_prior_fn(net):
    Computes the Gaussian prior log-density.
    model_state_dict = net.state_dict()
    n_params = sum(p.numel() for p in model_state_dict.values())
    exp_term = sum((-p**2 / (2 * PRIOR_VARIANCE)).sum()
                   for p in model_state_dict.values())
    norm_constant = -0.5 * n_params * math.log((2 * math.pi * PRIOR_VARIANCE))
    return exp_term + norm_constant
"""

def log_posterior_fn(net, batch):
    """Compute the log posterior"""
    log_lik = log_likelihood_fn(net=net, batch=batch)
    log_prior = log_prior_fn(net=net)
    return log_lik + log_prior


def get_accuracy_fn(net, batch):
    """Computes accuracy of a batch"""
    x, y = batch
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    # get logits
    net.eval()
    with torch.no_grad():
        logits = net(x)

    net.train()
    # get log probs
    log_probs = F.log_softmax(logits, dim=1)
    # get preds
    probs = torch.exp(log_probs)
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == y).float().mean()

    return accuracy, probs


def evaluate_stacked(data_loader, stacked_preds):
    """Compute accuracy given all test set predictions"""

    total_examples = 0
    sum_correct = 0
    index = 0
    for _, y in data_loader:
        batch_num_examples = y.shape[0]
        total_examples += batch_num_examples
        preds = stacked_preds[index:index + batch_num_examples].to(y.device)
        top_1_preds = torch.argmax(preds, dim=1)
        batch_correct = (top_1_preds == y).float().sum().item()
        sum_correct += batch_correct
        index += batch_num_examples

    return sum_correct / total_examples


def evaluate_fn(data_loader, net):
    """Compute accuracy and return all probabilities used to compute accuracy"""

    sum_accuracy = 0
    all_probs = []
    for x, y in data_loader:
        batch_accuracy, batch_probs = get_accuracy_fn(net=net, batch=(x, y))
        sum_accuracy += batch_accuracy.item()
        all_probs.append(batch_probs)
    all_probs = torch.cat(all_probs, dim=0)
    return sum_accuracy / len(data_loader), all_probs

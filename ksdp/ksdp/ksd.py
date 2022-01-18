#%%
import torch

def _row_get_pairwise_distances(base_point, A):
    """Computes the n-dimensional vector of pairwise euclidean distances between point at index i and other points in A.

    arguments:
    base_point -- relevant point for row
    A -- n x d matrix of points
    """
    return torch.pow(base_point - A,2).sum(axis=1)

def med_h(x):
    raise NotImplementedError
    pdist = _pairwise_distance(x)

    # https://stackoverflow.com/questions/43824665/tensorflow-median-value
    lower = tfp.stats.percentile(pdist, 50.0, interpolation='lower')
    higher = tfp.stats.percentile(pdist, 50.0, interpolation='higher')

    median = (lower + higher) / 2.
    median = tf.cast(median, tf.float32)

    return median

def get_K_row(samples, gradients, kernel_type, h, index=-1):
    #assume last row is the new row
    if torch.is_tensor(index):
        index = index.squeeze()#index.detach().item()
    
    base_point = samples[index]

    pdists = _row_get_pairwise_distances(base_point, samples)

    pdiffs = base_point.unsqueeze(0) - samples

    if kernel_type == 'rbf':
        kernel_values = torch.exp(-pdists / h)
        kernel_grads = (2 / h) * torch.multiply(kernel_values.unsqueeze(1), pdiffs)
        jacobian_values = -((2 / h)**2) * (pdiffs**2) * kernel_values.unsqueeze(1) + (2 / h) * kernel_values.unsqueeze(1)

    elif kernel_type == 'imq':

        beta = -0.5
        kernel_values = torch.pow(pdists + 1, beta)
        kernel_grads = -(kernel_values / (1 + pdists)).unsqueeze(1) * beta * 2 * pdiffs

        jacobian_values = -(2 * beta) * torch.pow(1 + pdists, beta - 1).unsqueeze(1) - 4 * beta * (beta - 1) * (
                pdiffs**2) * torch.pow(1 + pdists, beta - 2).unsqueeze(1)

    a = (gradients[index] * gradients).sum(1) * kernel_values

    b = -(gradients * kernel_grads).sum(1)

    c = (kernel_grads * gradients[index]).sum(1)

    d = jacobian_values.sum(1)
    """
    print(a)
    print(b)
    print(c)
    print(d)
    """
    row = a + b + c + d

    return row

def _get_h(samples,h_method):
    if h_method == 'dim':
        h = samples.shape[1]
    elif h_method == 'med':
        h = _med_h(samples)
    else:
        raise NotImplementedError("Bandwith method {} not supported".format(h_method))
    return h

def get_K_matrix(samples,
                 gradients,
                 kernel_type,
                 h_method):

    rows = []

    h = _get_h(samples=samples,h_method=h_method) if kernel_type=='rbf' else None

    for i in range(samples.shape[0]):

        rows.append(get_K_row(samples=samples[:i+1], gradients=gradients[:i+1], kernel_type=kernel_type, h=h))
   

    K_mat = rows[0].reshape(1,1)
    for i,row in enumerate(rows[1:]):
        K_mat = torch.cat([K_mat,row[:i+1].unsqueeze(0)])
        K_mat = torch.cat([K_mat,row.unsqueeze(1)],dim=1)

    return K_mat


def get_KSD(samples,
            gradients,
            kernel_type,
            h_method):
    """Computes the Kernelized Stein Discrepancy 
    
    """

    num_samples = samples.shape[0]

    K = get_K_matrix(samples=samples,
                     gradients=gradients,
                     kernel_type=kernel_type,
                     h_method=h_method)

    return K.sum().sqrt() / num_samples

def get_sequential_KSDs(samples,
                        gradients,
                        kernel_type,
                        h_method):

    K= get_K_matrix(samples=samples,
                 gradients=gradients,
                 kernel_type=kernel_type,
                 h_method=h_method)

    K_sums = 2.0*torch.tril(K).sum(dim=1)-K.diag()

    cum_sums = torch.cumsum(K_sums,dim=0)

    return [x for x in cum_sums.sqrt().divide(torch.arange(1,cum_sums.shape[0]+1).to(cum_sums.device))]

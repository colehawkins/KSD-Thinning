#Bayesian sampling and thinning utilities

def select_samples(pruning_container,new_samples,new_gradients,addition_rule):

    if addition_rule=='std':
        index = 0 
    elif addition_rule=='thin':
        index=-1
    elif addition_rule=='spmcmc':

        index = pruning_container.best_index(candidate_points=new_samples, candidate_gradients=new_gradients)

    return new_samples[index],new_gradients[index]

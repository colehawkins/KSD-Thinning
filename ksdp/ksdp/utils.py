import math

def get_min_samples(growth,step,exponent=1.0, coeff=1.0):

    if growth == 'linear':
        min_samples = step / 2.0
    elif growth == 'exponent':
        if exponent>(2.0-1e-10):
            min_samples = step/2.0
        else:
            min_samples = math.sqrt((step**(exponent)) * max(math.log(step + 1.0), 1.0))
    elif growth == 'sqrt':
        min_samples = math.sqrt(step * max(math.log(step + 1.0), 1.0))
    else: 
        raise NotImplementedError("Dictionary growth style {} not implemented".format(growth))

    return min_samples*coeff


import numpy as np

def topp_prot(u, v):
    '''
    given a list of predictions (i.e. scores) what is the probability of being at the top position
    for group u out of all items
    example: what is the probability of each female (or male respectively) item (u) to be in position 1

    FIXME: rewrite this such that finding the items per group happens here instead of somewhere else
    and that we pass only the whole dataset to this function
    '''
    return np.exp(u / np.sum(np.exp(v)))

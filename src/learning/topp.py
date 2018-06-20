import numpy as np

def topp(v):
    return np.exp(v)/np.sum(np.exp(v))
